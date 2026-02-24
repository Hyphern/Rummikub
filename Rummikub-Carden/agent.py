"""Agent definitions for Rummikub ML training.

Provides base agent class and several example agents.
"""

import numpy as np
import random
import json
import os

# Resolve paths relative to this script's directory, not cwd
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod


class RummikubAgent(ABC):
    """Base class for Rummikub agents."""
    
    def __init__(self, name: str = "Agent"):
        self.name = name
        self.stats = {
            'games_played': 0,
            'wins': 0,
            'total_reward': 0.0,
            'moves_made': 0,
        }
    
    @abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray], 
                     valid_actions: np.ndarray) -> int:
        """Select an action given the current observation.
        
        Args:
            observation: Current game state
            valid_actions: Binary mask of valid actions
            
        Returns:
            Selected action index
        """
        pass
    
    def reset(self):
        """Reset agent state for a new game."""
        pass
    
    def update_stats(self, won: bool, reward: float, moves: int):
        """Update agent statistics after a game."""
        self.stats['games_played'] += 1
        if won:
            self.stats['wins'] += 1
        self.stats['total_reward'] += reward
        self.stats['moves_made'] += moves
    
    def get_win_rate(self) -> float:
        """Get win rate as percentage."""
        if self.stats['games_played'] == 0:
            return 0.0
        return (self.stats['wins'] / self.stats['games_played']) * 100
    
    def __repr__(self):
        return f"{self.name}(win_rate={self.get_win_rate():.1f}%)"


class SmartWormAgent(RummikubAgent):
    """Agent that uses worm.py solver logic intelligently.
    
    Strategy:
    1. Initial meld: Find groups/runs that sum to 15+ points
    2. After initial: Try to add groups directly from hand to table
    3. Use worm: Check if tiles can be added to board (runs/groups)
    4. If can add: Restart algorithm (order-independent - key insight!)
    """
    
    DEBUG = False  # Set to True for verbose logging
    
    def __init__(self, name: str = "SmartWorm"):
        super().__init__(name)
        self._import_worm()
    
    def _import_worm(self):
        """Import worm functions."""
        try:
            from worm_integration import (
                hand_to_board_matrix,
                table_to_board_matrix,
            )
            
            self.worm_available = True
            self.hand_to_board_matrix = hand_to_board_matrix
            self.table_to_board_matrix = table_to_board_matrix
        except ImportError as e:
            self.worm_available = False
            print(f"Warning: worm_integration not available: {e}")
    
    def select_action(self, observation: Dict[str, np.ndarray],
                     valid_actions: np.ndarray) -> int:
        """Select action using worm solver logic."""
        valid_indices = np.where(valid_actions == 1)[0]
        if len(valid_indices) == 0:
            return 0
        
        # Always win if possible
        if 1 in valid_indices:
            return 1
        
        if not self.worm_available:
            # Fall back to simple heuristic
            meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
            if meld_actions:
                return random.choice(meld_actions)
            return 0 if 0 in valid_indices else random.choice(valid_indices)
        
        # Decode game state
        hand = self._decode_hand(observation)
        table = self._decode_table(observation)
        has_initial = observation['has_initial_meld'][0] > 0
        
        # Run the iterative algorithm
        action = self._find_best_action(hand, table, has_initial, valid_indices)
        
        return action if action is not None else (0 if 0 in valid_indices else random.choice(valid_indices))
    
    def _find_best_action(self, hand: List[Any], table: List[Any], 
                         has_initial: bool, valid_indices: np.ndarray) -> Optional[int]:
        """Find the best action.
        
        Strategy:
        1. Initial meld: Find and play meld with 15+ points if needed
        2. After initial: Use wormed to find+apply definite moves on
           combined board, then play any hand tiles consumed by those,
           then play any additional melds from hand, then add to table melds
        3. Fallback: Scan valid_indices directly for any remaining playable
           actions (PLAY_NEW_MELD or ADD_TO_MELD) that Steps 2a-2c missed
        4. Last resort: pick up
        """
        import wormed
        from meld import find_all_valid_melds, Meld
        from copy import deepcopy
        
        if self.DEBUG: print(f"  [SmartWorm] hand={len(hand)} tiles, table={len(table)} melds, has_initial={has_initial}", flush=True)
        
        # Compute valid melds from hand ONCE for the whole turn
        cached_melds = find_all_valid_melds(hand)
        
        # ── Step 1: Initial meld — must play 15+ point meld first ──
        if not has_initial:
            if self.DEBUG: print(f"  [SmartWorm] Initial meld search: {len(cached_melds)} valid melds", flush=True)
            # Sort by value descending so we play the best initial meld
            cached_melds.sort(key=lambda m: Meld.calculate_value(m), reverse=True)
            for meld_tiles in cached_melds:
                if Meld.calculate_value(meld_tiles) >= 15:
                    action = self._meld_to_action(meld_tiles, hand, valid_indices)
                    if action is not None:
                        if self.DEBUG: print(f"  [SmartWorm] Playing initial meld worth {Meld.calculate_value(meld_tiles)}", flush=True)
                        return action
            # Can't make initial meld — must draw
            if 0 in valid_indices:
                if self.DEBUG: print(f"  [SmartWorm] Drawing (no initial meld)", flush=True)
                return 0
            return None
        
        # ── Step 2: Definite tile removal + hand melds ──
        table_board = self.table_to_board_matrix(table)
        hand_board = self.hand_to_board_matrix(hand)
        combined_board = [[table_board[r][c] + hand_board[r][c]
                          for c in range(13)] for r in range(4)]
        
        # Iteratively apply ALL definite moves on combined board
        board_after = deepcopy(combined_board)
        all_definite = []
        while True:
            moves = wormed.find_explicit_moves(board_after)
            if not moves:
                break
            for move in moves:
                all_definite.append(move)
                board_after = wormed.apply_move(board_after, move)
        
        if self.DEBUG: print(f"  [SmartWorm] Found {len(all_definite)} definite moves on combined board", flush=True)
        
        # Figure out which hand tiles are consumed by definite moves
        remaining_hand = deepcopy(hand_board)
        tiles_to_play = []  # (row, col) of hand tiles consumed
        for move in all_definite:
            if move[0] == 'run':
                _, row, start, end = move
                for col in range(start, end + 1):
                    if remaining_hand[row][col] > 0:
                        remaining_hand[row][col] -= 1
                        tiles_to_play.append((row, col))
            elif move[0] == 'group':
                _, col, colors = move
                for row in colors:
                    if remaining_hand[row][col] > 0:
                        remaining_hand[row][col] -= 1
                        tiles_to_play.append((row, col))
        
        # Play hand tiles consumed by definite moves (one at a time, env will re-call us)
        if tiles_to_play:
            if self.DEBUG: print(f"  [SmartWorm] Definite moves consume {len(tiles_to_play)} hand tiles", flush=True)
            for tile_pos in tiles_to_play:
                action = self._play_tile_from_hand(tile_pos, hand, table, valid_indices, cached_melds)
                if action is not None:
                    return action
        
        # Play any direct melds from hand (prefer larger melds)
        if cached_melds:
            sorted_melds = sorted(cached_melds, key=lambda m: len(m), reverse=True)
            for meld_tiles in sorted_melds:
                action = self._meld_to_action(meld_tiles, hand, valid_indices)
                if action is not None:
                    if self.DEBUG: print(f"  [SmartWorm] Playing meld from hand ({len(meld_tiles)} tiles)", flush=True)
                    return action
        
        # Try adding individual tiles to existing table melds
        add_action = self._find_add_to_meld_action(hand, table, valid_indices)
        if add_action is not None:
            if self.DEBUG: print(f"  [SmartWorm] Adding tile to existing meld", flush=True)
            return add_action
        
        # ── Step 3: Direct valid_indices scan ──
        # Safety net: scan all remaining valid actions in case Steps 2a-2c
        # missed something due to tile matching or index edge cases.
        # Prefer PLAY_NEW_MELD (plays more tiles) over ADD_TO_MELD.
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            if self.DEBUG: print(f"  [SmartWorm] Fallback: found {len(meld_actions)} valid PLAY_NEW_MELD actions", flush=True)
            return meld_actions[0]
        
        add_base_idx = 2 + 30
        add_actions = [a for a in valid_indices if a >= add_base_idx]
        if add_actions:
            if self.DEBUG: print(f"  [SmartWorm] Fallback: found {len(add_actions)} valid ADD_TO_MELD actions", flush=True)
            return add_actions[0]
        
        # ── Step 4: Last resort — pick up ──
        if 0 in valid_indices:
            if self.DEBUG: print(f"  [SmartWorm] Drawing (last resort)", flush=True)
            return 0
        
        return None
    
    def _play_tile_from_hand(self, tile_pos: tuple, hand: List[Any], 
                             table: List[Any], valid_indices: np.ndarray,
                             cached_melds: Optional[List] = None) -> Optional[int]:
        """Try to play a hand tile identified by (row, col) board position.
        
        Tries adding to existing table meld first, then forming a new meld.
        """
        from meld import Meld, find_all_valid_melds
        row, col = tile_pos
        row_to_color = {0: 'red', 1: 'blue', 2: 'black', 3: 'orange'}
        target_color = row_to_color.get(row)
        target_number = col + 1
        
        if target_color is None:
            return None
        
        # Find this tile in hand
        for tile_idx, tile in enumerate(hand):
            if (not tile.is_joker and tile.color == target_color 
                    and tile.number == target_number):
                # Try adding to existing table melds
                add_base_idx = 2 + 30
                for meld_idx, meld in enumerate(table):
                    if Meld.can_add_tile(meld.tiles, tile):
                        action_idx = add_base_idx + tile_idx * 40 + meld_idx
                        if action_idx in valid_indices:
                            return action_idx
                
                # Try forming a new meld containing this tile
                melds = cached_melds if cached_melds is not None else find_all_valid_melds(hand)
                for meld_tiles in melds:
                    if tile in meld_tiles:
                        action = self._meld_to_action(meld_tiles, hand, valid_indices)
                        if action is not None:
                            return action
                break  # Only try first matching tile in hand
        
        return None
    
    
    def _find_add_to_meld_action(self, hand: List[Any], table: List[Any], 
                                 valid_indices: np.ndarray) -> Optional[int]:
        """Try to add any hand tile to an existing table meld."""
        from meld import Meld
        
        add_base_idx = 2 + 30
        
        for tile_idx, tile in enumerate(hand):
            for meld_idx, meld in enumerate(table):
                if Meld.can_add_tile(meld.tiles, tile):
                    action_idx = add_base_idx + tile_idx * 40 + meld_idx
                    if action_idx in valid_indices:
                        return action_idx
        
        return None
    
    def _meld_to_action(self, meld_tiles: List[Any], hand: List[Any],
                        valid_indices: np.ndarray) -> Optional[int]:
        """Convert a meld (list of tiles) to an action index.
        
        Action encoding: action = 2 + tile_hand_index.
        The env finds a valid meld containing that tile.
        """
        # Find indices of meld tiles in hand
        tile_indices = []
        used = set()
        for tile in meld_tiles:
            for idx, hand_tile in enumerate(hand):
                if tile == hand_tile and idx not in used:
                    tile_indices.append(idx)
                    used.add(idx)
                    break
        
        if len(tile_indices) < 3:
            return None
        
        # Try each tile in the meld as the action trigger
        for idx in tile_indices:
            action_idx = 2 + idx
            if action_idx in valid_indices:
                return action_idx
        
        return None
    
    def _decode_hand(self, observation: Dict[str, np.ndarray]) -> List[Any]:
        """Decode hand tiles from observation."""
        from tile import Tile
        hand = []
        for i, tile_code in enumerate(observation['hand']):
            if observation['hand_mask'][i] > 0:
                try:
                    hand.append(Tile.decode(int(tile_code)))
                except:
                    pass
        return hand
    
    def _decode_table(self, observation: Dict[str, np.ndarray]) -> List[Any]:
        """Decode table melds from observation."""
        from meld import Meld
        from tile import Tile
        
        melds = []
        for meld_idx in range(40):
            tiles = []
            for tile_idx in range(13):
                if observation['table_mask'][meld_idx, tile_idx] > 0:
                    try:
                        tile_code = int(observation['table'][meld_idx, tile_idx])
                        tiles.append(Tile.decode(tile_code))
                    except:
                        pass
            if len(tiles) >= 3:
                try:
                    melds.append(Meld(tiles))
                except:
                    pass
        return melds


class WeightedWormAgent(SmartWormAgent):
    """Configurable SmartWorm agent with tunable strategy weights.
    
    Inherits all solver logic from SmartWormAgent but adds weight parameters
    that control decision-making at each step.
    
    Weights:
        aggressiveness (0.0-1.0): How eagerly to play tiles.
            0.0 = hoard tiles, only play when hand is large or moves are certain
            1.0 = play anything valid immediately
        meld_size_preference (0.0-1.0): Preferred meld size when choosing.
            0.0 = prefer small melds (3 tiles) to get tiles out fast
            1.0 = prefer large melds for bigger impact
        draw_preference (0.0-1.0): Tendency to draw instead of playing.
            0.0 = never draw if any play action exists
            1.0 = strongly prefer drawing to accumulate tiles
    """
    
    def __init__(self, name: str = "WeightedWorm",
                 aggressiveness: float = 0.5,
                 meld_size_preference: float = 0.5,
                 draw_preference: float = 0.3):
        super().__init__(name)
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))
        self.meld_size_preference = max(0.0, min(1.0, meld_size_preference))
        self.draw_preference = max(0.0, min(1.0, draw_preference))
    
    def _find_best_action(self, hand: List[Any], table: List[Any],
                         has_initial: bool, valid_indices: np.ndarray) -> Optional[int]:
        """Find the best action using weighted decision logic.
        
        Same 4-step structure as SmartWormAgent but weights influence:
        - Which melds to prefer (size sorting)
        - Whether to play marginal tiles or hold them
        - Whether to draw instead of playing
        """
        import wormed
        from meld import find_all_valid_melds, Meld
        from copy import deepcopy
        
        if self.DEBUG:
            print(f"  [{self.name}] hand={len(hand)}, table={len(table)}, "
                  f"agg={self.aggressiveness}, meld_pref={self.meld_size_preference}, "
                  f"draw_pref={self.draw_preference}", flush=True)
        
        cached_melds = find_all_valid_melds(hand)
        
        # ── Step 1: Initial meld (mandatory, but weight affects which one) ──
        if not has_initial:
            valid_initial = [m for m in cached_melds if Meld.calculate_value(m) >= 15]
            if not valid_initial:
                return 0 if 0 in valid_indices else None
            
            # meld_size_preference: high = play biggest initial, low = play smallest valid
            if self.meld_size_preference > 0.5:
                valid_initial.sort(key=lambda m: Meld.calculate_value(m), reverse=True)
            else:
                valid_initial.sort(key=lambda m: Meld.calculate_value(m))
            
            for meld_tiles in valid_initial:
                action = self._meld_to_action(meld_tiles, hand, valid_indices)
                if action is not None:
                    return action
            return 0 if 0 in valid_indices else None
        
        # ── Step 2: Definite tile removal (always apply — these are certain) ──
        table_board = self.table_to_board_matrix(table)
        hand_board = self.hand_to_board_matrix(hand)
        combined_board = [[table_board[r][c] + hand_board[r][c]
                          for c in range(13)] for r in range(4)]
        
        board_after = deepcopy(combined_board)
        all_definite = []
        while True:
            moves = wormed.find_explicit_moves(board_after)
            if not moves:
                break
            for move in moves:
                all_definite.append(move)
                board_after = wormed.apply_move(board_after, move)
        
        # Identify hand tiles consumed by definite moves
        remaining_hand = deepcopy(hand_board)
        tiles_to_play = []
        for move in all_definite:
            if move[0] == 'run':
                _, row, start, end = move
                for col in range(start, end + 1):
                    if remaining_hand[row][col] > 0:
                        remaining_hand[row][col] -= 1
                        tiles_to_play.append((row, col))
            elif move[0] == 'group':
                _, col, colors = move
                for row in colors:
                    if remaining_hand[row][col] > 0:
                        remaining_hand[row][col] -= 1
                        tiles_to_play.append((row, col))
        
        # Always play definite tiles (these are mathematically certain)
        if tiles_to_play:
            for tile_pos in tiles_to_play:
                action = self._play_tile_from_hand(tile_pos, hand, table, valid_indices, cached_melds)
                if action is not None:
                    return action
        
        # ── Weight gate: should we play optional tiles or draw? ──
        hand_size = len(hand)
        
        # play_threshold: how many tiles before we're willing to play optional moves
        # aggressiveness=1.0 -> threshold=0 (always play)
        # aggressiveness=0.0 -> threshold=12 (only play when hand is very full)
        play_threshold = int((1.0 - self.aggressiveness) * 12)
        
        # draw_preference creates a chance to skip optional plays and draw instead
        # Only applies when hand is below threshold
        should_draw_instead = (hand_size < play_threshold
                               and 0 in valid_indices
                               and random.random() < self.draw_preference)
        
        if should_draw_instead:
            if self.DEBUG:
                print(f"  [{self.name}] Drawing (hand={hand_size} < threshold={play_threshold}, "
                      f"draw_pref={self.draw_preference})", flush=True)
            return 0
        
        # ── Play optional melds from hand (weighted by meld_size_preference) ──
        if cached_melds and hand_size >= play_threshold:
            if self.meld_size_preference > 0.5:
                sorted_melds = sorted(cached_melds, key=lambda m: len(m), reverse=True)
            else:
                sorted_melds = sorted(cached_melds, key=lambda m: len(m))
            
            for meld_tiles in sorted_melds:
                action = self._meld_to_action(meld_tiles, hand, valid_indices)
                if action is not None:
                    return action
        
        # Try adding tiles to existing table melds
        if hand_size >= play_threshold:
            add_action = self._find_add_to_meld_action(hand, table, valid_indices)
            if add_action is not None:
                return add_action
        
        # ── Step 3: Fallback scan ──
        if hand_size >= play_threshold:
            meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
            if meld_actions:
                return meld_actions[0]
            
            add_base_idx = 2 + 30
            add_actions = [a for a in valid_indices if a >= add_base_idx]
            if add_actions:
                return add_actions[0]
        
        # ── Step 4: Draw ──
        if 0 in valid_indices:
            return 0
        
        # Absolute fallback: play anything valid even if below threshold
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            return meld_actions[0]
        add_base_idx = 2 + 30
        add_actions = [a for a in valid_indices if a >= add_base_idx]
        if add_actions:
            return add_actions[0]
        
        return None


# ── Named Presets ──
# Each creates a WeightedWormAgent with a distinct playstyle.

def create_hoarder(name: str = "Hoarder") -> WeightedWormAgent:
    """Holds tiles, draws aggressively, only plays when hand is large."""
    return WeightedWormAgent(
        name=name,
        aggressiveness=0.1,
        meld_size_preference=0.9,
        draw_preference=0.8,
    )


def create_aggressive(name: str = "Aggressive") -> WeightedWormAgent:
    """Plays tiles ASAP, prefers small fast melds, rarely draws."""
    return WeightedWormAgent(
        name=name,
        aggressiveness=1.0,
        meld_size_preference=0.1,
        draw_preference=0.0,
    )


def create_strategic(name: str = "Strategic") -> WeightedWormAgent:
    """Balanced play — waits for medium-large melds, moderate draw tendency."""
    return WeightedWormAgent(
        name=name,
        aggressiveness=0.5,
        meld_size_preference=0.7,
        draw_preference=0.3,
    )


def create_greedy(name: str = "Greedy") -> WeightedWormAgent:
    """Always plays the biggest available meld, never draws if can play."""
    return WeightedWormAgent(
        name=name,
        aggressiveness=0.9,
        meld_size_preference=1.0,
        draw_preference=0.0,
    )


def create_cautious(name: str = "Cautious") -> WeightedWormAgent:
    """Plays conservatively, prefers small safe melds, moderate hoarding."""
    return WeightedWormAgent(
        name=name,
        aggressiveness=0.3,
        meld_size_preference=0.2,
        draw_preference=0.5,
    )

class AdaptiveWormAgent(WeightedWormAgent):
    """Weighted agent that updates its own weights after every game.
    
    Uses a win/loss gradient approach with decaying learning rate:
    - On WIN:  Small reinforcement — nudge weights toward best-known values
    - On LOSS: Larger perturbation — shift weights with random exploration + drift toward best
    - Learning rate decays over games: lr = initial_lr / (1 + decay * games)
    """
    
    def __init__(self, name: str = "Adaptive",
                 aggressiveness: float = 0.5,
                 meld_size_preference: float = 0.5,
                 draw_preference: float = 0.3,
                 initial_lr: float = 0.15,
                 lr_decay: float = 0.05,
                 win_momentum: float = 0.3,
                 loss_perturbation: float = 1.0,
                 save_file: Optional[str] = os.path.join(_SCRIPT_DIR, "adaptive_weights.json")):
        super().__init__(name, aggressiveness, meld_size_preference, draw_preference)
        self.initial_lr = initial_lr
        self.lr_decay = lr_decay
        self.win_momentum = win_momentum
        self.loss_perturbation = loss_perturbation
        self.save_file = save_file
        
        # Track best-known weights (updated when we win)
        self.best_weights = self._get_weights()
        self.best_win_streak = 0
        self.current_streak = 0
        
        # History for analysis
        self.weight_history: List[dict] = []
        
        # Load persisted state if available
        if self.save_file:
            self._load_state()
    
    def _get_weights(self) -> dict:
        return {
            'aggressiveness': self.aggressiveness,
            'meld_size_preference': self.meld_size_preference,
            'draw_preference': self.draw_preference,
        }
    
    def _set_weights(self, weights: dict):
        self.aggressiveness = max(0.0, min(1.0, weights['aggressiveness']))
        self.meld_size_preference = max(0.0, min(1.0, weights['meld_size_preference']))
        self.draw_preference = max(0.0, min(1.0, weights['draw_preference']))
    
    def _save_state(self):
        """Persist current weights and learning state to JSON file."""
        if not self.save_file:
            return
        state = {
            'weights': self._get_weights(),
            'best_weights': self.best_weights,
            'best_win_streak': self.best_win_streak,
            'stats': self.stats,
            'weight_history': self.weight_history,
        }
        try:
            with open(self.save_file, 'w') as f:
                json.dump(state, f, indent=2)
        except OSError as e:
            print(f"Warning: Could not save adaptive weights: {e}")
    
    def _load_state(self):
        """Load persisted weights and learning state from JSON file."""
        if not self.save_file or not os.path.exists(self.save_file):
            return
        try:
            with open(self.save_file, 'r') as f:
                state = json.load(f)
            self._set_weights(state['weights'])
            self.best_weights = state['best_weights']
            self.best_win_streak = state.get('best_win_streak', 0)
            self.stats = state.get('stats', self.stats)
            self.weight_history = state.get('weight_history', [])
            self.current_streak = 0  # Reset streak between runs
            print(f"  Loaded {self.name} weights from {self.save_file}"
                  f" ({self.stats['games_played']} prior games)")
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load adaptive weights: {e}")
    
    def _current_lr(self) -> float:
        games = self.stats['games_played']
        return self.initial_lr / (1.0 + self.lr_decay * games)
    
    def update_stats(self, won: bool, reward: float, moves: int):
        super().update_stats(won, reward, moves)
        self._update_weights(won)
    
    def _update_weights(self, won: bool):
        lr = self._current_lr()
        current = self._get_weights()
        
        if won:
            self.current_streak += 1
            if self.current_streak >= self.best_win_streak:
                self.best_win_streak = self.current_streak
                self.best_weights = current.copy()
            
            # Reinforce: blend current weights toward best-known weights
            new_weights = {}
            for key in current:
                blend = (current[key] * (1 - lr * self.win_momentum)
                         + self.best_weights[key] * lr * self.win_momentum)
                new_weights[key] = blend
        else:
            self.current_streak = 0
            
            # Explore: random perturbation with drift toward best-known
            new_weights = {}
            for key in current:
                delta = (random.random() - 0.5) * 2 * lr * self.loss_perturbation
                drift = (self.best_weights[key] - current[key]) * lr * 0.5
                new_weights[key] = current[key] + delta + drift
        
        self._set_weights(new_weights)
        
        self.weight_history.append({
            'game': self.stats['games_played'],
            'won': won,
            'lr': lr,
            **self._get_weights(),
        })
        
        self._save_state()
    
    def get_weight_summary(self) -> str:
        """Return a human-readable summary of weight evolution."""
        if not self.weight_history:
            return f"{self.name}: no games played"
        
        first = self.weight_history[0]
        last = self.weight_history[-1]
        wins = sum(1 for h in self.weight_history if h['won'])
        
        lines = [
            f"{self.name} Weight Evolution ({len(self.weight_history)} games, {wins} wins):",
            f"  LR: {first['lr']:.3f} -> {last['lr']:.3f}",
            f"  Aggressiveness:      {first['aggressiveness']:.3f} -> {last['aggressiveness']:.3f}",
            f"  Meld size pref:      {first['meld_size_preference']:.3f} -> {last['meld_size_preference']:.3f}",
            f"  Draw preference:     {first['draw_preference']:.3f} -> {last['draw_preference']:.3f}",
            f"  Best streak: {self.best_win_streak}",
            f"  Best weights: agg={self.best_weights['aggressiveness']:.3f}, "
            f"meld={self.best_weights['meld_size_preference']:.3f}, "
            f"draw={self.best_weights['draw_preference']:.3f}",
        ]
        return '\n'.join(lines)


def create_adaptive(name: str = "Adaptive") -> AdaptiveWormAgent:
    """Adaptive agent that learns weights over the course of a tournament."""
    return AdaptiveWormAgent(
        name=name,
        aggressiveness=0.5,
        meld_size_preference=0.5,
        draw_preference=0.3,
    )


# Registry for easy access
AGENT_PRESETS = {
    'smartworm': SmartWormAgent,
    'hoarder': create_hoarder,
    'aggressive': create_aggressive,
    'strategic': create_strategic,
    'greedy': create_greedy,
    'cautious': create_cautious,
    'adaptive': create_adaptive,
}


if __name__ == "__main__":
    # Test all agent types
    print("Testing Agent Presets...")
    print("=" * 50)
    
    agents = [
        SmartWormAgent("SmartWorm"),
        create_hoarder(),
        create_aggressive(),
        create_strategic(),
        create_greedy(),
        create_cautious(),
        create_adaptive(),
    ]
    
    for agent in agents:
        weights = ""
        if isinstance(agent, WeightedWormAgent):
            weights = (f" agg={agent.aggressiveness},"
                      f" meld={agent.meld_size_preference},"
                      f" draw={agent.draw_preference}")
        adaptive_tag = " [ADAPTIVE]" if isinstance(agent, AdaptiveWormAgent) else ""
        print(f"  {agent.name:<15} worm={agent.worm_available}{weights}{adaptive_tag}")
    
    print(f"\nAll {len(agents)} agents created successfully.")
