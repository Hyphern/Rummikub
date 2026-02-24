"""Agent definitions for Rummikub ML training.

Provides base agent class and several example agents.
"""

import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Any
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
    
    def _play_single_tile(self, tile_idx: int, hand: List[Any], 
                          table: List[Any], valid_indices: np.ndarray,
                          cached_melds: Optional[List] = None) -> Optional[int]:
        """Try to play a single tile (by index) from hand.
        
        Tries adding to existing table melds first, then forming a new meld.
        """
        from meld import Meld, find_all_valid_melds
        
        if tile_idx >= len(hand):
            return None
        
        tile = hand[tile_idx]
        
        # Try adding to an existing table meld
        add_base_idx = 2 + 30
        for meld_idx, meld in enumerate(table):
            if Meld.can_add_tile(meld.tiles, tile):
                action_idx = add_base_idx + tile_idx * 40 + meld_idx
                if action_idx in valid_indices:
                    return action_idx
        
        # Try forming a new meld that includes this tile
        melds = cached_melds if cached_melds is not None else find_all_valid_melds(hand)
        for meld_tiles in melds:
            if tile in meld_tiles:
                action = self._meld_to_action(meld_tiles, hand, valid_indices)
                if action is not None:
                    return action
        
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


if __name__ == "__main__":
    # Test SmartWormAgent
    print("Testing SmartWormAgent...")
    
    agent = SmartWormAgent("SmartWorm")
    
    # Mock observation
    obs = {
        'hand': np.zeros(30, dtype=np.int32),
        'hand_mask': np.zeros(30, dtype=np.float32),
        'table': np.zeros((40, 13), dtype=np.int32),
        'table_mask': np.zeros((40, 13), dtype=np.float32),
        'pool_size': np.array([50], dtype=np.int32),
        'has_initial_meld': np.array([0], dtype=np.float32),
        'opponent_hands': np.array([14], dtype=np.int32),
        'valid_actions_mask': np.zeros(1232, dtype=np.float32),
        'current_player': np.array([0], dtype=np.int32),
        'turn_count': np.array([0], dtype=np.int32),
    }
    
    # Set some valid actions
    obs['valid_actions_mask'][0] = 1.0  # DRAW
    obs['valid_actions_mask'][2] = 1.0  # PLAY_NEW_MELD
    
    print(f"\nSmartWormAgent created: {agent.name}")
    print(f"Worm available: {agent.worm_available}")
