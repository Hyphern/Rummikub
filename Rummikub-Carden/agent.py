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
        2. Try to add tiles using wormed to find solvable combinations
        3. Play valid melds from hand
        """
        import random
        import wormed
        from meld import find_all_valid_melds, Meld
        
        print(f"  [SmartWorm] hand={len(hand)} tiles, table={len(table)} melds, has_initial={has_initial}")
        
        # Step 1: Initial meld - must play 15+ point meld first
        if not has_initial:
            valid_melds = find_all_valid_melds(hand)
            print(f"  [SmartWorm] Initial meld search: found {len(valid_melds)} valid melds")
            for meld_tiles in valid_melds:
                if Meld.calculate_value(meld_tiles) >= 15:
                    action = self._meld_to_action(meld_tiles, hand, valid_indices)
                    if action is not None:
                        print(f"  [SmartWorm] Playing initial meld: {meld_tiles}")
                        return action
            # Can't make initial meld - must draw
            if 0 in valid_indices:
                print(f"  [SmartWorm] Drawing (no initial meld)")
                return 0
            return None
        
        # Step 2: Build table board
        table_board = self.table_to_board_matrix(table)
        
        # Step 3: Try direct meld plays from hand first
        valid_melds = find_all_valid_melds(hand)
        print(f"  [SmartWorm] Found {len(valid_melds)} valid melds in hand")
        if valid_melds:
            for meld_tiles in valid_melds:
                action = self._meld_to_action(meld_tiles, hand, valid_indices)
                if action is not None:
                    print(f"  [SmartWorm] Playing meld: {meld_tiles[:3]}...")
                    return action
        
        # Step 4: Try to add tiles using wormed (like testing.py)
        print(f"  [SmartWorm] Trying wormed solve for {len(hand)} tiles...")
        for tile_idx, tile in enumerate(hand):
            if tile.is_joker or not tile.color or not tile.number:
                continue
            
            # Create test board with this tile added
            test_board = [row[:] for row in table_board]
            color_to_row = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}
            row = color_to_row.get(tile.color)
            col = tile.number - 1
            
            if row is None:
                continue
            
            test_board[row][col] += 1
            
            # Check if solvable using wormed
            try:
                print(f"  [SmartWorm] Testing tile {tile.color} {tile.number}...")
                solution = wormed.solve(test_board)
                if solution is not None:
                    print(f"  [SmartWorm] SOLVABLE! Playing {tile.color} {tile.number}")
                    # This tile helps! Find action to play it
                    for action in valid_indices:
                        if 2 <= action < 2 + 30:
                            if action - 2 == tile_idx:
                                return action
            except Exception as e:
                print(f"  [SmartWorm] Error solving: {e}")
                pass  # If solve fails, try next tile
        
        # Step 5: Try to add to existing table melds (backup)
        add_action = self._find_add_to_meld_action(hand, table, valid_indices)
        if add_action is not None:
            print(f"  [SmartWorm] Adding to existing meld")
            return add_action
        
        # Step 6: Fallback - try any valid meld action
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        if meld_actions:
            print(f"  [SmartWorm] Random meld action")
            return random.choice(meld_actions)
        
        # Step 7: Last resort - draw
        if 0 in valid_indices:
            print(f"  [SmartWorm] Drawing (fallback)")
            return 0
        
        return None
    
    def _find_add_to_meld_action(self, hand: List[Any], table: List[Any], 
                                 valid_indices: np.ndarray) -> Optional[int]:
        """Try to add groups directly from hand to table melds."""
        from meld import Meld
        
        add_base_idx = 2 + 30
        
        for tile_idx, tile in enumerate(hand):
            for meld_idx, meld in enumerate(table):
                if Meld.can_add_tile(meld.tiles, tile):
                    action_idx = add_base_idx + tile_idx * 20 + meld_idx
                    if action_idx in valid_indices:
                        return action_idx
        
        return None
    
    def _meld_to_action(self, meld_tiles: List[Any], hand: List[Any],
                        valid_indices: np.ndarray) -> Optional[int]:
        """Convert a meld (list of tiles) to an action index."""
        # Find the indices of these tiles in hand
        tile_indices = []
        for tile in meld_tiles:
            for idx, hand_tile in enumerate(hand):
                if tile == hand_tile and idx not in tile_indices:
                    tile_indices.append(idx)
                    break
        
        return self._tile_indices_to_action(tile_indices, hand, valid_indices)
    
    def _tile_indices_to_action(self, tile_indices: List[int], hand: List[Any],
                                valid_indices: np.ndarray) -> Optional[int]:
        """Convert tile indices to action.
        
        The action encoding is simplified: actions 2 to 2+max_hand_size represent
        PLAY_NEW_MELD starting at tile index (action-2). The environment then
        determines what meld can be formed starting from that tile.
        
        So we just need to return any valid PLAY_NEW_MELD action - the environment
        will figure out the actual meld to play.
        """
        if len(tile_indices) < 3:
            return None
        
        # Find any valid PLAY_NEW_MELD action
        # The encoding is: action 2 = start at tile 0, action 3 = start at tile 1, etc.
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        
        if meld_actions:
            # Return a random one - the environment will figure out the meld
            import random
            return random.choice(meld_actions)
        
        # If no match, return first valid meld action
        meld_actions = [a for a in valid_indices if 2 <= a < 2 + 30]
        return meld_actions[0] if meld_actions else None
    
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
        for meld_idx in range(20):
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
        'table': np.zeros((20, 13), dtype=np.int32),
        'table_mask': np.zeros((20, 13), dtype=np.float32),
        'pool_size': np.array([50], dtype=np.int32),
        'has_initial_meld': np.array([0], dtype=np.float32),
        'opponent_hands': np.array([14], dtype=np.int32),
        'valid_actions_mask': np.zeros(652, dtype=np.float32),
        'current_player': np.array([0], dtype=np.int32),
        'turn_count': np.array([0], dtype=np.int32),
    }
    
    # Set some valid actions
    obs['valid_actions_mask'][0] = 1.0  # DRAW
    obs['valid_actions_mask'][2] = 1.0  # PLAY_NEW_MELD
    
    print(f"\nSmartWormAgent created: {agent.name}")
    print(f"Worm available: {agent.worm_available}")
