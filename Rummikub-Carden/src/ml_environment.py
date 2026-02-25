"""Machine Learning Environment for Rummikub.

Provides a gym-like interface for training agents to play Rummikub.
Integrates with the local game logic (tile.py, meld.py, game_state.py).
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any
from game_state import GameState
from tile import Tile, TileSet, sort_tiles_for_hand
from meld import Meld, find_all_valid_melds, find_best_initial_meld


class RummikubMLEnv:
    """ML Environment for Rummikub training.
    
    Provides:
    - State observation (encoded for neural networks)
    - Action space enumeration
    - Reward signals
    - Episode management
    """
    
    def __init__(self, num_players: int = 2, max_steps: int = 1000):
        """Initialize environment.
        
        Args:
            num_players: Number of players (2-4)
            max_steps: Maximum steps per episode
        """
        self.num_players = num_players
        self.max_steps = max_steps
        self.game_state: Optional[GameState] = None
        self.current_step = 0
        self.current_player = 0
        
        # Action space sizes
        self.max_hand_size = 30  # Can grow if drawing a lot
        self.max_melds = 40
        self.max_tiles_per_meld = 13
        
    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed
            
        Returns:
            Initial observation
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.game_state = GameState(self.num_players)
        self.game_state.reset()
        self.current_step = 0
        self.current_player = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation as numpy arrays.
        
        Returns dict with:
        - hand: Encoded hand tiles (fixed size, padded)
        - hand_mask: Which hand positions are valid
        - table: Encoded table melds (2D: melds x tiles)
        - table_mask: Which table positions are valid
        - pool_size: Tiles remaining in pool
        - has_initial_meld: Whether current player has made initial meld
        - opponent_hands: Sizes of opponent hands
        - valid_actions_mask: Binary mask of valid actions
        """
        if self.game_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Get current player's hand
        hand = self.game_state.get_current_player_hand()
        hand_encoded = np.zeros(self.max_hand_size, dtype=np.int32)
        hand_mask = np.zeros(self.max_hand_size, dtype=np.float32)
        
        for i, tile in enumerate(hand[:self.max_hand_size]):
            hand_encoded[i] = tile.encode()
            hand_mask[i] = 1.0
        
        # Get table melds
        table = np.zeros((self.max_melds, self.max_tiles_per_meld), dtype=np.int32)
        table_mask = np.zeros((self.max_melds, self.max_tiles_per_meld), dtype=np.float32)
        
        for meld_idx, meld in enumerate(self.game_state.table_melds[:self.max_melds]):
            for tile_idx, tile in enumerate(meld.tiles[:self.max_tiles_per_meld]):
                table[meld_idx, tile_idx] = tile.encode()
                table_mask[meld_idx, tile_idx] = 1.0
        
        # Get valid actions mask
        valid_actions = self.get_valid_actions()
        
        return {
            'hand': hand_encoded,
            'hand_mask': hand_mask,
            'table': table,
            'table_mask': table_mask,
            'pool_size': np.array([self.game_state.tile_pool.remaining()], dtype=np.int32),
            'has_initial_meld': np.array([1 if self.game_state.has_initial_meld[self.game_state.current_player] else 0], dtype=np.float32),
            'opponent_hands': np.array(self.game_state.get_opponent_hand_sizes(), dtype=np.int32),
            'valid_actions_mask': valid_actions,
            'current_player': np.array([self.game_state.current_player], dtype=np.int32),
            'turn_count': np.array([self.game_state.turn_count], dtype=np.int32),
        }
    
    def get_valid_actions(self) -> np.ndarray:
        """Get binary mask of valid actions.
        
        Action encoding:
        - 0: DRAW
        - 1: DECLARE_OUT
        - 2 to 2+max_hand_size: PLAY_NEW_MELD starting with tile at index (action-2)
        - Remaining: ADD_TO_MELD (encoded as hand_idx * max_melds + meld_idx)
        
        Returns:
            Binary array indicating which actions are valid
        """
        if self.game_state is None:
            raise RuntimeError("Environment not initialized")
        
        # Calculate action space size
        # Base actions: DRAW, DECLARE_OUT
        # PLAY_NEW_MELD: For each possible starting tile (simplified - we'd need combinations)
        # ADD_TO_MELD: hand_idx * max_melds + meld_idx
        action_size = 2 + self.max_hand_size + (self.max_hand_size * self.max_melds)
        
        valid_mask = np.zeros(action_size, dtype=np.float32)
        
        # DRAW is always valid if pool has tiles
        if not self.game_state.tile_pool.is_empty():
            valid_mask[0] = 1.0
        
        # DECLARE_OUT if hand is empty
        if self.game_state.can_go_out(self.game_state.current_player):
            valid_mask[1] = 1.0
        
        hand = self.game_state.get_current_player_hand()
        player_id = self.game_state.current_player
        has_initial = self.game_state.has_initial_meld[player_id]
        
        # Check for valid new melds (simplified - check individual tiles as start)
        # In reality we'd need combinations, but for now let's use a simplified approach
        base_idx = 2
        
        # Find all valid melds from hand
        valid_melds = find_all_valid_melds(hand)
        
        for meld_tiles in valid_melds:
            # Get indices of tiles in this meld
            tile_indices = []
            for tile in meld_tiles:
                for idx, hand_tile in enumerate(hand):
                    if tile == hand_tile and idx not in tile_indices:
                        tile_indices.append(idx)
                        break
            
            if len(tile_indices) >= 3:
                # Check initial meld requirement
                if has_initial or Meld.calculate_value(meld_tiles) >= 15:
                    # Mark action as valid (using first tile index as identifier)
                    # This is a simplification - real implementation would need full encoding
                    if tile_indices[0] < self.max_hand_size:
                        valid_mask[base_idx + tile_indices[0]] = 1.0
        
        # Check for valid additions to table melds
        add_base_idx = 2 + self.max_hand_size
        
        if has_initial:
            for tile_idx, tile in enumerate(hand):
                for meld_idx, meld in enumerate(self.game_state.table_melds):
                    if Meld.can_add_tile(meld.tiles, tile):
                        action_idx = add_base_idx + tile_idx * self.max_melds + meld_idx
                        if action_idx < action_size:
                            valid_mask[action_idx] = 1.0
        
        return valid_mask
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            (observation, reward, done, info)
        """
        if self.game_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.current_step += 1
        player_id = self.game_state.current_player
        
        # Execute action
        reward = 0.0
        done = False
        info = {
            'action_taken': None,
            'action_success': False,
            'winner': None,
        }
        
        if action == 0:
            # DRAW
            tile = self.game_state.draw_tile(player_id)
            if tile:
                info['action_taken'] = 'DRAW'
                info['action_success'] = True
                reward = -0.1  # Small penalty for drawing
                self.game_state.next_player()
            else:
                info['action_taken'] = 'DRAW_FAILED'
                reward = -1.0
                # Pool empty — pass turn so game doesn't stall
                self.game_state.next_player()
        elif action == 1:
            # DECLARE_OUT
            if self.game_state.declare_out(player_id):
                info['action_taken'] = 'DECLARE_OUT'
                info['action_success'] = True
                info['winner'] = player_id
                reward = 100.0  # Big reward for winning
                done = True
            else:
                info['action_taken'] = 'DECLARE_OUT_FAILED'
                reward = -5.0
                
        elif action < 2 + self.max_hand_size:
            # PLAY_NEW_MELD (simplified - action indicates starting tile)
            # Find a valid meld containing this tile
            tile_idx = action - 2
            hand = self.game_state.get_current_player_hand()
            
            if tile_idx < len(hand):
                # Try to find a valid meld starting with this tile
                # This is simplified - in practice you'd need full combination encoding
                valid_melds = find_all_valid_melds(hand)
                
                for meld_tiles in valid_melds:
                    # Check if this meld contains the selected tile
                    tile_indices = []
                    for tile in meld_tiles:
                        for idx, hand_tile in enumerate(hand):
                            if tile == hand_tile and idx not in tile_indices:
                                tile_indices.append(idx)
                                break
                    
                    if tile_idx in tile_indices and len(tile_indices) >= 3:
                        # Check initial meld requirement (15 points)
                        if (self.game_state.has_initial_meld[player_id] or 
                            Meld.calculate_value(meld_tiles) >= 15):
                            
                            if self.game_state.play_meld(player_id, tile_indices):
                                info['action_taken'] = 'PLAY_NEW_MELD'
                                info['action_success'] = True
                                info['meld_value'] = Meld.calculate_value(meld_tiles)
                                reward = 1.0 + (info['meld_value'] / 30.0) * 0.5
                                
                                if self.game_state.can_go_out(player_id):
                                    self.game_state.declare_out(player_id)
                                    info['winner'] = player_id
                                    reward += 100.0
                                    done = True
                                else:
                                    self.game_state.next_player()
                                break
                else:
                    info['action_taken'] = 'PLAY_NEW_MELD_FAILED'
                    reward = -1.0
            else:
                info['action_taken'] = 'INVALID_ACTION'
                reward = -2.0
                
        else:
            # ADD_TO_MELD
            add_base_idx = 2 + self.max_hand_size
            add_idx = action - add_base_idx
            
            tile_idx = add_idx // self.max_melds
            meld_idx = add_idx % self.max_melds
            
            if tile_idx < len(self.game_state.get_current_player_hand()):
                if meld_idx < len(self.game_state.table_melds):
                    if self.game_state.add_to_meld(player_id, tile_idx, meld_idx):
                        info['action_taken'] = 'ADD_TO_MELD'
                        info['action_success'] = True
                        reward = 0.5
                        
                        if self.game_state.can_go_out(player_id):
                            self.game_state.declare_out(player_id)
                            info['winner'] = player_id
                            reward += 100.0
                            done = True
                        else:
                            self.game_state.next_player()
                    else:
                        info['action_taken'] = 'ADD_TO_MELD_FAILED'
                        reward = -1.0
                else:
                    info['action_taken'] = 'INVALID_MELD'
                    reward = -1.0
            else:
                info['action_taken'] = 'INVALID_TILE'
                reward = -2.0
        
        # Check for game over conditions
        if not done:
            if self.game_state.game_over:
                done = True
                info['winner'] = self.game_state.winner
                if info['winner'] == player_id:
                    reward = 100.0
                else:
                    # Penalty based on hand value
                    hand_value = self.game_state.calculate_hand_value(player_id)
                    reward = -hand_value
            elif self.current_step >= self.max_steps:
                done = True
                # Calculate reward based on hand value
                hand_value = self.game_state.calculate_hand_value(player_id)
                info['winner'] = self.game_state.winner
                if info['winner'] == player_id:
                    reward = 100.0
                else:
                    reward = -hand_value
        
        return self._get_observation(), reward, done, info
    
    def get_state_for_worm(self) -> List[List[int]]:
        """Convert current game state to worm.py board format.
        
        Returns 4x13 matrix representing tiles on table.
        """
        if self.game_state is None:
            return [[0] * 13 for _ in range(4)]
        
        # Initialize empty board
        board = [[0] * 13 for _ in range(4)]
        color_to_row = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}
        
        # Fill in tiles from table melds
        for meld in self.game_state.table_melds:
            for tile in meld.tiles:
                if not tile.is_joker and tile.color and tile.number:
                    row = color_to_row.get(tile.color, 0)
                    col = tile.number - 1
                    board[row][col] += 1
        
        return board


if __name__ == "__main__":
    # Test the environment
    env = RummikubMLEnv(num_players=2)
    obs = env.reset(seed=42)
    
    print("Initial observation shapes:")
    for key, value in obs.items():
        print(f"  {key}: {value.shape} - {value}")
    
    print("\nValid actions count:", int(obs['valid_actions_mask'].sum()))
    
    # Take a few random steps
    for i in range(5):
        valid_actions = np.where(obs['valid_actions_mask'] == 1)[0]
        if len(valid_actions) == 0:
            print(f"Step {i}: No valid actions!")
            break
        
        action = np.random.choice(valid_actions)
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {i}:")
        print(f"  Action: {action} ({info['action_taken']})")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Valid actions: {int(obs['valid_actions_mask'].sum())}")
        
        if done:
            print(f"  Winner: {info.get('winner')}")
            break
