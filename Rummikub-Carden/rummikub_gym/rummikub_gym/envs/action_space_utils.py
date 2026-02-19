"""Action space utilities for Rummikub."""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from .tile import Tile
from .game_state import GameState
from .meld import Meld


class ActionSpace:
    """Manages action encoding and decoding for Rummikub environment."""
    
    # Action type constants
    DRAW = 0
    PLAY_NEW_MELD = 1
    ADD_TO_MELD = 2
    DECLARE_OUT = 3
    
    ACTION_TYPES = ['DRAW', 'PLAY_NEW_MELD', 'ADD_TO_MELD', 'DECLARE_OUT']
    
    def __init__(self, max_hand_size: int = 14, max_melds: int = 20):
        """Initialize action space.
        
        Args:
            max_hand_size: Maximum tiles in hand (for action encoding)
            max_melds: Maximum melds on table
        """
        self.max_hand_size = max_hand_size
        self.max_melds = max_melds
    
    def encode_action(self, action: Dict) -> int:
        """Encode structured action to discrete integer.
        
        Action encoding scheme:
        - 0: DRAW
        - 1: DECLARE_OUT
        - 2-15: PLAY_NEW_MELD with single tile (indices 0-13)
        - 16+: Complex actions (combinations)
        
        For simplicity, we use a hierarchical encoding:
        - Type 0 (DRAW): action_id = 0
        - Type 1 (DECLARE_OUT): action_id = 1
        - Type 2 (PLAY_NEW_MELD): action_id = 2 + combination_index
        - Type 3 (ADD_TO_MELD): action_id = 2 + num_combinations + addition_index
        
        Args:
            action: Action dictionary with 'type' and parameters
            
        Returns:
            Encoded action as integer
        """
        action_type = action['type']
        
        if action_type == 'DRAW':
            return 0
        
        if action_type == 'DECLARE_OUT':
            return 1
        
        if action_type == 'PLAY_NEW_MELD':
            # Encode tile indices as a combination
            indices = tuple(sorted(action['tile_indices']))
            combo_id = self._encode_combination(indices)
            return 2 + combo_id
        
        if action_type == 'ADD_TO_MELD':
            # Encode as: base + meld_offset * max_hand_size + tile_idx
            tile_idx = action['tile_idx']
            meld_idx = action['meld_idx']
            
            # Calculate base offset for ADD_TO_MELD actions
            # We reserve space for all possible PLAY_NEW_MELD combinations
            base = 2 + self._get_max_combinations()
            
            action_id = base + meld_idx * self.max_hand_size + tile_idx
            return action_id
        
        raise ValueError(f"Unknown action type: {action_type}")
    
    def decode_action(self, action_id: int) -> Dict:
        """Decode integer to structured action.
        
        Args:
            action_id: Encoded action integer
            
        Returns:
            Decoded action dictionary
        """
        if action_id == 0:
            return {'type': 'DRAW'}
        
        if action_id == 1:
            return {'type': 'DECLARE_OUT'}
        
        max_combos = self._get_max_combinations()
        
        if action_id < 2 + max_combos:
            # PLAY_NEW_MELD
            combo_id = action_id - 2
            indices = self._decode_combination(combo_id)
            return {'type': 'PLAY_NEW_MELD', 'tile_indices': list(indices)}
        
        # ADD_TO_MELD
        base = 2 + max_combos
        offset = action_id - base
        
        meld_idx = offset // self.max_hand_size
        tile_idx = offset % self.max_hand_size
        
        return {
            'type': 'ADD_TO_MELD',
            'tile_idx': tile_idx,
            'meld_idx': meld_idx
        }
    
    def _encode_combination(self, indices: Tuple[int, ...]) -> int:
        """Encode a combination of tile indices to integer.
        
        Uses a simple scheme for combinations of 3-14 tiles from 0-13.
        """
        # For simplicity, we'll use a hash-based encoding
        # In practice, you'd want a more systematic bijective encoding
        result = 0
        for i, idx in enumerate(indices):
            result += idx * (self.max_hand_size ** i)
        
        # Add offset based on combination size
        size_offset = sum(self.max_hand_size ** i for i in range(3))  # Minimum 3 tiles
        result += size_offset * (len(indices) - 3)
        
        return result
    
    def _decode_combination(self, combo_id: int) -> Tuple[int, ...]:
        """Decode integer to combination of tile indices."""
        # Reverse the encoding
        # This is a simplified version - full implementation would be more complex
        indices = []
        remaining = combo_id
        
        while remaining > 0:
            idx = remaining % self.max_hand_size
            indices.append(idx)
            remaining //= self.max_hand_size
        
        return tuple(indices)
    
    def _get_max_combinations(self) -> int:
        """Calculate maximum number of possible combinations."""
        # Sum of C(14, k) for k=3 to 14
        from math import comb
        return sum(comb(self.max_hand_size, k) for k in range(3, self.max_hand_size + 1))
    
    def get_action_space_size(self) -> int:
        """Get total size of action space."""
        # DRAW + DECLARE_OUT + PLAY_NEW_MELD combinations + ADD_TO_MELD possibilities
        return 2 + self._get_max_combinations() + self.max_melds * self.max_hand_size
    
    def get_valid_actions(self, game_state: GameState, player_id: int) -> List[int]:
        """Get list of valid action IDs for a player.
        
        Args:
            game_state: Current game state
            player_id: Player to get actions for
            
        Returns:
            List of valid action integers
        """
        valid_actions = []
        hand = game_state.player_hands[player_id]
        
        # DRAW is always valid (if tiles remain)
        if not game_state.tile_pool.is_empty():
            valid_actions.append(0)  # DRAW
        
        # Check for valid new melds
        from itertools import combinations
        for size in range(3, len(hand) + 1):
            for indices in combinations(range(len(hand)), size):
                tiles = [hand[i] for i in indices]
                if Meld.is_valid(tiles):
                    # Check initial meld requirement
                    if (game_state.has_initial_meld[player_id] or 
                        Meld.calculate_value(tiles) >= 30):
                        action = {
                            'type': 'PLAY_NEW_MELD',
                            'tile_indices': list(indices)
                        }
                        valid_actions.append(self.encode_action(action))
        
        # Check for valid additions to table melds
        if game_state.has_initial_meld[player_id]:
            for tile_idx, tile in enumerate(hand):
                for meld_idx, meld in enumerate(game_state.table_melds):
                    if Meld.can_add_tile(meld.tiles, tile):
                        action = {
                            'type': 'ADD_TO_MELD',
                            'tile_idx': tile_idx,
                            'meld_idx': meld_idx
                        }
                        valid_actions.append(self.encode_action(action))
        
        # DECLARE_OUT if hand is empty
        if game_state.can_go_out(player_id):
            valid_actions.append(1)  # DECLARE_OUT
        
        return valid_actions
    
    def create_action_mask(self, game_state: GameState, player_id: int) -> np.ndarray:
        """Create binary mask of valid actions.
        
        Args:
            game_state: Current game state
            player_id: Player to create mask for
            
        Returns:
            Binary numpy array where 1 = valid action
        """
        mask = np.zeros(self.get_action_space_size(), dtype=np.int32)
        valid_actions = self.get_valid_actions(game_state, player_id)
        mask[valid_actions] = 1
        return mask


def apply_action(game_state: GameState, player_id: int, action: Dict) -> Tuple[bool, str]:
    """Apply an action to the game state.
    
    Args:
        game_state: Current game state (will be modified)
        player_id: Player taking the action
        action: Action dictionary
        
    Returns:
        Tuple of (success, message)
    """
    action_type = action['type']
    
    if action_type == 'DRAW':
        tile = game_state.draw_tile(player_id)
        if tile:
            return True, f"Drew {tile}"
        else:
            return False, "No tiles remaining in pool"
    
    if action_type == 'PLAY_NEW_MELD':
        indices = action['tile_indices']
        success = game_state.play_meld(player_id, indices)
        if success:
            return True, f"Played meld with tiles at indices {indices}"
        else:
            return False, "Invalid meld or insufficient points for initial meld"
    
    if action_type == 'ADD_TO_MELD':
        tile_idx = action['tile_idx']
        meld_idx = action['meld_idx']
        success = game_state.add_to_meld(player_id, tile_idx, meld_idx)
        if success:
            return True, f"Added tile {tile_idx} to meld {meld_idx}"
        else:
            return False, "Cannot add tile to meld"
    
    if action_type == 'DECLARE_OUT':
        success = game_state.declare_out(player_id)
        if success:
            return True, f"Player {player_id} declared Rummikub!"
        else:
            return False, "Cannot declare out - hand not empty"
    
    return False, f"Unknown action type: {action_type}"
