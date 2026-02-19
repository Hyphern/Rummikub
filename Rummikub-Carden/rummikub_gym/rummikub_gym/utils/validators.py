"""Utility functions for Rummikub."""

from typing import List
from ..envs.tile import Tile
from ..envs.meld import Meld


def validate_game_state(table_melds: List[Meld], player_hands: List[List[Tile]]) -> bool:
    """Validate that a game state is legal.
    
    Args:
        table_melds: List of melds on the table
        player_hands: List of player hands
        
    Returns:
        True if game state is valid
    """
    # All table melds must be valid
    for meld in table_melds:
        if not Meld.is_valid(meld.tiles):
            return False
    
    # Check for duplicate tiles (no tile should appear twice)
    all_tiles = []
    for meld in table_melds:
        all_tiles.extend(meld.tiles)
    for hand in player_hands:
        all_tiles.extend(hand)
    
    # Check for duplicates by comparing length of list vs set
    if len(all_tiles) != len(set(all_tiles)):
        return False
    
    return True
