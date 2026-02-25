"""Worm integration module for Rummikub agents.

Provides utilities to combine hand tiles with board state and use
worm.py logic to find optimal moves.
"""

import sys
sys.path.insert(0, 'worm')
from worm import (
    find_explicit_moves, 
    get_all_runs, 
    get_all_groups,
    apply_move,
    is_solved,
    solve,
    board_to_hashable
)

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from tile import Tile
from meld import Meld


def hand_to_board_matrix(hand: List[Tile]) -> List[List[int]]:
    """Convert hand tiles to 4x13 board matrix format.
    
    Returns matrix where each cell is tile count (0, 1, or 2 for duplicates).
    """
    # Initialize empty board
    board = [[0] * 13 for _ in range(4)]
    color_to_row = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}
    
    for tile in hand:
        if not tile.is_joker and tile.color and tile.number:
            row = color_to_row.get(tile.color, 0)
            col = tile.number - 1
            board[row][col] += 1
    
    return board


def table_to_board_matrix(table_melds: List[Meld]) -> List[List[int]]:
    """Convert table melds to 4x13 board matrix format."""
    board = [[0] * 13 for _ in range(4)]
    color_to_row = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}
    
    for meld in table_melds:
        for tile in meld.tiles:
            if not tile.is_joker and tile.color and tile.number:
                row = color_to_row.get(tile.color, 0)
                col = tile.number - 1
                board[row][col] += 1
    
    return board


def combine_boards(hand_board: List[List[int]], 
                   table_board: List[List[int]]) -> List[List[int]]:
    """Combine hand and table boards into single matrix.
    
    This allows worm logic to find moves using both hand and table tiles.
    """
    combined = []
    for h_row, t_row in zip(hand_board, table_board):
        combined.append([h + t for h, t in zip(h_row, t_row)])
    return combined


def find_hand_only_moves(hand: List[Tile]) -> List[Tuple]:
    """Find all valid moves that can be made using only hand tiles.
    
    Returns list of moves in worm format:
    - ('run', row, start_col, end_col)
    - ('group', col, [rows])
    """
    if len(hand) < 3:
        return []
    
    board = hand_to_board_matrix(hand)
    board_hashable = board_to_hashable(board)
    
    # Get all possible runs and groups
    runs = get_all_runs(board_hashable)
    groups = get_all_groups(board_hashable)
    
    return runs + groups


def find_combined_moves(hand: List[Tile], 
                       table_melds: List[Meld]) -> Dict[str, List[Tuple]]:
    """Find moves considering both hand and table tiles.
    
    Returns dict with:
    - 'hand_only': Moves using only hand tiles
    - 'add_to_meld': Moves adding hand tiles to existing melds
    - 'combined': New moves possible with hand+table combined
    """
    result = {
        'hand_only': [],
        'add_to_meld': [],
        'combined': []
    }
    
    if len(hand) < 3 and not table_melds:
        return result
    
    # Convert to board format
    hand_board = hand_to_board_matrix(hand)
    table_board = table_to_board_matrix(table_melds)
    combined_board = combine_boards(hand_board, table_board)
    combined_hashable = board_to_hashable(combined_board)
    
    # Find hand-only moves
    result['hand_only'] = find_hand_only_moves(hand)
    
    # Find moves using worm logic on combined board
    # These are moves that become possible when considering hand+table together
    all_combined_runs = get_all_runs(combined_hashable)
    all_combined_groups = get_all_groups(combined_hashable)
    
    # Filter to moves that use at least one hand tile
    for move in all_combined_runs + all_combined_groups:
        uses_hand = False
        if move[0] == 'run':
            _, row, start, end = move
            for col in range(start, end + 1):
                if hand_board[row][col] > 0:
                    uses_hand = True
                    break
        else:  # group
            _, col, rows = move
            for row in rows:
                if hand_board[row][col] > 0:
                    uses_hand = True
                    break
        
        if uses_hand:
            result['combined'].append(move)
    
    # Find add-to-meld moves
    for meld_idx, meld in enumerate(table_melds):
        for tile_idx, tile in enumerate(hand):
            if Meld.can_add_tile(meld.tiles, tile):
                result['add_to_meld'].append((meld_idx, tile_idx, tile))
    
    return result


def score_move_for_hoarding(move: Tuple, hand: List[Tile], 
                            table_melds: List[Meld]) -> float:
    """Score a move for hoarding strategy (higher = better for hoarding).
    
    Hoarding agents want to:
    - Save tiles for one big round
    - Not reveal their hand
    - Keep flexible tiles (jokers, middle numbers)
    - Only play when they can make significant progress
    """
    score = 0.0
    
    if move[0] == 'run':
        _, row, start, end = move
        length = end - start + 1
        
        # Prefer longer runs (more tiles played = closer to going out)
        score += length * 2
        
        # But penalize using too many tiles too early
        if length < 5:
            score -= 5  # Small runs waste tiles
        
        # High numbers are harder to extend, so playing them is okay
        if end >= 10:
            score += 2
            
    elif move[0] == 'group':
        _, col, rows = move
        
        # Groups are efficient (3-4 tiles at once)
        score += len(rows) * 3
        
        # Middle numbers are more flexible
        if 4 <= col <= 9:
            score -= 3  # Save middle numbers
    
    return score


def score_move_for_aggressive(move: Tuple, hand: List[Tile],
                              table_melds: List[Meld]) -> float:
    """Score a move for aggressive strategy (higher = better for aggressive).
    
    Aggressive agents want to:
    - Play tiles as soon as possible
    - Reduce hand size quickly
    - Open up the board for more plays
    - Not worry about saving tiles
    """
    score = 0.0
    
    if move[0] == 'run':
        _, row, start, end = move
        length = end - start + 1
        
        # Prefer any run - just play tiles!
        score += length * 5
        
        # Shorter runs are fine for aggressive
        score += 10  # Bonus for playing anything
        
    elif move[0] == 'group':
        _, col, rows = move
        
        # Groups are great - play multiple tiles at once
        score += len(rows) * 5
        score += 10  # Bonus for groups
    
    return score


def score_move_for_balanced(move: Tuple, hand: List[Tile],
                           table_melds: List[Meld]) -> float:
    """Score a move for balanced strategy.
    
    Balanced agents consider:
    - Value of tiles played
    - Board position
    - Flexibility remaining in hand
    """
    score = 0.0
    
    if move[0] == 'run':
        _, row, start, end = move
        length = end - start + 1
        
        # Prefer medium-length runs
        if length >= 3 and length <= 6:
            score += length * 3
        elif length > 6:
            score += 15  # Long runs are good
        else:
            score += length * 2
            
    elif move[0] == 'group':
        _, col, rows = move
        
        # Groups are efficient
        score += len(rows) * 4
    
    return score


def get_best_move_for_strategy(hand: List[Tile],
                               table_melds: List[Meld],
                               strategy: str = 'balanced') -> Optional[Tuple]:
    """Get the best move for a given strategy.
    
    Args:
        hand: List of tiles in hand
        table_melds: List of melds on table
        strategy: 'hoarding', 'aggressive', or 'balanced'
    
    Returns:
        Best move tuple or None
    """
    moves = find_hand_only_moves(hand)
    
    if not moves:
        return None
    
    # Score each move
    if strategy == 'hoarding':
        scored_moves = [(move, score_move_for_hoarding(move, hand, table_melds)) 
                       for move in moves]
    elif strategy == 'aggressive':
        scored_moves = [(move, score_move_for_aggressive(move, hand, table_melds))
                       for move in moves]
    else:  # balanced
        scored_moves = [(move, score_move_for_balanced(move, hand, table_melds))
                       for move in moves]
    
    # Return best move
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves[0][0] if scored_moves else None


def can_go_out_in_one_move(hand: List[Tile], 
                           table_melds: List[Meld]) -> bool:
    """Check if player can play all tiles in one move.
    
    This is the hoarding agent's dream scenario.
    """
    moves = find_hand_only_moves(hand)
    
    # Check if any single move uses all tiles
    for move in moves:
        if move[0] == 'run':
            _, row, start, end = move
            length = end - start + 1
            if length == len(hand):
                return True
        elif move[0] == 'group':
            _, col, rows = move
            if len(rows) == len(hand):
                return True
    
    return False


def count_playable_tiles(hand: List[Tile], 
                        table_melds: List[Meld]) -> int:
    """Count how many tiles can be played immediately.
    
    Useful for aggressive agents to see their options.
    """
    moves = find_combined_moves(hand, table_melds)
    
    playable = 0
    for move in moves['hand_only']:
        if move[0] == 'run':
            _, row, start, end = move
            playable += (end - start + 1)
        elif move[0] == 'group':
            _, col, rows = move
            playable += len(rows)
    
    return playable


if __name__ == "__main__":
    # Test the integration
    from tile import Tile
    
    # Create test hand
    test_hand = [
        Tile('red', 4), Tile('red', 5), Tile('red', 6),  # Run
        Tile('blue', 7), Tile('black', 7), Tile('orange', 7),  # Group
        Tile('red', 10), Tile('red', 11),  # Partial run
    ]
    
    print("Test Hand:")
    for tile in test_hand:
        print(f"  {tile}")
    
    print("\nHand Board Matrix:")
    board = hand_to_board_matrix(test_hand)
    for i, row in enumerate(board):
        print(f"  Row {i}: {row}")
    
    print("\nHand-Only Moves:")
    moves = find_hand_only_moves(test_hand)
    for move in moves:
        print(f"  {move}")
    
    print("\nBest Move for Each Strategy:")
    for strategy in ['hoarding', 'aggressive', 'balanced']:
        best = get_best_move_for_strategy(test_hand, [], strategy)
        print(f"  {strategy}: {best}")
