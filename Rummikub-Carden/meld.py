"""Meld validation logic for Rummikub."""

from typing import List, Optional, Set, Tuple
from tile import Tile


class Meld:
    """Represents a valid meld (run or group) on the table."""
    
    def __init__(self, tiles: List[Tile]):
        """Initialize a meld with tiles.
        
        Args:
            tiles: List of tiles in the meld
            
        Raises:
            ValueError: If tiles don't form a valid meld
        """
        self.tiles = list(tiles)  # Copy to avoid external modification
        if not self.is_valid(self.tiles):
            raise ValueError(f"Tiles do not form a valid meld: {tiles}")
    
    @classmethod
    def is_valid(cls, tiles: List[Tile]) -> bool:
        """Check if tiles form a valid meld (run or group).
        
        Args:
            tiles: List of tiles to check
            
        Returns:
            True if valid meld, False otherwise
        """
        if len(tiles) < 3:
            return False
        return cls.is_valid_run(tiles) or cls.is_valid_group(tiles)
    
    @classmethod
    def is_valid_run(cls, tiles: List[Tile]) -> bool:
        """Check if tiles form a valid run (consecutive same-color sequence).
        
        A valid run has:
        - 3+ tiles
        - All same color (except jokers)
        - Consecutive numbers (jokers can fill gaps)
        
        Args:
            tiles: List of tiles to check
            
        Returns:
            True if valid run, False otherwise
        """
        if len(tiles) < 3:
            return False
        
        # Separate jokers and regular tiles
        jokers = [t for t in tiles if t.is_joker]
        regular = [t for t in tiles if not t.is_joker]
        
        if not regular:
            # All jokers - not a valid run (can't determine color)
            return False
        
        # Check all regular tiles have same color
        color = regular[0].color
        if any(t.color != color for t in regular):
            return False
        
        # Sort regular tiles by number
        regular_sorted = sorted(regular, key=lambda t: t.number if t.number is not None else 0)
        
        # Check for duplicates
        numbers = [t.number for t in regular_sorted]
        if len(numbers) != len(set(numbers)):
            return False
        
        # Check if we can form consecutive sequence with jokers
        gaps = 0
        for i in range(len(regular_sorted) - 1):
            num1 = regular_sorted[i].number if regular_sorted[i].number is not None else 0
            num2 = regular_sorted[i + 1].number if regular_sorted[i + 1].number is not None else 0
            gap = num2 - num1 - 1
            if gap < 0:
                return False
            gaps += gap
        
        # Jokers must be able to fill all gaps
        return gaps <= len(jokers)
    
    @classmethod
    def is_valid_group(cls, tiles: List[Tile]) -> bool:
        """Check if tiles form a valid group (same number, different colors).
        
        A valid group has:
        - 3-4 tiles
        - All same number (except jokers)
        - All different colors
        - Jokers can substitute for missing colors
        
        Args:
            tiles: List of tiles to check
            
        Returns:
            True if valid group, False otherwise
        """
        if len(tiles) < 3 or len(tiles) > 4:
            return False
        
        # Separate jokers and regular tiles
        jokers = [t for t in tiles if t.is_joker]
        regular = [t for t in tiles if not t.is_joker]
        
        if not regular:
            # All jokers - not valid (can't determine number)
            return False
        
        # Check all regular tiles have same number
        number = regular[0].number
        if any(t.number != number for t in regular):
            return False
        
        # Check all regular tiles have different colors
        colors = set(t.color for t in regular)
        if len(colors) != len(regular):
            return False
        
        # Valid if we have 3-4 unique colors (jokers fill in missing ones)
        total_unique_colors = len(colors) + len(jokers)
        return 3 <= total_unique_colors <= 4
    
    @classmethod
    def calculate_value(cls, tiles: List[Tile]) -> int:
        """Calculate the point value of a meld.
        
        For runs: sum of all tile numbers (jokers take value of replaced tile)
        For groups: sum of tile numbers (all same number)
        
        Args:
            tiles: List of tiles in the meld
            
        Returns:
            Total point value
        """
        if not tiles:
            return 0
        
        if cls.is_valid_run(tiles):
            return cls._calculate_run_value(tiles)
        elif cls.is_valid_group(tiles):
            return cls._calculate_group_value(tiles)
        else:
            # Invalid meld - just sum what we can
            return sum(t.get_value() for t in tiles)
    
    @classmethod
    def _calculate_run_value(cls, tiles: List[Tile]) -> int:
        """Calculate value of a run."""
        jokers = [t for t in tiles if t.is_joker]
        regular = [t for t in tiles if not t.is_joker]
        
        if not regular:
            return 0
        
        # Sort regular tiles
        regular_sorted = sorted(regular, key=lambda t: t.number if t.number is not None else 0)
        
        # Calculate gaps and assign joker values
        total = sum(t.number if t.number is not None else 0 for t in regular_sorted)
        
        # Assign jokers to fill gaps optimally
        joker_idx = 0
        for i in range(len(regular_sorted) - 1):
            gap_start = regular_sorted[i].number if regular_sorted[i].number is not None else 0
            gap_end = regular_sorted[i + 1].number if regular_sorted[i + 1].number is not None else 0
            
            for val in range(gap_start + 1, gap_end):
                if joker_idx < len(jokers):
                    total += val
                    joker_idx += 1
        
        return total
    
    @classmethod
    def _calculate_group_value(cls, tiles: List[Tile]) -> int:
        """Calculate value of a group."""
        # All tiles have the same number
        regular = [t for t in tiles if not t.is_joker]
        if not regular:
            return 0
        
        number = regular[0].number if regular[0].number is not None else 0
        return number * len(tiles)
    
    @classmethod
    def can_add_tile(cls, meld_tiles: List[Tile], tile: Tile) -> bool:
        """Check if a tile can be added to an existing meld.
        
        Args:
            meld_tiles: Current tiles in the meld
            tile: Tile to potentially add
            
        Returns:
            True if tile can be added while maintaining validity
        """
        new_tiles = meld_tiles + [tile]
        return cls.is_valid(new_tiles)
    
    @classmethod
    def get_valid_additions(cls, meld_tiles: List[Tile], hand: List[Tile]) -> List[Tile]:
        """Get list of tiles from hand that can be added to this meld.
        
        Args:
            meld_tiles: Current tiles in the meld
            hand: Player's hand to check
            
        Returns:
            List of tiles that can be added
        """
        return [tile for tile in hand if cls.can_add_tile(meld_tiles, tile)]
    
    def get_tiles(self) -> List[Tile]:
        """Return copy of tiles in this meld."""
        return [tile.copy() for tile in self.tiles]
    
    def get_display_tiles(self) -> List[Tile]:
        """Return tiles sorted for optimal display.
        
        For runs: sorted by number with jokers filling gaps
        For groups: sorted consistently by color
        """
        if self.is_valid_run(self.tiles):
            return self._sort_run_for_display()
        else:
            # For groups, just sort by color for consistency
            return sorted(self.tiles, key=lambda t: (t.color or 'z', t.number or 0))
    
    def _sort_run_for_display(self) -> List[Tile]:
        """Sort run tiles with jokers in proper positions."""
        jokers = [t for t in self.tiles if t.is_joker]
        regular = [t for t in self.tiles if not t.is_joker]
        
        if not regular:
            return self.tiles[:]
        
        # Sort regular tiles by number
        regular_sorted = sorted(regular, key=lambda t: t.number if t.number is not None else 0)
        
        # Build the sorted list with jokers filling gaps
        result = []
        joker_idx = 0
        
        for i, tile in enumerate(regular_sorted):
            result.append(tile)
            
            # Check if there's a gap after this tile
            if i < len(regular_sorted) - 1:
                current_num = tile.number if tile.number is not None else 0
                next_num = regular_sorted[i + 1].number if regular_sorted[i + 1].number is not None else 0
                gap = next_num - current_num - 1
                
                # Add jokers to fill the gap
                for _ in range(gap):
                    if joker_idx < len(jokers):
                        result.append(jokers[joker_idx])
                        joker_idx += 1
        
        # Add any remaining jokers at the end (for extending the run)
        while joker_idx < len(jokers):
            result.append(jokers[joker_idx])
            joker_idx += 1
        
        return result
    
    def copy(self) -> 'Meld':
        """Create a copy of this meld."""
        return Meld(self.get_tiles())
    
    def __len__(self) -> int:
        """Return number of tiles in meld."""
        return len(self.tiles)
    
    def __repr__(self) -> str:
        """String representation of the meld."""
        tile_strs = [str(t) for t in self.tiles]
        return f"Meld([{', '.join(tile_strs)}])"
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another meld."""
        if not isinstance(other, Meld):
            return NotImplemented
        # Compare sorted tiles to handle different orderings
        self_sorted = sorted(self.tiles, key=lambda t: (t.color or '', t.number or 0, t.is_joker))
        other_sorted = sorted(other.tiles, key=lambda t: (t.color or '', t.number or 0, t.is_joker))
        return self_sorted == other_sorted


def find_all_valid_melds(tiles: List[Tile], max_tiles: int = 6) -> List[List[Tile]]:
    """Find valid melds from a set of tiles.
    
    Checks all runs/groups of 3+ tiles.
    
    Args:
        tiles: List of tiles to check
        max_tiles: Maximum tiles in a meld (default 13)
        
    Returns:
        List of valid melds (each meld is a list of tiles)
    """
    valid_melds = []
    
    # Only check groups of 3-13 tiles
    from itertools import combinations
    
    # For performance, don't check ALL combinations if hand is large
    # But allow enough to find valid melds
    tiles_to_check = list(tiles)
    
    # Check sizes from 3 up to min(13, len(tiles))
    max_size = min(max_tiles, len(tiles))
    
    for size in range(3, max_size + 1):
        for combo in combinations(tiles_to_check, size):
            tile_list = list(combo)
            if Meld.is_valid(tile_list):
                valid_melds.append(tile_list)
    
    return valid_melds


def find_best_initial_meld(tiles: List[Tile]) -> Optional[List[Tile]]:
    """Find the highest value valid meld for initial play (must be >= 30 points).
    
    Args:
        tiles: List of tiles in hand
        
    Returns:
        Best valid meld (list of tiles) with value >= 30, or None
    """
    valid_melds = find_all_valid_melds(tiles)
    
    best_meld = None
    best_value = 30  # Minimum threshold
    
    for meld_tiles in valid_melds:
        value = Meld.calculate_value(meld_tiles)
        if value >= best_value:
            best_meld = meld_tiles
            best_value = value
    
    return best_meld


def can_form_initial_meld(tiles: List[Tile]) -> bool:
    """Check if player can form an initial meld of 30+ points.
    
    Args:
        tiles: List of tiles in hand
        
    Returns:
        True if initial meld possible with value >= 30
    """
    return find_best_initial_meld(tiles) is not None
