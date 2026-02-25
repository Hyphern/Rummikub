"""Tile class and utilities for Rummikub."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set
import random


@dataclass(frozen=True)
class Tile:
    """Represents a single Rummikub tile.
    
    Attributes:
        color: One of 'red', 'blue', 'black', 'orange' (or None for joker)
        number: Integer 1-13 (or None for joker)
        is_joker: Boolean indicating if this is a joker tile
    """
    color: Optional[str]
    number: Optional[int]
    is_joker: bool = False
    
    # Valid colors for regular tiles
    VALID_COLORS = {'red', 'blue', 'black', 'orange'}
    
    def __post_init__(self):
        """Validate tile attributes."""
        if self.is_joker:
            if self.color is not None or self.number is not None:
                raise ValueError("Joker tiles must have color=None and number=None")
        else:
            if self.color not in self.VALID_COLORS:
                raise ValueError(f"Invalid color: {self.color}. Must be one of {self.VALID_COLORS}")
            if self.number is None or not (1 <= self.number <= 13):
                raise ValueError(f"Invalid number: {self.number}. Must be 1-13")
    
    def __repr__(self) -> str:
        if self.is_joker:
            return "Joker"
        return f"{self.color.capitalize()} {self.number}"  # type: ignore
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tile):
            return NotImplemented
        return (self.color == other.color and 
                self.number == other.number and 
                self.is_joker == other.is_joker)
    
    def __hash__(self) -> int:
        return hash((self.color, self.number, self.is_joker))
    
    def get_value(self) -> int:
        """Return the point value of this tile.
        
        For regular tiles, returns the number.
        For jokers, returns 0 (value is determined by context in melds).
        """
        if self.is_joker:
            return 0
        return self.number if self.number is not None else 0
    
    def encode(self) -> int:
        """Encode tile as integer 0-55.
        
        Encoding:
        - 0-51: Regular tiles (13 numbers × 4 colors)
          - Colors ordered: red=0, blue=1, black=2, orange=3
          - For each color: numbers 1-13
        - 52-53: Jokers (2 jokers)
        
        Returns:
            Integer encoding of the tile
        """
        if self.is_joker:
            # Two jokers: return 52 or 53
            # We'll distinguish them by using a counter in TileSet
            return 52
        
        color_idx = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}[self.color]  # type: ignore
        return color_idx * 13 + (self.number - 1)  # type: ignore
    
    @classmethod
    def decode(cls, code: int) -> Tile:
        """Decode integer to Tile.
        
        Args:
            code: Integer 0-55
            
        Returns:
            Decoded Tile
        """
        if code >= 52:
            return cls.create_joker()
        
        color_idx = code // 13
        number = (code % 13) + 1
        color = ['red', 'blue', 'black', 'orange'][color_idx]
        return cls(color=color, number=number)
    
    @classmethod
    def create_joker(cls) -> Tile:
        """Create a joker tile."""
        return cls(color=None, number=None, is_joker=True)
    
    def copy(self) -> Tile:
        """Create a copy of this tile."""
        return Tile(color=self.color, number=self.number, is_joker=self.is_joker)


class TileSet:
    """Manages the complete set of Rummikub tiles."""
    
    def __init__(self, include_jokers: bool = False):
        """Initialize with standard Rummikub tile distribution.
        
        Args:
            include_jokers: Whether to include jokers in the tile set.
                          Default is False to work with worm.py solver.
        """
        self.tiles: List[Tile] = []
        self.include_jokers = include_jokers
        self._create_standard_set()
    
    def _create_standard_set(self):
        """Create the tile set."""
        # Two sets of 1-13 in each of 4 colors
        for _ in range(2):
            for color in Tile.VALID_COLORS:
                for number in range(1, 14):
                    self.tiles.append(Tile(color=color, number=number))
        
        # Add jokers only if requested
        if self.include_jokers:
            self.tiles.append(Tile.create_joker())
            self.tiles.append(Tile.create_joker())
    
    def shuffle(self) -> None:
        """Shuffle the tiles in place."""
        random.shuffle(self.tiles)
    
    def draw(self, count: int = 1) -> List[Tile]:
        """Draw tiles from the pool.
        
        Args:
            count: Number of tiles to draw
            
        Returns:
            List of drawn tiles
            
        Raises:
            ValueError: If not enough tiles remain
        """
        if count > len(self.tiles):
            raise ValueError(f"Cannot draw {count} tiles, only {len(self.tiles)} remain")
        
        drawn = self.tiles[:count]
        self.tiles = self.tiles[count:]
        return drawn
    
    def draw_one(self) -> Optional[Tile]:
        """Draw a single tile from the pool.
        
        Returns:
            The drawn tile, or None if pool is empty
        """
        if not self.tiles:
            return None
        return self.tiles.pop(0)
    
    def remaining(self) -> int:
        """Return number of tiles remaining in pool."""
        return len(self.tiles)
    
    def is_empty(self) -> bool:
        """Check if pool is empty."""
        return len(self.tiles) == 0
    
    def reset(self) -> None:
        """Reset to full tile set and shuffle."""
        self.tiles = []
        self._create_standard_set()
        self.shuffle()
    
    def total_tiles(self) -> int:
        """Return total number of tiles in a full set."""
        if self.include_jokers:
            return 106  # 104 regular + 2 jokers
        else:
            return 104  # Just regular tiles
    
    def get_all_tiles(self) -> List[Tile]:
        """Return a copy of all tiles."""
        return [tile.copy() for tile in self.tiles]


def sort_tiles_by_number(tiles: List[Tile]) -> List[Tile]:
    """Sort tiles by number (ascending), jokers last."""
    def sort_key(tile: Tile) -> tuple:
        if tile.is_joker:
            return (2, 0, 0)  # Jokers last
        return (0, tile.number, 0)
    return sorted(tiles, key=sort_key)


def sort_tiles_by_color_and_number(tiles: List[Tile]) -> List[Tile]:
    """Sort tiles by color then number, jokers last."""
    color_order = {'red': 0, 'blue': 1, 'black': 2, 'orange': 3}
    
    def sort_key(tile: Tile) -> tuple:
        if tile.is_joker:
            return (2, 0, 0)
        color_idx = color_order.get(tile.color, 99) if tile.color else 99
        num = tile.number if tile.number is not None else 0
        return (0, color_idx, num)
    
    return sorted(tiles, key=sort_key)


def sort_tiles_for_hand(tiles: List[Tile]) -> List[Tile]:
    """Sort tiles optimally for display in hand - by color then number."""
    return sort_tiles_by_color_and_number(tiles)


def count_tiles_by_number(tiles: List[Tile]) -> dict:
    """Count non-joker tiles by number."""
    counts = {}
    for tile in tiles:
        if not tile.is_joker:
            counts[tile.number] = counts.get(tile.number, 0) + 1
    return counts


def group_tiles_by_color(tiles: List[Tile]) -> dict:
    """Group non-joker tiles by color."""
    groups = {color: [] for color in Tile.VALID_COLORS}
    for tile in tiles:
        if not tile.is_joker and tile.color:
            groups[tile.color].append(tile)
    return groups
