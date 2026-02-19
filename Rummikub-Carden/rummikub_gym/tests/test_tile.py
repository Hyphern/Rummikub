"""Tests for Tile class."""

import pytest
from rummikub_gym.envs.tile import Tile, TileSet, sort_tiles_by_number, sort_tiles_by_color_and_number


class TestTile:
    """Test cases for Tile class."""
    
    def test_create_regular_tile(self):
        """Test creating a regular tile."""
        tile = Tile(color='red', number=5)
        assert tile.color == 'red'
        assert tile.number == 5
        assert not tile.is_joker
    
    def test_create_joker(self):
        """Test creating a joker tile."""
        tile = Tile.create_joker()
        assert tile.is_joker
        assert tile.color is None
        assert tile.number is None
    
    def test_invalid_color(self):
        """Test that invalid color raises error."""
        with pytest.raises(ValueError):
            Tile(color='pink', number=5)
    
    def test_invalid_number(self):
        """Test that invalid number raises error."""
        with pytest.raises(ValueError):
            Tile(color='red', number=15)
        with pytest.raises(ValueError):
            Tile(color='red', number=0)
    
    def test_joker_with_color(self):
        """Test that joker with color raises error."""
        with pytest.raises(ValueError):
            Tile(color='red', number=None, is_joker=True)
    
    def test_tile_equality(self):
        """Test tile equality."""
        tile1 = Tile(color='red', number=5)
        tile2 = Tile(color='red', number=5)
        tile3 = Tile(color='blue', number=5)
        
        assert tile1 == tile2
        assert tile1 != tile3
    
    def test_tile_hash(self):
        """Test tile can be used in sets/dicts."""
        tile1 = Tile(color='red', number=5)
        tile2 = Tile(color='red', number=5)
        
        s = {tile1}
        assert tile2 in s
    
    def test_tile_value(self):
        """Test getting tile value."""
        tile = Tile(color='red', number=7)
        assert tile.get_value() == 7
        
        joker = Tile.create_joker()
        assert joker.get_value() == 0
    
    def test_tile_encode_decode(self):
        """Test tile encoding and decoding."""
        # Test regular tile
        tile = Tile(color='red', number=5)
        encoded = tile.encode()
        decoded = Tile.decode(encoded)
        assert tile == decoded
        
        # Test joker
        joker = Tile.create_joker()
        encoded = joker.encode()
        decoded = Tile.decode(encoded)
        assert decoded.is_joker


class TestTileSet:
    """Test cases for TileSet class."""
    
    def test_create_standard_set(self):
        """Test creating standard 106 tile set."""
        tile_set = TileSet()
        assert tile_set.remaining() == 106
    
    def test_draw_tiles(self):
        """Test drawing tiles."""
        tile_set = TileSet()
        tiles = tile_set.draw(14)
        assert len(tiles) == 14
        assert tile_set.remaining() == 92
    
    def test_draw_one(self):
        """Test drawing single tile."""
        tile_set = TileSet()
        tile = tile_set.draw_one()
        assert tile is not None
        assert tile_set.remaining() == 105
    
    def test_draw_empty_pool(self):
        """Test drawing from empty pool."""
        tile_set = TileSet()
        # Draw all tiles
        for _ in range(106):
            tile_set.draw_one()
        
        assert tile_set.draw_one() is None
    
    def test_shuffle(self):
        """Test shuffling tiles."""
        tile_set1 = TileSet()
        tile_set2 = TileSet()
        
        tile_set1.shuffle()
        
        # After shuffle, order should be different (with high probability)
        # We can't test this deterministically, but we can test it doesn't crash
        assert tile_set1.remaining() == 106
    
    def test_reset(self):
        """Test resetting tile set."""
        tile_set = TileSet()
        tile_set.draw(50)
        assert tile_set.remaining() == 56
        
        tile_set.reset()
        assert tile_set.remaining() == 106


class TestTileSorting:
    """Test cases for tile sorting functions."""
    
    def test_sort_by_number(self):
        """Test sorting tiles by number."""
        tiles = [
            Tile(color='red', number=5),
            Tile(color='blue', number=2),
            Tile(color='black', number=10),
            Tile.create_joker(),
        ]
        
        sorted_tiles = sort_tiles_by_number(tiles)
        numbers = [t.number for t in sorted_tiles if not t.is_joker]
        assert numbers == [2, 5, 10]
        assert sorted_tiles[-1].is_joker  # Joker should be last
    
    def test_sort_by_color_and_number(self):
        """Test sorting tiles by color then number."""
        tiles = [
            Tile(color='orange', number=5),
            Tile(color='red', number=2),
            Tile(color='blue', number=10),
            Tile(color='red', number=5),
            Tile.create_joker(),
        ]
        
        sorted_tiles = sort_tiles_by_color_and_number(tiles)
        non_jokers = [t for t in sorted_tiles if not t.is_joker]
        
        # Should be sorted by color order: red, blue, black, orange
        assert non_jokers[0].color == 'red'
        assert non_jokers[1].color == 'red'
        assert non_jokers[2].color == 'blue'
        assert non_jokers[3].color == 'orange'
        assert sorted_tiles[-1].is_joker
