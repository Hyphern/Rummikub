"""Tests for Meld validation."""

import pytest
from rummikub_gym.envs.tile import Tile
from rummikub_gym.envs.meld import Meld, find_all_valid_melds, find_best_initial_meld, can_form_initial_meld


class TestMeldValidation:
    """Test cases for meld validation."""
    
    def test_valid_run(self):
        """Test valid run (consecutive same-color tiles)."""
        tiles = [
            Tile(color='red', number=3),
            Tile(color='red', number=4),
            Tile(color='red', number=5),
        ]
        assert Meld.is_valid_run(tiles)
        assert Meld.is_valid(tiles)
    
    def test_invalid_run_wrong_color(self):
        """Test invalid run with different colors."""
        tiles = [
            Tile(color='red', number=3),
            Tile(color='blue', number=4),
            Tile(color='red', number=5),
        ]
        assert not Meld.is_valid_run(tiles)
    
    def test_invalid_run_not_consecutive(self):
        """Test invalid run with gap."""
        tiles = [
            Tile(color='red', number=3),
            Tile(color='red', number=5),
            Tile(color='red', number=6),
        ]
        assert not Meld.is_valid_run(tiles)
    
    def test_valid_run_with_joker(self):
        """Test valid run using joker to fill gap."""
        tiles = [
            Tile(color='red', number=3),
            Tile.create_joker(),
            Tile(color='red', number=5),
        ]
        assert Meld.is_valid_run(tiles)
    
    def test_valid_group(self):
        """Test valid group (same number, different colors)."""
        tiles = [
            Tile(color='red', number=7),
            Tile(color='blue', number=7),
            Tile(color='black', number=7),
        ]
        assert Meld.is_valid_group(tiles)
        assert Meld.is_valid(tiles)
    
    def test_valid_group_four_colors(self):
        """Test valid group with all four colors."""
        tiles = [
            Tile(color='red', number=7),
            Tile(color='blue', number=7),
            Tile(color='black', number=7),
            Tile(color='orange', number=7),
        ]
        assert Meld.is_valid_group(tiles)
    
    def test_invalid_group_same_color(self):
        """Test invalid group with duplicate colors."""
        tiles = [
            Tile(color='red', number=7),
            Tile(color='red', number=7),
            Tile(color='blue', number=7),
        ]
        assert not Meld.is_valid_group(tiles)
    
    def test_invalid_group_different_numbers(self):
        """Test invalid group with different numbers."""
        tiles = [
            Tile(color='red', number=7),
            Tile(color='blue', number=8),
            Tile(color='black', number=7),
        ]
        assert not Meld.is_valid_group(tiles)
    
    def test_valid_group_with_joker(self):
        """Test valid group using joker."""
        tiles = [
            Tile(color='red', number=7),
            Tile(color='blue', number=7),
            Tile.create_joker(),
        ]
        assert Meld.is_valid_group(tiles)
    
    def test_meld_too_short(self):
        """Test that meld with less than 3 tiles is invalid."""
        tiles = [
            Tile(color='red', number=3),
            Tile(color='red', number=4),
        ]
        assert not Meld.is_valid(tiles)
    
    def test_run_value_calculation(self):
        """Test calculating run value."""
        tiles = [
            Tile(color='red', number=3),
            Tile(color='red', number=4),
            Tile(color='red', number=5),
        ]
        value = Meld.calculate_value(tiles)
        assert value == 12  # 3 + 4 + 5
    
    def test_group_value_calculation(self):
        """Test calculating group value."""
        tiles = [
            Tile(color='red', number=7),
            Tile(color='blue', number=7),
            Tile(color='black', number=7),
        ]
        value = Meld.calculate_value(tiles)
        assert value == 21  # 7 * 3
    
    def test_run_value_with_joker(self):
        """Test calculating run value with joker."""
        tiles = [
            Tile(color='red', number=3),
            Tile.create_joker(),  # Represents 4
            Tile(color='red', number=5),
        ]
        value = Meld.calculate_value(tiles)
        assert value == 12  # 3 + 4 + 5
    
    def test_can_add_tile_to_run(self):
        """Test adding tile to existing run."""
        meld_tiles = [
            Tile(color='red', number=3),
            Tile(color='red', number=4),
            Tile(color='red', number=5),
        ]
        new_tile = Tile(color='red', number=6)
        assert Meld.can_add_tile(meld_tiles, new_tile)
    
    def test_can_add_tile_to_group(self):
        """Test adding tile to existing group."""
        meld_tiles = [
            Tile(color='red', number=7),
            Tile(color='blue', number=7),
            Tile(color='black', number=7),
        ]
        new_tile = Tile(color='orange', number=7)
        assert Meld.can_add_tile(meld_tiles, new_tile)


class TestFindValidMelds:
    """Test cases for finding valid melds."""
    
    def test_find_all_valid_melds(self):
        """Test finding all valid melds from tiles."""
        tiles = [
            Tile(color='red', number=3),
            Tile(color='red', number=4),
            Tile(color='red', number=5),
            Tile(color='blue', number=7),
            Tile(color='black', number=7),
            Tile(color='orange', number=7),
        ]
        
        valid_melds = find_all_valid_melds(tiles)
        assert len(valid_melds) >= 2  # Should find the run and the group
    
    def test_find_best_initial_meld(self):
        """Test finding best initial meld."""
        tiles = [
            Tile(color='red', number=10),
            Tile(color='red', number=11),
            Tile(color='red', number=12),
        ]
        
        best_meld = find_best_initial_meld(tiles)
        assert best_meld is not None
        assert Meld.calculate_value(best_meld) >= 30
    
    def test_find_best_initial_meld_insufficient_value(self):
        """Test when no initial meld meets 30-point threshold."""
        tiles = [
            Tile(color='red', number=2),
            Tile(color='red', number=3),
            Tile(color='red', number=4),
        ]
        
        best_meld = find_best_initial_meld(tiles)
        assert best_meld is None
    
    def test_can_form_initial_meld(self):
        """Test checking if initial meld is possible."""
        tiles = [
            Tile(color='red', number=10),
            Tile(color='red', number=11),
            Tile(color='red', number=12),
        ]
        assert can_form_initial_meld(tiles)
        
        low_value_tiles = [
            Tile(color='red', number=2),
            Tile(color='red', number=3),
            Tile(color='red', number=4),
        ]
        assert not can_form_initial_meld(low_value_tiles)
