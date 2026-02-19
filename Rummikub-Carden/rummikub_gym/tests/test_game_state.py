"""Tests for GameState."""

import pytest
from rummikub_gym.envs.game_state import GameState
from rummikub_gym.envs.tile import Tile


class TestGameState:
    """Test cases for GameState class."""
    
    def test_init(self):
        """Test game state initialization."""
        state = GameState(num_players=2)
        assert state.num_players == 2
        assert state.tile_pool.remaining() == 106
        assert len(state.player_hands[0]) == 0
        assert len(state.player_hands[1]) == 0
    
    def test_invalid_num_players(self):
        """Test that invalid player count raises error."""
        with pytest.raises(ValueError):
            GameState(num_players=1)
        with pytest.raises(ValueError):
            GameState(num_players=5)
    
    def test_reset(self):
        """Test game reset."""
        state = GameState(num_players=2)
        state.reset()
        
        # After reset, each player should have 14 tiles
        assert len(state.player_hands[0]) == 14
        assert len(state.player_hands[1]) == 14
        assert state.tile_pool.remaining() == 106 - 28
    
    def test_deal_initial_hands(self):
        """Test dealing initial hands."""
        state = GameState(num_players=2)
        state._deal_initial_hands()
        
        assert len(state.player_hands[0]) == 14
        assert len(state.player_hands[1]) == 14
    
    def test_draw_tile(self):
        """Test drawing a tile."""
        state = GameState(num_players=2)
        state.reset()
        
        initial_hand_size = len(state.player_hands[0])
        tile = state.draw_tile(0)
        
        assert tile is not None
        assert len(state.player_hands[0]) == initial_hand_size + 1
    
    def test_play_meld(self):
        """Test playing a valid meld."""
        state = GameState(num_players=2)
        state.reset()
        
        # Give player a valid high-value meld
        state.player_hands[0] = [
            Tile(color='red', number=10),
            Tile(color='red', number=11),
            Tile(color='red', number=12),
        ]
        
        success = state.play_meld(0, [0, 1, 2])
        assert success
        assert len(state.table_melds) == 1
        assert len(state.player_hands[0]) == 0
        assert state.has_initial_meld[0]
    
    def test_play_meld_insufficient_value(self):
        """Test playing meld with insufficient value for initial meld."""
        state = GameState(num_players=2)
        state.reset()
        
        # Give player a low-value meld
        state.player_hands[0] = [
            Tile(color='red', number=2),
            Tile(color='red', number=3),
            Tile(color='red', number=4),
        ]
        
        success = state.play_meld(0, [0, 1, 2])
        assert not success  # Should fail - need 30+ points for initial meld
    
    def test_play_meld_after_initial(self):
        """Test playing meld after initial meld is made."""
        state = GameState(num_players=2)
        state.reset()
        
        # Mark initial meld as done
        state.has_initial_meld[0] = True
        
        # Give player a low-value meld
        state.player_hands[0] = [
            Tile(color='red', number=2),
            Tile(color='red', number=3),
            Tile(color='red', number=4),
        ]
        
        success = state.play_meld(0, [0, 1, 2])
        assert success  # Should succeed after initial meld
    
    def test_add_to_meld(self):
        """Test adding tile to existing meld."""
        state = GameState(num_players=2)
        state.reset()
        
        # Set up initial state
        state.has_initial_meld[0] = True
        state.table_melds = []  # We'll add a meld manually
        from rummikub_gym.envs.meld import Meld
        state.table_melds.append(Meld([
            Tile(color='red', number=3),
            Tile(color='red', number=4),
            Tile(color='red', number=5),
        ]))
        
        # Give player a tile that can be added
        state.player_hands[0] = [Tile(color='red', number=6)]
        
        success = state.add_to_meld(0, 0, 0)
        assert success
        assert len(state.table_melds[0].tiles) == 4
        assert len(state.player_hands[0]) == 0
    
    def test_add_to_meld_before_initial(self):
        """Test that adding to meld fails before initial meld."""
        state = GameState(num_players=2)
        state.reset()
        
        from rummikub_gym.envs.meld import Meld
        state.table_melds.append(Meld([
            Tile(color='red', number=3),
            Tile(color='red', number=4),
            Tile(color='red', number=5),
        ]))
        
        state.player_hands[0] = [Tile(color='red', number=6)]
        
        success = state.add_to_meld(0, 0, 0)
        assert not success  # Should fail - no initial meld yet
    
    def test_can_go_out(self):
        """Test checking if player can go out."""
        state = GameState(num_players=2)
        state.reset()
        
        # Empty hand should allow going out
        state.player_hands[0] = []
        assert state.can_go_out(0)
        
        # Non-empty hand should not allow going out
        state.player_hands[0] = [Tile(color='red', number=5)]
        assert not state.can_go_out(0)
    
    def test_declare_out(self):
        """Test declaring Rummikub."""
        state = GameState(num_players=2)
        state.reset()
        
        state.player_hands[0] = []
        success = state.declare_out(0)
        
        assert success
        assert state.game_over
        assert state.winner == 0
    
    def test_next_player(self):
        """Test advancing to next player."""
        state = GameState(num_players=2)
        state.reset()
        
        assert state.current_player == 0
        state.next_player()
        assert state.current_player == 1
        state.next_player()
        assert state.current_player == 0
    
    def test_calculate_hand_value(self):
        """Test calculating hand value."""
        state = GameState(num_players=2)
        state.reset()
        
        state.player_hands[0] = [
            Tile(color='red', number=5),
            Tile(color='blue', number=10),
            Tile.create_joker(),
        ]
        
        value = state.calculate_hand_value(0)
        assert value == 45  # 5 + 10 + 30 (joker)
    
    def test_clone(self):
        """Test cloning game state."""
        state = GameState(num_players=2)
        state.reset()
        
        cloned = state.clone()
        
        assert cloned.num_players == state.num_players
        assert cloned.current_player == state.current_player
        assert len(cloned.player_hands[0]) == len(state.player_hands[0])
    
    def test_validate_table_state(self):
        """Test validating table state."""
        state = GameState(num_players=2)
        state.reset()
        
        # Initially valid
        assert state.validate_table_state()
        
        # Add a valid meld
        from rummikub_gym.envs.meld import Meld
        state.table_melds.append(Meld([
            Tile(color='red', number=3),
            Tile(color='red', number=4),
            Tile(color='red', number=5),
        ]))
        
        assert state.validate_table_state()
    
    def test_get_valid_actions(self):
        """Test getting valid actions."""
        state = GameState(num_players=2)
        state.reset()
        
        # Give player a valid meld
        state.player_hands[0] = [
            Tile(color='red', number=10),
            Tile(color='red', number=11),
            Tile(color='red', number=12),
        ]
        
        actions = state.get_valid_actions_for_player(0)
        
        # Should have at least DRAW and PLAY_NEW_MELD
        assert len(actions) >= 2
        
        action_types = [a['type'] for a in actions]
        assert 'DRAW' in action_types
        assert 'PLAY_NEW_MELD' in action_types
