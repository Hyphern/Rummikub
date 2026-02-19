"""Tests for RummikubEnv."""

import pytest
import numpy as np
import gymnasium as gym
from rummikub_gym.envs.rummikub_env import RummikubEnv


class TestRummikubEnv:
    """Test cases for RummikubEnv class."""
    
    def test_init(self):
        """Test environment initialization."""
        env = RummikubEnv(num_players=2)
        assert env.num_players == 2
        assert env.observation_space is not None
        assert env.action_space is not None
    
    def test_reset(self):
        """Test environment reset."""
        env = RummikubEnv(num_players=2)
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert 'hand' in obs
        assert 'table_melds' in obs
        assert 'pool_size' in obs
        assert info is not None
        assert 'valid_actions' in info
    
    def test_step_draw(self):
        """Test taking a DRAW action."""
        env = RummikubEnv(num_players=2)
        obs, info = env.reset(seed=42)
        
        # Get a valid DRAW action
        draw_action = 0  # DRAW is encoded as 0
        
        obs, reward, terminated, truncated, info = env.step(draw_action)
        
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info is not None
    
    def test_observation_space(self):
        """Test observation space structure."""
        env = RummikubEnv(num_players=2)
        obs, _ = env.reset(seed=42)
        
        # Check all expected keys exist
        expected_keys = [
            'hand', 'hand_mask', 'table_melds', 'table_meld_mask',
            'pool_size', 'opponent_hand_sizes', 'has_initial_meld', 'current_player'
        ]
        for key in expected_keys:
            assert key in obs
        
        # Check shapes
        assert obs['hand'].shape == (14,)
        assert obs['hand_mask'].shape == (14,)
        assert obs['table_melds'].shape == (20, 13)
        assert obs['table_meld_mask'].shape == (20, 13)
        assert obs['opponent_hand_sizes'].shape == (1,)  # 2 players - 1 = 1 opponent
    
    def test_action_mask(self):
        """Test getting valid action mask."""
        env = RummikubEnv(num_players=2)
        env.reset(seed=42)
        
        mask = env.get_valid_action_mask()
        
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.int32
        assert np.any(mask == 1)  # At least some actions should be valid
    
    def test_render_human(self):
        """Test human rendering (smoke test)."""
        env = RummikubEnv(num_players=2, render_mode='human')
        env.reset(seed=42)
        
        # Should not raise an error
        env.render()
    
    def test_multiple_players(self):
        """Test environment with different player counts."""
        for num_players in [2, 3, 4]:
            env = RummikubEnv(num_players=num_players)
            obs, _ = env.reset(seed=42)
            
            assert obs['opponent_hand_sizes'].shape == (num_players - 1,)
            assert len(obs['has_initial_meld']) == num_players
    
    def test_reward_structure(self):
        """Test that rewards are returned correctly."""
        env = RummikubEnv(num_players=2)
        obs, info = env.reset(seed=42)
        
        # Take a DRAW action
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert isinstance(reward, (int, float))
        # Reward should be a valid number (combination of -0.1 turn penalty + 0.5 success bonus)
        assert reward > -1.0 and reward < 1.0
    
    def test_info_contains_valid_actions(self):
        """Test that info contains valid actions after step."""
        env = RummikubEnv(num_players=2)
        obs, info = env.reset(seed=42)
        
        assert 'valid_actions' in info
        assert isinstance(info['valid_actions'], list)
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        if not terminated:
            assert 'valid_actions' in info
            assert isinstance(info['valid_actions'], list)


class TestGymnasiumAPI:
    """Test that environment follows Gymnasium API."""
    
    def test_gymnasium_registration(self):
        """Test that environment can be registered with Gymnasium."""
        # This test verifies the environment follows the Gymnasium API
        # by checking it has all required methods
        env = RummikubEnv()
        
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
    
    def test_reset_signature(self):
        """Test reset method signature."""
        env = RummikubEnv()
        
        # Test with seed
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert info is not None
        
        # Test without seed
        obs, info = env.reset()
        assert obs is not None
        assert info is not None
    
    def test_step_signature(self):
        """Test step method signature."""
        env = RummikubEnv()
        env.reset(seed=42)
        
        # Take a valid action (DRAW = 0)
        result = env.step(0)
        
        # Should return 5 values: obs, reward, terminated, truncated, info
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
