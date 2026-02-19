"""Main Rummikub Gymnasium environment."""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .tile import Tile, TileSet
from .meld import Meld
from .game_state import GameState
from .action_space_utils import ActionSpace, apply_action


class RummikubEnv(gym.Env):
    """Rummikub environment for reinforcement learning.
    
    Implements the OpenAI Gymnasium API for the tile-based game Rummikub.
    
    Observation Space:
        Dict with keys:
        - 'hand': Box(0, 55, (14,)) - tiles in current player's hand (encoded)
        - 'hand_mask': Box(0, 1, (14,)) - which hand slots are valid
        - 'table_melds': Box(0, 55, (20, 13)) - tiles on table (max 20 melds, 13 tiles each)
        - 'table_meld_mask': Box(0, 1, (20, 13)) - which table slots are valid
        - 'pool_size': Discrete(107) - tiles remaining in pool
        - 'opponent_hand_sizes': Box(0, 106, (num_players-1,)) - opponent hand sizes
        - 'has_initial_meld': MultiBinary(num_players) - who has made initial meld
        - 'current_player': Discrete(num_players) - whose turn it is
    
    Action Space:
        Discrete with encoded actions:
        - 0: DRAW
        - 1: DECLARE_OUT
        - 2+: PLAY_NEW_MELD or ADD_TO_MELD
    
    Reward:
        - +100 for winning (going out)
        - -hand_value for losing
        - -0.1 per turn (to encourage faster play)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, num_players: int = 2, render_mode: Optional[str] = None):
        """Initialize the Rummikub environment.
        
        Args:
            num_players: Number of players (2-4)
            render_mode: Rendering mode ('human' or None)
        """
        super().__init__()
        
        self.num_players = num_players
        self.render_mode = render_mode
        
        # Initialize game state
        self.game_state = GameState(num_players)
        self.action_space_util = ActionSpace()
        
        # Define observation space
        self.observation_space = self._create_observation_space()
        
        # Define action space
        self.action_space = spaces.Discrete(self.action_space_util.get_action_space_size())
        
        # Track turn for reward calculation
        self.turns_taken = 0
    
    def _create_observation_space(self) -> spaces.Dict:
        """Create the observation space."""
        max_hand_size = 14
        max_melds = 20
        max_tiles_per_meld = 13
        
        return spaces.Dict({
            'hand': spaces.Box(low=0, high=55, shape=(max_hand_size,), dtype=np.int32),
            'hand_mask': spaces.Box(low=0, high=1, shape=(max_hand_size,), dtype=np.int32),
            'table_melds': spaces.Box(
                low=0, high=55, 
                shape=(max_melds, max_tiles_per_meld), 
                dtype=np.int32
            ),
            'table_meld_mask': spaces.Box(
                low=0, high=1, 
                shape=(max_melds, max_tiles_per_meld), 
                dtype=np.int32
            ),
            'pool_size': spaces.Discrete(107),
            'opponent_hand_sizes': spaces.Box(
                low=0, high=106, 
                shape=(self.num_players - 1,), 
                dtype=np.int32
            ),
            'has_initial_meld': spaces.MultiBinary(self.num_players),
            'current_player': spaces.Discrete(self.num_players),
        })
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset game state
        self.game_state.reset(seed=seed)
        self.turns_taken = 0
        
        # Get observation for first player
        obs = self._get_observation(self.game_state.current_player)
        info = {
            'current_player': self.game_state.current_player,
            'valid_actions': self.action_space_util.get_valid_actions(
                self.game_state, self.game_state.current_player
            )
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: Encoded action integer
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Decode action
        action_dict = self.action_space_util.decode_action(action)
        
        # Get current player before action
        current_player = self.game_state.current_player
        
        # Apply action
        success, message = apply_action(
            self.game_state, current_player, action_dict
        )
        
        # Calculate reward
        reward = self._calculate_reward(current_player, success)
        
        # Check if game is over
        terminated = self.game_state.game_over
        truncated = False
        
        # Move to next player if game not over
        if not terminated:
            self.game_state.next_player()
            self.turns_taken += 1
        
        # Get observation for next player
        next_player = self.game_state.current_player
        obs = self._get_observation(next_player)
        
        info = {
            'success': success,
            'message': message,
            'current_player': next_player,
            'valid_actions': self.action_space_util.get_valid_actions(
                self.game_state, next_player
            ) if not terminated else []
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, player_id: int) -> Dict:
        """Get observation for a specific player.
        
        Args:
            player_id: Player to get observation for
            
        Returns:
            Observation dictionary
        """
        # Encode hand
        hand = self.game_state.player_hands[player_id]
        hand_encoded = TileSet.encode_hand(hand, max_size=14)
        hand_mask = [1 if code >= 0 else 0 for code in hand_encoded]
        
        # Encode table melds
        table_melds = np.zeros((20, 13), dtype=np.int32)
        table_meld_mask = np.zeros((20, 13), dtype=np.int32)
        
        for i, meld in enumerate(self.game_state.table_melds[:20]):
            for j, tile in enumerate(meld.tiles[:13]):
                table_melds[i, j] = tile.encode()
                table_meld_mask[i, j] = 1
        
        # Get opponent hand sizes
        opponent_sizes = self.game_state.get_opponent_hand_sizes()
        
        return {
            'hand': np.array(hand_encoded, dtype=np.int32),
            'hand_mask': np.array(hand_mask, dtype=np.int32),
            'table_melds': table_melds,
            'table_meld_mask': table_meld_mask,
            'pool_size': self.game_state.tile_pool.remaining(),
            'opponent_hand_sizes': np.array(opponent_sizes, dtype=np.int32),
            'has_initial_meld': np.array(
                [self.game_state.has_initial_meld[i] for i in range(self.num_players)],
                dtype=np.int32
            ),
            'current_player': player_id,
        }
    
    def _calculate_reward(self, player_id: int, action_success: bool) -> float:
        """Calculate reward for a player.
        
        Args:
            player_id: Player who took the action
            action_success: Whether the action was successful
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Small negative reward per turn to encourage faster play
        reward -= 0.1
        
        # Check if game is over
        if self.game_state.game_over:
            if self.game_state.winner == player_id:
                # Winner gets +100
                reward += 100.0
            else:
                # Loser gets negative of their hand value
                hand_value = self.game_state.calculate_hand_value(player_id)
                reward -= float(hand_value)
        
        # Small positive reward for successful meld plays
        if action_success:
            reward += 0.5
        
        return reward
    
    def render(self) -> None:
        """Render the current game state."""
        if self.render_mode == 'human':
            self._render_human()
    
    def _render_human(self) -> None:
        """Render game state in human-readable format."""
        print("\n" + "=" * 50)
        print(f"RUMMIKUB - Turn {self.turns_taken}")
        print("=" * 50)
        
        # Show table
        print("\nTABLE MELDS:")
        if self.game_state.table_melds:
            for i, meld in enumerate(self.game_state.table_melds):
                tiles_str = ", ".join(str(t) for t in meld.tiles)
                value = Meld.calculate_value(meld.tiles)
                print(f"  Meld {i}: [{tiles_str}] (value: {value})")
        else:
            print("  (empty)")
        
        # Show each player's hand
        print(f"\nTILES IN POOL: {self.game_state.tile_pool.remaining()}")
        
        for player_id in range(self.num_players):
            hand = self.game_state.player_hands[player_id]
            hand_value = self.game_state.calculate_hand_value(player_id)
            initial_meld = "✓" if self.game_state.has_initial_meld[player_id] else "✗"
            
            marker = " >>>" if player_id == self.game_state.current_player else ""
            print(f"\nPlayer {player_id}{marker}:")
            print(f"  Initial meld: {initial_meld}")
            print(f"  Hand ({len(hand)} tiles, value: {hand_value}):")
            
            # Group tiles by color for better readability
            tiles_by_color = {}
            for tile in hand:
                if tile.is_joker:
                    color = 'joker'
                else:
                    color = tile.color
                
                if color not in tiles_by_color:
                    tiles_by_color[color] = []
                tiles_by_color[color].append(tile)
            
            for color, tiles in sorted(tiles_by_color.items()):
                tiles_str = ", ".join(str(t) for t in sorted(tiles, key=lambda t: t.number if not t.is_joker else 0))
                print(f"    {color.capitalize()}: {tiles_str}")
        
        print("\n" + "=" * 50)
    
    def get_valid_action_mask(self) -> np.ndarray:
        """Get binary mask of valid actions for current player.
        
        Returns:
            Binary numpy array where 1 = valid action
        """
        return self.action_space_util.create_action_mask(
            self.game_state, self.game_state.current_player
        )
    
    def close(self) -> None:
        """Clean up resources."""
        pass
