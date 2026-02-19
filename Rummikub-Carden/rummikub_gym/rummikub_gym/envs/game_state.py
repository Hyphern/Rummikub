"""Game state management for Rummikub."""

from typing import List, Dict, Optional, Set, Tuple
import copy
from .tile import Tile, TileSet
from .meld import Meld


class GameState:
    """Manages the complete state of a Rummikub game."""
    
    def __init__(self, num_players: int = 2):
        """Initialize game state.
        
        Args:
            num_players: Number of players (2-4)
            
        Raises:
            ValueError: If num_players not in valid range
        """
        if not (2 <= num_players <= 4):
            raise ValueError(f"Number of players must be 2-4, got {num_players}")
        
        self.num_players = num_players
        self.tile_pool = TileSet()
        self.table_melds: List[Meld] = []
        self.player_hands: Dict[int, List[Tile]] = {i: [] for i in range(num_players)}
        self.current_player = 0
        self.has_initial_meld: Dict[int, bool] = {i: False for i in range(num_players)}
        self.game_over = False
        self.winner: Optional[int] = None
        self.turn_count = 0
        self.max_turns = 1000  # Prevent infinite games
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset game to initial state.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            import random
            random.seed(seed)
        
        self.tile_pool.reset()
        self.table_melds = []
        self.player_hands = {i: [] for i in range(self.num_players)}
        self.current_player = 0
        self.has_initial_meld = {i: False for i in range(self.num_players)}
        self.game_over = False
        self.winner = None
        self.turn_count = 0
        
        # Deal initial hands
        self._deal_initial_hands()
    
    def _deal_initial_hands(self) -> None:
        """Deal 14 tiles to each player."""
        for player_id in range(self.num_players):
            self.player_hands[player_id] = self.tile_pool.draw(14)
    
    def validate_table_state(self) -> bool:
        """Validate that all melds on table are valid.
        
        Returns:
            True if all table melds are valid
        """
        for meld in self.table_melds:
            if not Meld.is_valid(meld.tiles):
                return False
        return True
    
    def calculate_hand_value(self, player_id: int) -> int:
        """Calculate total value of tiles in player's hand.
        
        Args:
            player_id: Player to check
            
        Returns:
            Sum of tile values (jokers = 30)
        """
        hand = self.player_hands[player_id]
        total = 0
        for tile in hand:
            if tile.is_joker:
                total += 30  # Jokers are worth 30 in hand
            else:
                total += tile.number if tile.number is not None else 0
        return total
    
    def clone(self) -> 'GameState':
        """Create a deep copy of the game state.
        
        Returns:
            Deep copy of this game state
        """
        new_state = GameState(self.num_players)
        new_state.tile_pool.tiles = [t.copy() for t in self.tile_pool.tiles]
        new_state.table_melds = [meld.copy() for meld in self.table_melds]
        new_state.player_hands = {
            pid: [t.copy() for t in hand] 
            for pid, hand in self.player_hands.items()
        }
        new_state.current_player = self.current_player
        new_state.has_initial_meld = self.has_initial_meld.copy()
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        new_state.turn_count = self.turn_count
        new_state.max_turns = self.max_turns
        return new_state
    
    def get_current_player_hand(self) -> List[Tile]:
        """Get the hand of the current player."""
        return self.player_hands[self.current_player]
    
    def get_opponent_hand_sizes(self) -> List[int]:
        """Get hand sizes of all opponents."""
        return [
            len(self.player_hands[pid]) 
            for pid in range(self.num_players) 
            if pid != self.current_player
        ]
    
    def draw_tile(self, player_id: int) -> Optional[Tile]:
        """Draw a tile from pool for a player.
        
        Args:
            player_id: Player drawing the tile
            
        Returns:
            The drawn tile, or None if pool is empty
        """
        tile = self.tile_pool.draw_one()
        if tile:
            self.player_hands[player_id].append(tile)
        return tile
    
    def play_meld(self, player_id: int, tile_indices: List[int]) -> bool:
        """Play a new meld from hand to table.
        
        Args:
            player_id: Player playing the meld
            tile_indices: Indices of tiles in player's hand to play
            
        Returns:
            True if meld was played successfully
        """
        hand = self.player_hands[player_id]
        
        # Get tiles from hand
        try:
            tiles = [hand[i] for i in tile_indices]
        except IndexError:
            return False
        
        # Validate meld
        if not Meld.is_valid(tiles):
            return False
        
        # Check initial meld requirement
        if not self.has_initial_meld[player_id]:
            value = Meld.calculate_value(tiles)
            if value < 30:
                return False
            self.has_initial_meld[player_id] = True
        
        # Remove tiles from hand
        # Remove in reverse order to maintain correct indices
        for idx in sorted(tile_indices, reverse=True):
            hand.pop(idx)
        
        # Add to table
        self.table_melds.append(Meld(tiles))
        return True
    
    def add_to_meld(self, player_id: int, tile_idx: int, meld_idx: int) -> bool:
        """Add a tile from hand to an existing table meld.
        
        Args:
            player_id: Player adding the tile
            tile_idx: Index of tile in player's hand
            meld_idx: Index of meld on table
            
        Returns:
            True if tile was added successfully
        """
        if not self.has_initial_meld[player_id]:
            return False  # Must have initial meld first
        
        hand = self.player_hands[player_id]
        
        if tile_idx >= len(hand):
            return False
        
        if meld_idx >= len(self.table_melds):
            return False
        
        tile = hand[tile_idx]
        meld = self.table_melds[meld_idx]
        
        if not Meld.can_add_tile(meld.tiles, tile):
            return False
        
        # Remove from hand and add to meld
        hand.pop(tile_idx)
        new_tiles = meld.tiles + [tile]
        self.table_melds[meld_idx] = Meld(new_tiles)
        return True
    
    def can_go_out(self, player_id: int) -> bool:
        """Check if player can declare Rummikub (go out).
        
        Args:
            player_id: Player to check
            
        Returns:
            True if player has no tiles in hand
        """
        return len(self.player_hands[player_id]) == 0
    
    def declare_out(self, player_id: int) -> bool:
        """Declare Rummikub (going out).
        
        Args:
            player_id: Player declaring out
            
        Returns:
            True if successfully went out
        """
        if not self.can_go_out(player_id):
            return False
        
        self.game_over = True
        self.winner = player_id
        return True
    
    def next_player(self) -> None:
        """Advance to next player."""
        self.current_player = (self.current_player + 1) % self.num_players
        self.turn_count += 1
        
        # Check for max turns (draw)
        if self.turn_count >= self.max_turns:
            self._end_game_by_draw()
    
    def _end_game_by_draw(self) -> None:
        """End game when max turns reached - lowest hand value wins."""
        self.game_over = True
        
        # Find player with lowest hand value
        min_value = float('inf')
        winner = None
        
        for pid in range(self.num_players):
            value = self.calculate_hand_value(pid)
            if value < min_value:
                min_value = value
                winner = pid
        
        self.winner = winner
    
    def get_valid_actions_for_player(self, player_id: int) -> List[Dict]:
        """Get list of valid actions for a player.
        
        Args:
            player_id: Player to get actions for
            
        Returns:
            List of valid action dictionaries
        """
        actions = []
        hand = self.player_hands[player_id]
        
        # Always can draw (if tiles remain)
        if not self.tile_pool.is_empty():
            actions.append({'type': 'DRAW'})
        
        # Check for valid new melds
        from itertools import combinations
        for size in range(3, min(len(hand) + 1, 14)):
            for indices in combinations(range(len(hand)), size):
                tiles = [hand[i] for i in indices]
                if Meld.is_valid(tiles):
                    # Check initial meld requirement
                    if self.has_initial_meld[player_id] or Meld.calculate_value(tiles) >= 30:
                        actions.append({
                            'type': 'PLAY_NEW_MELD',
                            'tile_indices': list(indices)
                        })
        
        # Check for valid additions to table melds
        if self.has_initial_meld[player_id]:
            for tile_idx, tile in enumerate(hand):
                for meld_idx, meld in enumerate(self.table_melds):
                    if Meld.can_add_tile(meld.tiles, tile):
                        actions.append({
                            'type': 'ADD_TO_MELD',
                            'tile_idx': tile_idx,
                            'meld_idx': meld_idx
                        })
        
        # Check if can go out
        if self.can_go_out(player_id):
            actions.append({'type': 'DECLARE_OUT'})
        
        return actions
    
    def encode(self) -> Dict:
        """Encode game state to dictionary for serialization."""
        return {
            'num_players': self.num_players,
            'current_player': self.current_player,
            'table_melds': [
                [tile.encode() for tile in meld.tiles]
                for meld in self.table_melds
            ],
            'player_hands': {
                str(pid): [tile.encode() for tile in hand]
                for pid, hand in self.player_hands.items()
            },
            'pool_size': self.tile_pool.remaining(),
            'has_initial_meld': self.has_initial_meld,
            'game_over': self.game_over,
            'winner': self.winner,
            'turn_count': self.turn_count
        }
    
    @classmethod
    def decode(cls, data: Dict) -> 'GameState':
        """Decode game state from dictionary."""
        state = cls(data['num_players'])
        state.current_player = data['current_player']
        state.table_melds = [
            Meld([Tile.decode(code) for code in meld])
            for meld in data['table_melds']
        ]
        state.player_hands = {
            int(pid): [Tile.decode(code) for code in hand]
            for pid, hand in data['player_hands'].items()
        }
        state.tile_pool.tiles = []  # Pool is reconstructed from remaining count
        state.has_initial_meld = {int(k): v for k, v in data['has_initial_meld'].items()}
        state.game_over = data['game_over']
        state.winner = data['winner']
        state.turn_count = data['turn_count']
        return state
