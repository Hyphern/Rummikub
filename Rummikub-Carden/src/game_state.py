"""Game state management for Rummikub."""

from typing import List, Dict, Optional, Set, Tuple
import copy
from tile import Tile, TileSet, sort_tiles_for_hand
from meld import Meld
from config import (
    INCLUDE_JOKERS,
    LOG_GAME_OUTCOMES,
    LOG_FILE_PATH,
    INITIAL_MELD_MIN_VALUE,
    INITIAL_HAND_SIZE,
    MAX_TURNS,
)


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
        self.tile_pool = TileSet(include_jokers=INCLUDE_JOKERS)
        self.table_melds: List[Meld] = []
        self.player_hands: Dict[int, List[Tile]] = {i: [] for i in range(num_players)}
        self.current_player = 0
        self.has_initial_meld: Dict[int, bool] = {i: False for i in range(num_players)}
        self.game_over = False
        self.winner: Optional[int] = None
        self.turn_count = 0
        self.melds_played_this_turn = 0  # Track melds played in current turn
        self.initial_meld_value_this_turn = (
            0  # Accumulate initial meld value across multi-meld turn
        )
        self.table_melds_at_turn_start = (
            0  # Index into table_melds marking start of this turn's melds
        )
        self.max_turns = MAX_TURNS  # Prevent infinite games

        # Deal initial hands
        self._deal_initial_hands()

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
        self.melds_played_this_turn = 0
        self.initial_meld_value_this_turn = 0
        self.table_melds_at_turn_start = 0

        # Deal initial hands
        self._deal_initial_hands()

    def _deal_initial_hands(self) -> None:
        """Deal initial tiles to each player."""
        for player_id in range(self.num_players):
            self.player_hands[player_id] = self.tile_pool.draw(INITIAL_HAND_SIZE)
            # Sort the hand for better display
            self.player_hands[player_id] = sort_tiles_for_hand(
                self.player_hands[player_id]
            )

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

    def clone(self) -> "GameState":
        """Create a deep copy of the game state.

        Returns:
            Deep copy of this game state
        """
        new_state = GameState(self.num_players)
        new_state.tile_pool.tiles = [t.copy() for t in self.tile_pool.tiles]
        new_state.table_melds = [meld.copy() for meld in self.table_melds]
        new_state.player_hands = {
            pid: [t.copy() for t in hand] for pid, hand in self.player_hands.items()
        }
        new_state.current_player = self.current_player
        new_state.has_initial_meld = self.has_initial_meld.copy()
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        new_state.turn_count = self.turn_count
        new_state.max_turns = self.max_turns
        new_state.melds_played_this_turn = self.melds_played_this_turn
        new_state.initial_meld_value_this_turn = self.initial_meld_value_this_turn
        new_state.table_melds_at_turn_start = self.table_melds_at_turn_start
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
            # Re-sort the hand after drawing
            self.player_hands[player_id] = sort_tiles_for_hand(
                self.player_hands[player_id]
            )
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

        # Validate indices
        if not tile_indices:
            return False
        if any(not isinstance(i, int) or i < 0 or i >= len(hand) for i in tile_indices):
            return False
        if len(set(tile_indices)) != len(tile_indices):
            return False  # No duplicate indices

        tiles = [hand[i] for i in tile_indices]

        # Validate meld
        if not Meld.is_valid(tiles):
            return False

        # Check initial meld requirement (accumulate across melds this turn)
        if not self.has_initial_meld[player_id]:
            value = Meld.calculate_value(tiles) or 0
            self.initial_meld_value_this_turn += value
            if self.initial_meld_value_this_turn >= INITIAL_MELD_MIN_VALUE:
                self.has_initial_meld[player_id] = True

        # Remove tiles from hand
        # Remove in reverse order to maintain correct indices
        for idx in sorted(tile_indices, reverse=True):
            hand.pop(idx)

        # Add to table (tiles remain in the order they were played - unsorted)
        self.table_melds.append(Meld(tiles))
        self.melds_played_this_turn += 1
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

        if not (0 <= tile_idx < len(hand)):
            return False

        if not (0 <= meld_idx < len(self.table_melds)):
            return False

        tile = hand[tile_idx]
        meld = self.table_melds[meld_idx]

        if not Meld.can_add_tile(meld.tiles, tile):
            return False

        # Remove from hand and add to meld
        hand.pop(tile_idx)
        new_tiles = meld.tiles + [tile]
        self.table_melds[meld_idx] = Meld(new_tiles)
        self.melds_played_this_turn += 1
        return True

    def replace_joker(self, player_id: int, tile_idx: int, meld_idx: int) -> bool:
        """Replace a joker on table with a tile from hand.

        In Rummikub, you can replace a joker on the table with the tile it represents.
        The joker goes to your hand (must be played in a new meld that turn).

        Args:
            player_id: Player replacing the joker
            tile_idx: Index of the replacement tile in player's hand
            meld_idx: Index of the meld containing the joker on table

        Returns:
            True if replacement was successful
        """
        if not self.has_initial_meld[player_id]:
            return False  # Must have initial meld first

        hand = self.player_hands[player_id]

        if not (0 <= tile_idx < len(hand)):
            return False

        if not (0 <= meld_idx < len(self.table_melds)):
            return False

        tile = hand[tile_idx]
        meld = self.table_melds[meld_idx]

        # Check if meld has any jokers
        jokers = [t for t in meld.tiles if t.is_joker]
        if not jokers:
            return False  # No jokers to replace

        # Get possible replacements for jokers
        possible_replacements = meld.get_joker_replacements()

        # Check if the tile matches any possible replacement
        if tile not in possible_replacements:
            return False

        # Find which joker to replace (first one that matches)
        joker_idx = None
        for i, joker in enumerate(meld.tiles):
            if joker.is_joker:
                joker_idx = i
                break

        if joker_idx is None:
            return False

        # Perform the replacement:
        # 1. Remove the replacement tile from hand
        hand.pop(tile_idx)

        # 2. Replace joker with the tile in the meld
        new_tiles = list(meld.tiles)
        new_tiles[joker_idx] = tile  # Replace joker with actual tile
        self.table_melds[meld_idx] = Meld(new_tiles)

        # 3. Add joker to player's hand
        joker = Tile.create_joker()
        hand.append(joker)

        # Re-sort hand
        from tile import sort_tiles_for_hand

        self.player_hands[player_id] = sort_tiles_for_hand(hand)

        self.melds_played_this_turn += 1
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
        self._log_game_outcome("win")
        return True

    def next_player(self) -> None:
        """Advance to next player."""
        self.current_player = (self.current_player + 1) % self.num_players
        self.turn_count += 1
        self.melds_played_this_turn = 0  # Reset for new player's turn
        self.initial_meld_value_this_turn = 0  # Reset initial meld accumulator
        self.table_melds_at_turn_start = len(
            self.table_melds
        )  # Mark start of next turn's melds

        # Check for max turns (draw)
        if self.turn_count >= self.max_turns:
            self._end_game_by_draw()

    def end_turn(self) -> None:
        """End the current player's turn without playing any more melds.

        If the player hasn't met the initial meld threshold, any melds
        played this turn are rolled back to the player's hand.
        """
        self._rollback_if_initial_incomplete()
        self.next_player()

    def rollback_initial_melds(self) -> None:
        """Roll back melds played this turn if initial meld threshold not met.

        Public interface for ml_environment to call on DRAW when initial
        meld is incomplete. Returns tiles from this turn's melds to hand.
        """
        self._rollback_if_initial_incomplete()

    def _rollback_if_initial_incomplete(self) -> None:
        """Roll back melds played this turn if initial meld threshold not met.

        In Rummikub, if a player plays melds toward their initial meld but
        doesn't reach the required threshold (e.g., 30 points), all melds
        played that turn must be returned to the player's hand.
        """
        player_id = self.current_player
        if self.has_initial_meld[player_id]:
            return  # Threshold already met, nothing to roll back

        if self.melds_played_this_turn == 0:
            return  # Nothing was played this turn

        # Return tiles from this turn's melds back to the player's hand
        melds_to_rollback = self.table_melds[self.table_melds_at_turn_start :]
        for meld in melds_to_rollback:
            self.player_hands[player_id].extend(meld.tiles)

        # Remove the rolled-back melds from the table
        self.table_melds = self.table_melds[: self.table_melds_at_turn_start]

        # Reset turn tracking
        self.melds_played_this_turn = 0
        self.initial_meld_value_this_turn = 0

        # Re-sort hand after returning tiles
        self.player_hands[player_id] = sort_tiles_for_hand(self.player_hands[player_id])

    # ── Table Rearrangement ──────────────────────────────────────

    def can_rearrange_with_tile(self, player_id: int, tile_idx: int) -> bool:
        """Check if a hand tile can be played via table rearrangement.

        Combines all non-joker table tiles with the selected hand tile into a
        4×13 board matrix and checks if ``wormed.solve()`` can decompose it
        into valid runs and groups.

        Args:
            player_id: Player attempting the rearrangement.
            tile_idx:  Index of the tile in the player's hand.

        Returns:
            True if rearrangement is possible with this tile.
        """
        from worm_integration import table_to_board_matrix, combine_boards
        import wormed

        if not self.has_initial_meld[player_id]:
            return False  # Must have initial meld first

        hand = self.player_hands[player_id]
        if not (0 <= tile_idx < len(hand)):
            return False

        tile = hand[tile_idx]
        if tile.is_joker:
            # Jokers can always "rearrange in" trivially — they add flexibility.
            # But we still need at least 1 table meld to rearrange.
            if not self.table_melds:
                return False
            # Build board from table non-joker tiles (jokers excluded from matrix)
            table_board = table_to_board_matrix(self.table_melds)
            # A joker doesn't change the non-joker matrix, so the current
            # table must already be solvable (it should be by invariant).
            # Playing a joker via rearrange is only useful if combined with
            # other hand tiles — for a single joker, it's not meaningful.
            return False  # Single joker rearrangement not supported

        # Build board matrices
        table_board = table_to_board_matrix(self.table_melds)

        # Create a single-tile hand matrix for the tile we want to play
        tile_board = [[0] * 13 for _ in range(4)]
        color_to_row = {"red": 0, "blue": 1, "black": 2, "orange": 3}
        if tile.color and tile.number:
            row = color_to_row.get(tile.color, 0)
            col = tile.number - 1
            tile_board[row][col] = 1

        # Combine: table + this one hand tile
        combined = combine_boards(table_board, tile_board)

        # Clear the wormed caches before solving (boards change each call)
        wormed._solve_cache.clear()
        wormed.get_all_runs.cache_clear()
        wormed.get_all_groups.cache_clear()
        wormed.get_tile_moves.cache_clear()

        solution = wormed.solve(combined)
        return solution is not None

    def rearrange_with_tiles(self, player_id: int, tile_indices: List[int]) -> bool:
        """Play hand tiles onto the table by rearranging existing table melds.

        All non-joker tiles on the table are combined with the selected hand
        tiles into a 4×13 board matrix.  ``wormed.solve()`` finds a valid
        decomposition of the combined set.  If successful, the table melds are
        reconstructed from the solution and the played tiles are removed from
        the player's hand.  Any joker tiles that were on the table are freed
        and added to the player's hand.

        Args:
            player_id:    Player performing the rearrangement.
            tile_indices: Indices of tiles in the player's hand to play.

        Returns:
            True if rearrangement succeeded.
        """
        from worm_integration import table_to_board_matrix, combine_boards
        import wormed

        if not self.has_initial_meld[player_id]:
            return False

        hand = self.player_hands[player_id]

        # Validate indices
        if not tile_indices:
            return False
        if any(not isinstance(i, int) or i < 0 or i >= len(hand) for i in tile_indices):
            return False
        if len(set(tile_indices)) != len(tile_indices):
            return False

        selected_tiles = [hand[i] for i in tile_indices]

        # Separate joker and non-joker tiles from the selected hand tiles
        hand_jokers = [t for t in selected_tiles if t.is_joker]
        hand_regular = [t for t in selected_tiles if not t.is_joker]

        # Collect table jokers (they'll be freed during rearrangement)
        table_jokers: List[Tile] = []
        for meld in self.table_melds:
            for tile in meld.tiles:
                if tile.is_joker:
                    table_jokers.append(tile)

        # Build combined board: table non-jokers + hand non-jokers
        table_board = table_to_board_matrix(self.table_melds)
        hand_board = [[0] * 13 for _ in range(4)]
        color_to_row = {"red": 0, "blue": 1, "black": 2, "orange": 3}
        for tile in hand_regular:
            if tile.color and tile.number:
                row = color_to_row.get(tile.color, 0)
                col = tile.number - 1
                hand_board[row][col] += 1

        combined = combine_boards(table_board, hand_board)

        # Clear wormed caches
        wormed._solve_cache.clear()
        wormed.get_all_runs.cache_clear()
        wormed.get_all_groups.cache_clear()
        wormed.get_tile_moves.cache_clear()

        solution = wormed.solve(combined)
        if solution is None:
            return False

        # Reconstruct table melds from the solution
        row_to_color = {0: "red", 1: "blue", 2: "black", 3: "orange"}
        new_melds: List[Meld] = []
        for move in solution:
            tiles: List[Tile] = []
            if move[0] == "run":
                _, row, start, end = move
                color = row_to_color[row]
                for c in range(start, end + 1):
                    tiles.append(Tile(color=color, number=c + 1))
            elif move[0] == "group":
                _, col, colors = move
                number = col + 1
                for row in colors:
                    color = row_to_color[row]
                    tiles.append(Tile(color=color, number=number))

            if tiles:
                try:
                    new_melds.append(Meld(tiles))
                except ValueError:
                    # Solution produced an invalid meld — shouldn't happen
                    return False

        # Success — commit the changes
        self.table_melds = new_melds

        # Remove played tiles from hand (reverse order to keep indices valid)
        for idx in sorted(tile_indices, reverse=True):
            hand.pop(idx)

        # Freed table jokers go to the player's hand
        for joker in table_jokers:
            hand.append(joker)

        # Hand jokers played during rearrangement: they were already removed
        # from hand above (part of tile_indices).  We don't add them to any
        # specific meld — they're simply consumed.  In a future enhancement,
        # jokers could be placed into melds that accept them.

        # Re-sort hand
        self.player_hands[player_id] = sort_tiles_for_hand(hand)
        self.melds_played_this_turn += 1
        return True

    def _end_game_by_draw(self) -> None:
        """End game when max turns reached - no winner (draw)."""
        self.game_over = True
        self.winner = None  # No winner — nobody went out
        self._log_game_outcome("draw")

    def _log_game_outcome(self, end_type: str) -> None:
        """Log game outcome if logging is enabled.

        Args:
            end_type: 'win' or 'draw'
        """
        import datetime

        # Calculate hand values for all players
        hand_values = {
            pid: self.calculate_hand_value(pid) for pid in range(self.num_players)
        }

        # Text log
        if LOG_GAME_OUTCOMES:
            log_entry = (
                f"{datetime.datetime.now().isoformat()} | "
                f"Players: {self.num_players} | "
                f"Jokers: {INCLUDE_JOKERS} | "
                f"Turns: {self.turn_count} | "
                f"End: {end_type} | "
                f"Winner: P{self.winner} | "
                f"Hand values: {hand_values}\n"
            )
            try:
                with open(LOG_FILE_PATH, "a") as f:
                    f.write(log_entry)
            except IOError:
                pass  # Silently fail if logging fails

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
            actions.append({"type": "DRAW"})

        # Can end turn if at least one meld was played this turn
        # AND the initial meld threshold has been met
        if self.melds_played_this_turn > 0 and self.has_initial_meld[player_id]:
            actions.append({"type": "END_TURN"})

        # Check for valid new melds
        # With multi-meld turns, any valid meld can be played toward the
        # initial meld threshold (values accumulate across melds this turn)
        from itertools import combinations

        for size in range(3, min(len(hand) + 1, 14)):
            for indices in combinations(range(len(hand)), size):
                tiles = [hand[i] for i in indices]
                if Meld.is_valid(tiles):
                    actions.append(
                        {"type": "PLAY_NEW_MELD", "tile_indices": list(indices)}
                    )

        # Check for valid additions to table melds
        if self.has_initial_meld[player_id]:
            for tile_idx, tile in enumerate(hand):
                for meld_idx, meld in enumerate(self.table_melds):
                    if Meld.can_add_tile(meld.tiles, tile):
                        actions.append(
                            {
                                "type": "ADD_TO_MELD",
                                "tile_idx": tile_idx,
                                "meld_idx": meld_idx,
                            }
                        )

        # Check if can go out
        if self.can_go_out(player_id):
            actions.append({"type": "DECLARE_OUT"})

        return actions
