"""Quick test to verify jokers are removed from the game."""

from game_state import GameState
from tile import TileSet

# Test 1: Check TileSet without jokers
print("Test 1: TileSet without jokers")
tile_set = TileSet(include_jokers=False)
print(f"  Total tiles: {tile_set.total_tiles()}")
print(f"  Remaining: {tile_set.remaining()}")
joker_count = sum(1 for t in tile_set.tiles if t.is_joker)
print(f"  Jokers in set: {joker_count}")
assert joker_count == 0, "Should have no jokers!"
print("  ✓ PASSED - No jokers in set\n")

# Test 2: Check TileSet with jokers (for comparison)
print("Test 2: TileSet with jokers (optional)")
tile_set_with_jokers = TileSet(include_jokers=True)
print(f"  Total tiles: {tile_set_with_jokers.total_tiles()}")
joker_count_with = sum(1 for t in tile_set_with_jokers.tiles if t.is_joker)
print(f"  Jokers in set: {joker_count_with}")
assert joker_count_with == 2, "Should have 2 jokers when enabled!"
print("  ✓ PASSED - Has 2 jokers when enabled\n")

# Test 3: Check GameState doesn't have jokers
print("Test 3: GameState default (no jokers)")
game = GameState(num_players=2)
print(f"  Tile pool size: {game.tile_pool.remaining()}")
joker_count_game = sum(1 for t in game.tile_pool.tiles if t.is_joker)
print(f"  Jokers in pool: {joker_count_game}")
assert joker_count_game == 0, "GameState should have no jokers!"
print("  ✓ PASSED - No jokers in game\n")

# Test 4: Deal hands and verify no jokers
print("Test 4: Deal initial hands")
game.reset()
for player_id in range(2):
    hand = game.player_hands[player_id]
    jokers_in_hand = sum(1 for t in hand if t.is_joker)
    print(f"  Player {player_id + 1} hand size: {len(hand)}, jokers: {jokers_in_hand}")
    assert jokers_in_hand == 0, f"Player {player_id} should have no jokers!"
print("  ✓ PASSED - No jokers dealt to players\n")

print("="*60)
print("All tests passed! Jokers have been successfully removed.")
print("="*60)
