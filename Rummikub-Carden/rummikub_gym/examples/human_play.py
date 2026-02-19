"""Interactive human play interface for Rummikub."""

from typing import List, Optional
from rummikub_gym.envs.rummikub_env import RummikubEnv
from rummikub_gym.envs.action_space_utils import ActionSpace


def display_hand(hand, show_indices=True):
    """Display tiles in hand with optional indices."""
    if not hand:
        print("  (empty)")
        return
    
    # Group by color
    by_color = {}
    for i, tile in enumerate(hand):
        color = 'joker' if tile.is_joker else tile.color
        if color not in by_color:
            by_color[color] = []
        by_color[color].append((i, tile))
    
    for color in sorted(by_color.keys()):
        tiles = by_color[color]
        tiles.sort(key=lambda x: x[1].number if not x[1].is_joker else 0)
        
        if show_indices:
            tile_strs = [f"[{idx}]{str(tile)}" for idx, tile in tiles]
        else:
            tile_strs = [str(tile) for _, tile in tiles]
        
        print(f"  {color.capitalize()}: {', '.join(tile_strs)}")


def display_table_melds(melds):
    """Display melds on the table."""
    if not melds:
        print("  (empty)")
        return
    
    for i, meld in enumerate(melds):
        tiles_str = ', '.join(str(t) for t in meld.tiles)
        print(f"  [{i}] {tiles_str}")


def get_user_action(valid_actions, action_space):
    """Get action from user input."""
    print("\nValid actions:")
    print("  0: DRAW - Draw a tile from the pool")
    
    action_list = [0]  # DRAW is always option 0
    
    # Group actions by type
    play_meld_actions = []
    add_to_meld_actions = []
    declare_out_action = None
    
    for action_id in valid_actions:
        if action_id == 0:
            continue  # Already listed
        
        action_dict = action_space.decode_action(action_id)
        action_type = action_dict['type']
        
        if action_type == 'PLAY_NEW_MELD':
            play_meld_actions.append((action_id, action_dict))
        elif action_type == 'ADD_TO_MELD':
            add_to_meld_actions.append((action_id, action_dict))
        elif action_type == 'DECLARE_OUT':
            declare_out_action = action_id
    
    # List PLAY_NEW_MELD actions
    if play_meld_actions:
        print("\n  Play new meld:")
        for i, (action_id, action_dict) in enumerate(play_meld_actions[:10], 1):
            indices = action_dict['tile_indices']
            print(f"    {i}: Play meld with tiles {indices}")
            action_list.append(action_id)
        
        if len(play_meld_actions) > 10:
            print(f"    ... and {len(play_meld_actions) - 10} more")
    
    # List ADD_TO_MELD actions
    if add_to_meld_actions:
        print("\n  Add to existing meld:")
        offset = len(action_list)
        for i, (action_id, action_dict) in enumerate(add_to_meld_actions[:10], offset):
            tile_idx = action_dict['tile_idx']
            meld_idx = action_dict['meld_idx']
            print(f"    {i}: Add tile {tile_idx} to meld {meld_idx}")
            action_list.append(action_id)
        
        if len(add_to_meld_actions) > 10:
            print(f"    ... and {len(add_to_meld_actions) - 10} more")
    
    # List DECLARE_OUT
    if declare_out_action is not None:
        action_list.append(declare_out_action)
        print(f"\n  {len(action_list) - 1}: DECLARE OUT (Rummikub!)")
    
    # Get user input
    while True:
        try:
            choice = input(f"\nEnter action number (0-{len(action_list) - 1}): ").strip()
            choice_idx = int(choice)
            
            if 0 <= choice_idx < len(action_list):
                return action_list[choice_idx]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(action_list) - 1}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def human_play():
    """Main function for human play."""
    print("=" * 60)
    print("RUMMIKUB - Human Play Interface")
    print("=" * 60)
    print("\nInstructions:")
    print("- You are Player 0")
    print("- The AI is Player 1 (random agent)")
    print("- Valid actions are shown with numbers")
    print("- Enter the number of your chosen action")
    print("- Goal: Be the first to play all your tiles!")
    print("- Initial meld must be worth 30+ points")
    print()
    
    # Create environment
    env = RummikubEnv(num_players=2)
    action_space = ActionSpace()
    
    # Reset
    obs, info = env.reset()
    
    done = False
    step_count = 0
    
    while not done and step_count < 200:
        current_player = info['current_player']
        
        print(f"\n{'=' * 60}")
        print(f"TURN {step_count} - Player {current_player}")
        print(f"{'=' * 60}")
        
        # Display game state
        print(f"\nTiles in pool: {obs['pool_size']}")
        
        print("\nTable melds:")
        # Decode table melds from observation
        table_melds = []
        for i in range(20):
            if obs['table_meld_mask'][i][0] == 1:
                meld_tiles = []
                for j in range(13):
                    if obs['table_meld_mask'][i][j] == 1:
                        from rummikub_gym.envs.tile import Tile
                        tile_code = obs['table_melds'][i][j]
                        meld_tiles.append(Tile.decode(tile_code))
                if meld_tiles:
                    table_melds.append(meld_tiles)
        
        for i, meld in enumerate(table_melds):
            tiles_str = ', '.join(str(t) for t in meld)
            print(f"  [{i}] {tiles_str}")
        
        # Display all players' hands (only show counts for opponents)
        for pid in range(2):
            if pid == current_player:
                print(f"\nYOUR HAND (Player {pid}) - {obs['hand_mask'].sum()} tiles:")
                from rummikub_gym.envs.tile import Tile
                hand = []
                for i in range(14):
                    if obs['hand_mask'][i] == 1:
                        hand.append(Tile.decode(obs['hand'][i]))
                display_hand(hand, show_indices=(pid == 0))
            else:
                opponent_idx = 0 if pid == 1 else 0
                hand_size = obs['opponent_hand_sizes'][opponent_idx]
                print(f"\nPlayer {pid} hand: {hand_size} tiles (hidden)")
        
        # Get valid actions
        valid_actions = info['valid_actions']
        
        if current_player == 0:
            # Human turn
            action = get_user_action(valid_actions, action_space)
        else:
            # AI turn (random)
            import random
            action = random.choice(valid_actions)
            action_dict = action_space.decode_action(action)
            print(f"\nAI chose: {action_dict}")
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            winner = info['current_player']
            print(f"\n{'=' * 60}")
            print(f"GAME OVER! Player {winner} wins!")
            print(f"{'=' * 60}")
        
        done = terminated or truncated
        step_count += 1
    
    if not terminated:
        print("\nGame ended (max steps reached)")
    
    env.close()


if __name__ == '__main__':
    human_play()
