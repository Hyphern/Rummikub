"""Example random agent for Rummikub."""

import random
import numpy as np
from rummikub_gym.envs.rummikub_env import RummikubEnv


def random_agent():
    """Run a random agent that selects valid actions randomly."""
    # Create environment
    env = RummikubEnv(num_players=2, render_mode='human')
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print("Starting Rummikub game with random agent...")
    print(f"Initial valid actions: {info['valid_actions'][:5]}...")  # Show first 5
    
    done = False
    step_count = 0
    max_steps = 100
    
    while not done and step_count < max_steps:
        # Get valid actions
        valid_actions = info['valid_actions']
        
        if not valid_actions:
            print("No valid actions available!")
            break
        
        # Select random valid action
        action = random.choice(valid_actions)
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render state
        env.render()
        
        print(f"Step {step_count}: Action={action}, Reward={reward:.2f}")
        
        done = terminated or truncated
        step_count += 1
    
    if terminated:
        print(f"\nGame over! Winner: Player {info.get('current_player', 'Unknown')}")
    elif truncated:
        print("\nGame truncated (max steps reached)")
    
    env.close()


def random_agent_silent(num_games=10):
    """Run multiple games without rendering to test stability."""
    env = RummikubEnv(num_players=2)
    
    wins = {0: 0, 1: 0}
    total_steps = 0
    
    for game in range(num_games):
        obs, info = env.reset(seed=game)
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            valid_actions = info['valid_actions']
            if not valid_actions:
                break
            
            action = random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            step_count += 1
        
        if terminated and 'current_player' in info:
            winner = info['current_player']
            wins[winner] += 1
        
        total_steps += step_count
        print(f"Game {game + 1}: {step_count} steps")
    
    print(f"\nResults after {num_games} games:")
    print(f"Player 0 wins: {wins[0]}")
    print(f"Player 1 wins: {wins[1]}")
    print(f"Average steps per game: {total_steps / num_games:.1f}")
    
    env.close()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--silent':
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        random_agent_silent(num_games)
    else:
        random_agent()
