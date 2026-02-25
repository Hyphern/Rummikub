"""Debug script to test a single game and see where it hangs."""

from ml_environment import RummikubMLEnv
from agent import RandomAgent, HeuristicAgent
import numpy as np

print("="*60)
print("DEBUG: Running single game with verbose output")
print("="*60)

# Create simple agents
agents = [
    RandomAgent("Random1"),
    RandomAgent("Random2"),
]

# Create environment
env = RummikubMLEnv(num_players=2)
obs = env.reset(seed=42)

print(f"\nInitial state:")
print(f"  Pool size: {obs['pool_size'][0]}")
print(f"  Hand size: {np.sum(obs['hand_mask'])}")
print(f"  Valid actions: {np.sum(obs['valid_actions_mask'])}")

done = False
step = 0
max_steps = 200  # Limit for debugging

while not done and step < max_steps:
    current_player = obs['current_player'][0]
    agent = agents[current_player]
    valid_actions = obs['valid_actions_mask']
    
    # Get valid action indices
    valid_indices = np.where(valid_actions == 1)[0]
    
    print(f"\n[Step {step}] Player {current_player} ({agent.name})")
    print(f"  Hand size: {np.sum(obs['hand_mask'])}")
    print(f"  Valid actions: {len(valid_indices)}")
    
    if len(valid_indices) == 0:
        print("  ERROR: No valid actions! Breaking.")
        break
    
    # Select action
    action = agent.select_action(obs, valid_actions)
    print(f"  Selected action: {action}")
    
    # Execute
    obs, reward, done, info = env.step(action)
    print(f"  Result: {info.get('action_taken', 'UNKNOWN')}, reward={reward:.1f}, done={done}")
    
    step += 1
    
    if step >= max_steps:
        print(f"\n[DEBUG] Reached max steps ({max_steps})")
        break

print(f"\n{'='*60}")
print(f"Game ended after {step} steps")
print(f"Done: {done}")
if done:
    print(f"Winner: Player {info.get('winner', 'None')}")
