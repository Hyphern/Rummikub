"""Super simple test - just check if env.step() works at all."""

import sys
sys.path.insert(0, '.')

print("Importing modules...")
from ml_environment import RummikubMLEnv
import numpy as np

print("Creating environment...")
env = RummikubMLEnv(num_players=2)

print("Resetting environment...")
obs = env.reset(seed=42)

print(f"Initial observation shapes:")
for key, value in obs.items():
    print(f"  {key}: shape={value.shape}, sum={np.sum(value)}")

print(f"\nPool size: {obs['pool_size'][0]}")
print(f"Hand size: {np.sum(obs['hand_mask'])}")

# Count valid actions
valid_count = np.sum(obs['valid_actions_mask'])
print(f"Valid actions: {valid_count}")

if valid_count == 0:
    print("ERROR: No valid actions available!")
    sys.exit(1)

print("\nTrying first step...")
action = 0  # Try DRAW
print(f"Action: {action}")

obs, reward, done, info = env.step(action)
print(f"Step result:")
print(f"  Action taken: {info.get('action_taken', 'UNKNOWN')}")
print(f"  Success: {info.get('action_success', False)}")
print(f"  Reward: {reward}")
print(f"  Done: {done}")
print(f"  New pool size: {obs['pool_size'][0]}")
print(f"  New hand size: {np.sum(obs['hand_mask'])}")

print("\nTest passed! Environment is working.")
