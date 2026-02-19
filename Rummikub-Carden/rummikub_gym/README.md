# Rummikub Gym Environment

A fully functional [OpenAI Gymnasium](https://gymnasium.farama.org/) environment for the tile-based game Rummikub, suitable for reinforcement learning research and training agents.

## Overview

Rummikub is a popular tile-based game where players aim to be the first to play all their tiles by forming valid combinations (melds) on the table. This environment implements the complete game rules and provides a standard Gymnasium API for RL training.

### Game Rules Summary

- **Tiles**: 106 tiles total - 2 sets of numbers 1-13 in 4 colors (red, blue, black, orange) + 2 jokers
- **Initial Deal**: Each player receives 14 tiles
- **Valid Melds**:
  - **Runs**: 3+ consecutive numbers of the same color (e.g., red 3-4-5)
  - **Groups**: 3-4 tiles of the same number, different colors (e.g., red 7, blue 7, black 7)
  - **Jokers**: Can substitute for any tile in a meld
- **Initial Meld**: First meld played must be worth 30+ points
- **Winning**: First player to play all tiles declares "Rummikub!" and wins

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rummikub-gym.git
cd rummikub-gym

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Basic Usage

```python
import gymnasium as gym
from rummikub_gym.envs.rummikub_env import RummikubEnv

# Create environment
env = RummikubEnv(num_players=2)

# Reset environment
obs, info = env.reset(seed=42)

# Take actions
action = 0  # DRAW
obs, reward, terminated, truncated, info = env.step(action)

# Render game state
env.render()
```

### Random Agent Example

```bash
# Run interactive random agent
python examples/random_agent.py

# Run silent test (multiple games)
python examples/random_agent.py --silent 10
```

### Human Play

```bash
# Play against a random AI
python examples/human_play.py
```

## Environment API

### Observation Space

The observation is a `Dict` space containing:

```python
{
    'hand': Box(0, 55, (14,)),              # Tiles in current player's hand (encoded)
    'hand_mask': Box(0, 1, (14,)),          # Which hand slots are valid
    'table_melds': Box(0, 55, (20, 13)),    # Tiles on table (max 20 melds, 13 tiles each)
    'table_meld_mask': Box(0, 1, (20, 13)), # Which table slots are valid
    'pool_size': Discrete(107),             # Tiles remaining in pool
    'opponent_hand_sizes': Box(0, 106, (num_players-1,)),  # Opponent hand sizes
    'has_initial_meld': MultiBinary(num_players),          # Who has made initial meld
    'current_player': Discrete(num_players),               # Whose turn it is
}
```

### Action Space

Actions are encoded as integers:
- `0`: DRAW - Draw a tile from the pool
- `1`: DECLARE_OUT - Declare Rummikub (win)
- `2+`: PLAY_NEW_MELD or ADD_TO_MELD (complex encoding)

Use `env.get_valid_action_mask()` to get valid actions as a binary mask.

### Reward Structure

- `+100`: Winning (going out first)
- `-hand_value`: Losing (negative of hand tile values)
- `-0.1`: Per turn (encourages faster play)
- `+0.5`: Successful action

## Core Components

### Tile

```python
from rummikub_gym.envs.tile import Tile

# Create regular tile
tile = Tile(color='red', number=5)

# Create joker
joker = Tile.create_joker()

# Encode/decode
code = tile.encode()  # 0-55
tile = Tile.decode(code)
```

### Meld

```python
from rummikub_gym.envs.meld import Meld

# Check if tiles form a valid meld
tiles = [Tile('red', 3), Tile('red', 4), Tile('red', 5)]
is_valid = Meld.is_valid(tiles)

# Calculate meld value
value = Meld.calculate_value(tiles)

# Check if tile can be added to meld
can_add = Meld.can_add_tile(tiles, new_tile)
```

### GameState

```python
from rummikub_gym.envs.game_state import GameState

# Create game state
state = GameState(num_players=2)
state.reset()

# Play a meld
state.play_meld(player_id=0, tile_indices=[0, 1, 2])

# Add to existing meld
state.add_to_meld(player_id=0, tile_idx=0, meld_idx=0)

# Draw tile
state.draw_tile(player_id=0)

# Check if can go out
can_win = state.can_go_out(player_id=0)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=rummikub_gym tests/

# Run specific test file
pytest tests/test_tile.py
pytest tests/test_meld.py
pytest tests/test_game_state.py
pytest tests/test_env.py
```

## Project Structure

```
rummikub_gym/
├── rummikub_gym/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── rummikub_env.py      # Main Gymnasium environment
│   │   ├── game_state.py        # Game state management
│   │   ├── tile.py              # Tile class and utilities
│   │   ├── meld.py              # Meld validation logic
│   │   └── action_space_utils.py # Action encoding/decoding
│   └── utils/
│       ├── __init__.py
│       └── validators.py        # Game rule validators
├── tests/
│   ├── test_tile.py
│   ├── test_meld.py
│   ├── test_game_state.py
│   └── test_env.py
├── examples/
│   ├── random_agent.py          # Random agent example
│   └── human_play.py            # Interactive human play
├── setup.py
├── requirements.txt
└── README.md
```

## Training RL Agents

### With Stable-Baselines3

```python
from stable_baselines3 import PPO
from rummikub_gym.envs.rummikub_env import RummikubEnv

# Create environment
env = RummikubEnv(num_players=2)

# Create and train model
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Test trained model
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

### Action Masking

For better training efficiency, use action masking to prevent invalid actions:

```python
from rummikub_gym.envs.rummikub_env import RummikubEnv

env = RummikubEnv(num_players=2)
obs, info = env.reset()

# Get valid action mask
valid_mask = env.get_valid_action_mask()

# Only sample from valid actions
valid_actions = np.where(valid_mask == 1)[0]
action = np.random.choice(valid_actions)
```

## Advanced Features

### Custom Reward Functions

Override the reward calculation:

```python
class CustomRummikubEnv(RummikubEnv):
    def _calculate_reward(self, player_id, action_success):
        # Custom reward logic
        reward = super()._calculate_reward(player_id, action_success)
        
        # Add bonus for playing high-value melds
        # Add penalty for holding jokers too long
        # etc.
        
        return reward
```

### Multi-Agent Training

The environment supports 2-4 players:

```python
# 3-player game
env = RummikubEnv(num_players=3)

# 4-player game
env = RummikubEnv(num_players=4)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Rummikub is a registered trademark of Pressman Toy Corporation
- This implementation is for educational and research purposes
- Built with [Gymnasium](https://gymnasium.farama.org/) for RL research

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{rummikub_gym,
  title = {Rummikub Gym: A Gymnasium Environment for Rummikub},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/rummikub-gym}
}
```
