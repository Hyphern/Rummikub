# Rummikub Machine Learning System

A machine learning framework for training agents to play Rummikub against each other.

## Architecture

### Core Files

- **`tile.py`** - Tile representation and management
- **`meld.py`** - Meld validation (runs and groups)
- **`game_state.py`** - Game state management
- **`ml_environment.py`** - ML environment wrapper (gym-like interface)
- **`agent.py`** - Agent base class and example implementations
- **`tournament.py`** - Tournament system for training and evaluation

### Worm Integration

The `worm/worm.py` solver logic is integrated through the `WormAgent` class, which attempts to use constraint propagation for better decision-making.

## Usage

### 1. Quick Start - Run a Tournament

```bash
python tournament.py
```

This runs a quick tournament with:
- RandomAgent
- HeuristicAgent
- GreedyAgent
- ConservativeAgent
- QLearningAgent

### 2. Train Agents

```python
from tournament import train_agents
from agent import QLearningAgent, HeuristicAgent

# Create agents
agents = [
    QLearningAgent("QLearning", learning_rate=0.01),
    HeuristicAgent("Heuristic"),
    RandomAgent("Random"),
]

# Train for 1000 episodes
train_agents(agents, num_episodes=1000, eval_interval=100)
```

### 3. Custom Tournament

```python
from tournament import Tournament
from agent import RandomAgent, GreedyAgent

# Create tournament
agents = [RandomAgent(), GreedyAgent()]
tournament = Tournament(agents, num_players=2)

# Run round-robin
tournament.run_round_robin(games_per_matchup=20)
tournament.print_results()

# Or run random matchups
tournament.run_random_matchups(num_games=100)
tournament.print_results()
```

### 4. Create Custom Agent

```python
from agent import RummikubAgent
import numpy as np

class MyAgent(RummikubAgent):
    def __init__(self):
        super().__init__("MyAgent")
    
    def select_action(self, observation, valid_actions):
        # Get valid action indices
        valid_indices = np.where(valid_actions == 1)[0]
        
        # Your logic here
        # 0 = DRAW
        # 1 = DECLARE_OUT
        # 2-31 = PLAY_NEW_MELD
        # 32+ = ADD_TO_MELD
        
        return np.random.choice(valid_indices)
```

## Action Space

The environment provides a discrete action space:

- **0**: DRAW - Draw a tile from the pool
- **1**: DECLARE_OUT - Declare Rummikub (win)
- **2 to 31**: PLAY_NEW_MELD - Play tiles as new meld (simplified encoding)
- **32+**: ADD_TO_MELD - Add tile to existing table meld

## Observation Space

The observation is a dictionary with:

```python
{
    'hand': np.ndarray[30],           # Encoded tiles in hand
    'hand_mask': np.ndarray[30],      # Which positions are valid
    'table': np.ndarray[20, 13],      # Encoded tiles on table
    'table_mask': np.ndarray[20, 13], # Which positions are valid
    'pool_size': np.ndarray[1],       # Tiles remaining in pool
    'has_initial_meld': np.ndarray[1], # Initial meld completed?
    'opponent_hands': np.ndarray[n-1], # Sizes of opponent hands
    'valid_actions_mask': np.ndarray[652], # Valid action mask
    'current_player': np.ndarray[1],  # Current player ID
    'turn_count': np.ndarray[1],      # Turn counter
}
```

## Reward Structure

- **+100.0**: Winning the game
- **+1.0**: Successfully playing a new meld (+ bonus for higher value)
- **+0.5**: Successfully adding to an existing meld
- **-0.1**: Drawing a tile (encourages playing)
- **-1.0**: Failed action
- **-hand_value**: Penalty for losing

## Agent Types

### RandomAgent
Selects completely random valid actions. Good baseline.

### HeuristicAgent
Uses simple heuristics:
1. Win if possible
2. Play new melds
3. Add to existing melds
4. Draw as last resort

### GreedyAgent
Aggressive play style, prioritizes playing tiles over drawing.

### ConservativeAgent
Defensive play, prefers drawing to accumulate options.

### WormAgent
Uses constraint propagation from `worm.py` to find definite moves.

### QLearningAgent
Simple Q-learning with tabular Q-values. Demonstrates RL approach.

## Training Your Own Agent

### Option 1: Extend QLearningAgent

Modify the Q-learning agent with better state representation:

```python
class BetterQLearningAgent(QLearningAgent):
    def _state_to_key(self, observation):
        # Better state representation
        hand = observation['hand']
        # Encode hand features instead of raw tiles
        return str((
            tuple(sorted(hand[:10])),
            int(observation['has_initial_meld'][0]),
            len(observation['opponent_hands']),
        ))
```

### Option 2: Neural Network Agent

Create an agent with a neural network policy:

```python
import torch
import torch.nn as nn

class NeuralAgent(RummikubAgent):
    def __init__(self):
        super().__init__("Neural")
        self.network = self._build_network()
    
    def _build_network(self):
        # Input: observation features
        # Output: action probabilities
        return nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 652),  # Action space size
        )
    
    def select_action(self, observation, valid_actions):
        # Convert observation to tensor
        features = self._extract_features(observation)
        logits = self.network(features)
        
        # Mask invalid actions
        logits[valid_actions == 0] = -float('inf')
        
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action
```

### Option 3: Use Stable-Baselines3

Integrate with standard RL libraries:

```python
from stable_baselines3 import PPO
from ml_environment import RummikubMLEnv

# Wrap environment for SB3
class GymWrapper:
    def __init__(self):
        self.env = RummikubMLEnv()
    
    def reset(self):
        obs = self.env.reset()
        return self._flatten_obs(obs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten_obs(obs), reward, done, info

# Train with PPO
env = GymWrapper()
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

## Evaluation

Run a comprehensive tournament:

```python
from tournament import Tournament

agents = [...]  # Your agents
tournament = Tournament(agents)

# Round-robin (each plays each)
results = tournament.run_round_robin(games_per_matchup=50)

# Or random matchups
results = tournament.run_random_matchups(num_games=500)

# Print detailed results
tournament.print_results()

# Save to file
tournament.save_results('my_tournament.json')
```

## Integration with Worm Solver

The `WormAgent` attempts to use the constraint propagation logic from `worm.py`. To fully leverage it:

1. Convert current game state to board matrix format
2. Use `find_explicit_moves()` to get definite moves
3. Map those moves back to action space

Example:

```python
from worm.worm import find_explicit_moves

class AdvancedWormAgent(RummikubAgent):
    def select_action(self, observation, valid_actions):
        # Convert to worm format
        board = self.env.get_state_for_worm()
        
        # Find definite moves
        definite = find_explicit_moves(board)
        
        if definite:
            # Map worm move to action space
            action = self._worm_move_to_action(definite[0])
            return action
        
        # Fall back to heuristic
        return super().select_action(observation, valid_actions)
```

## Future Improvements

1. **Better State Representation**: Include features like:
   - Count of tiles by color
   - Count of tiles by number
   - Available runs/groups
   - Opponent hand size trends

2. **Multi-Agent Training**: Train agents against evolving opponents

3. **Hierarchical Actions**: Separate policies for:
   - Whether to draw or play
   - Which meld to play
   - Whether to add to existing melds

4. **Self-Play**: AlphaZero-style training through self-play

5. **Transfer Learning**: Pre-train on worm solver data

## Files Overview

```
rummikub/
├── tile.py              # Tile and TileSet classes
├── meld.py              # Meld validation
├── game_state.py        # Game state management
├── ml_environment.py    # ML environment wrapper
├── agent.py             # Agent base and implementations
├── tournament.py        # Training and evaluation
└── worm/
    └── worm.py          # Constraint propagation solver
```

Run `python tournament.py` to see it in action!
