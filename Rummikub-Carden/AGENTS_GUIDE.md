# New Rummikub Agents with Worm Integration

This document describes the new agents that integrate worm.py logic for more sophisticated decision-making.

## Overview

The new agents use the `worm_integration.py` module to:
1. Convert hand tiles to 4x13 board matrix format
2. Find valid runs and groups using worm.py logic
3. Score moves based on agent-specific strategies
4. Make intelligent decisions about when to play vs when to draw

## New Agent Types

### 1. HoardingAgent
**Strategy**: Save tiles and try to go out in one big play

**Behavior**:
- Accumulates tiles until reaching a threshold (default 10 tiles)
- Only plays when it can make a significant move (5+ tiles)
- Tries to find moves that use most/all tiles at once
- Falls back to drawing if it can't make a big play

**Parameters**:
- `tiles_to_hoard`: Number of tiles to accumulate before playing (default: 10)

**When to use**: When you want an agent that plans ahead and makes explosive plays

**Example**:
```python
HoardingAgent("Hoarding10", tiles_to_hoard=10)  # Conservative
HoardingAgent("Hoarding15", tiles_to_hoard=15)  # Very conservative
```

---

### 2. AggressiveAgent
**Strategy**: Play tiles as soon as possible

**Behavior**:
- Never draws if it can play something
- Uses worm scoring to find the best immediate play
- Prioritizes playing new melds over adding to existing ones
- Only draws when absolutely necessary

**When to use**: When you want fast-paced, action-oriented gameplay

**Example**:
```python
AggressiveAgent("Aggressive")
```

---

### 3. BalancedAgent
**Strategy**: Middle ground between hoarding and aggressive

**Behavior**:
- Scores moves using balanced scoring function
- Plays when score is above threshold
- Will draw occasionally to improve hand
- Adapts based on game state

**Parameters**:
- `play_threshold`: Minimum score to play (0.0-1.0, default: 0.6)

**When to use**: When you want adaptable, well-rounded play

**Example**:
```python
BalancedAgent("Balanced", play_threshold=0.6)  # Default
BalancedAgent("BalancedHigh", play_threshold=0.8)  # More selective
```

---

### 4. SmartWormAgent
**Strategy**: Deep integration with worm.py solver

**Behavior**:
- Uses worm's `find_explicit_moves` for forced moves
- Attempts to solve hand optimally using `solve()`
- Finds moves considering both hand and table tiles
- Most sophisticated decision-making

**Parameters**:
- `strategy`: Base strategy ('balanced', 'aggressive', 'hoarding')

**When to use**: When you want the strongest baseline agent

**Example**:
```python
SmartWormAgent("SmartWorm", strategy="balanced")
SmartWormAgent("SmartWormAggressive", strategy="aggressive")
```

## Worm Integration Functions

The `worm_integration.py` module provides:

### Board Conversion
- `hand_to_board_matrix(hand)`: Convert hand tiles to 4x13 matrix
- `table_to_board_matrix(melds)`: Convert table melds to matrix
- `combine_boards(hand_board, table_board)`: Merge hand and table

### Move Finding
- `find_hand_only_moves(hand)`: Find moves using only hand tiles
- `find_combined_moves(hand, table)`: Find moves using hand + table
- `get_best_move_for_strategy(hand, table, strategy)`: Score-based selection

### Scoring Functions
- `score_move_for_hoarding(move, hand, table)`: Score for hoarding strategy
- `score_move_for_aggressive(move, hand, table)`: Score for aggressive strategy
- `score_move_for_balanced(move, hand, table)`: Score for balanced strategy

### Utilities
- `can_go_out_in_one_move(hand, table)`: Check if all tiles playable at once
- `count_playable_tiles(hand, table)`: Count immediately playable tiles

## Scoring Systems

### Hoarding Scoring
- Rewards: Long runs (5+ tiles), high numbers (easier to play later)
- Penalizes: Short runs (wastes tiles), middle numbers (too flexible)
- Goal: Save tiles for one explosive round

### Aggressive Scoring
- Rewards: Any run, any group, more tiles played
- Penalizes: Nothing (just play!)
- Goal: Empty hand as fast as possible

### Balanced Scoring
- Rewards: Medium-length runs (3-6 tiles), groups
- Considers: Board position, remaining flexibility
- Goal: Efficient play without wasting opportunities

## Running a Tournament

```python
from tournament import Tournament
from agent import (
    HoardingAgent, AggressiveAgent, BalancedAgent,
    SmartWormAgent, RandomAgent
)

# Create diverse agents
agents = [
    RandomAgent("Random"),
    HoardingAgent("Hoarding10", tiles_to_hoard=10),
    HoardingAgent("Hoarding15", tiles_to_hoard=15),
    AggressiveAgent("Aggressive"),
    BalancedAgent("Balanced"),
    SmartWormAgent("SmartWorm"),
]

# Run tournament
tournament = Tournament(agents, num_players=2)
results = tournament.run_random_matchups(
    num_games=100,
    show_progress=True,
    progress_interval=2.0
)

tournament.print_results()
```

## Expected Behaviors

### Hoarding vs Aggressive
- **Hoarding**: Will draw many times early, then play big moves
- **Aggressive**: Will play immediately, even suboptimal moves
- **Winner depends on**: How the game develops, opponent strategy

### SmartWorm vs Basic Agents
- **SmartWorm**: Should consistently beat Random and Heuristic
- **SmartWorm**: Competitive with well-tuned Hoarding/Aggressive
- **Advantage**: Uses constraint propagation, sees "obvious" moves

### Balanced as Baseline
- **Balanced**: Good all-around performance
- **Balanced**: Hard to exploit
- **Use for**: Benchmarking new agents

## Extending the Agents

To create custom agents using worm logic:

```python
from agent import RummikubAgent
from worm_integration import (
    find_hand_only_moves,
    score_move_for_balanced
)

class CustomAgent(RummikubAgent):
    def __init__(self, name="Custom"):
        super().__init__(name)
    
    def select_action(self, observation, valid_actions):
        # Decode hand
        hand = self._decode_hand(observation)
        
        # Find moves using worm
        moves = find_hand_only_moves(hand)
        
        # Score them your way
        scored = [(move, self._custom_score(move, hand)) 
                 for move in moves]
        
        # Pick best
        best = max(scored, key=lambda x: x[1])
        
        # Convert to action...
```

## Testing

Run the full test suite:

```bash
python tournament.py
```

This will:
1. Create 10 diverse agents
2. Run 100 random matchups
3. Show live progress with statistics
4. Display final rankings

## Performance Tips

1. **Hoarding**: Higher threshold = more patient, but might lose to fast players
2. **Aggressive**: Great for quick games, but might waste good tiles
3. **Balanced**: Most consistent performance across game lengths
4. **SmartWorm**: Best overall, but computationally more expensive

## Future Improvements

1. **Adaptive strategies**: Adjust hoarding threshold based on opponent behavior
2. **MCTS integration**: Use Monte Carlo Tree Search with worm for simulation
3. **Neural scoring**: Train neural net to score moves instead of hand-crafted rules
4. **Opponent modeling**: Track opponent tendencies and adapt strategy
