# Rummikub Agent Guide

## Available Agents

### SmartWormAgent
Baseline solver using worm logic. No special configuration.

### HoarderAgent
Accumulates tiles before playing big melds.
- `tiles_to_hoard`: tiles to hold before playing (default: 10)

### AggressiveAgent
Plays tiles as soon as possible. Never draws if can play.

### StrategicAgent
Balanced play, prefers medium-large melds.

### GreedyAgent
Always plays the biggest meld available.

### CautiousAgent
Conservative play, prefers small melds and drawing.

### AdaptiveWormAgent
Learns weights from game outcomes.
- Saves learning to `adaptive_weights.json`

## Creating Agents

```python
from agent import (
    SmartWormAgent, 
    create_hoarder, 
    create_aggressive,
    create_strategic,
    create_greedy,
    create_cautious,
    create_adaptive,
)

agents = [
    SmartWormAgent("SmartWorm"),
    create_hoarder(),
    create_aggressive(),
    create_strategic(),
    create_greedy(),
    create_cautious(),
    create_adaptive(),
]
```

## Running a Tournament

```python
from tournament import Tournament

tournament = Tournament(agents, num_players=2)

# Parallel (faster)
tournament.run_random_matchups_parallel(num_games=200, num_workers=4)

# Sequential
tournament.run_random_matchups(num_games=200)

tournament.print_results()
```
