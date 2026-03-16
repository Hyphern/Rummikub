# Rummikub-Carden

A Python implementation of the Rummikub board game with AI agents and machine learning capabilities.

## Overview

Rummikub-Carden is a programmatic Rummikub game implementation featuring multiple AI agents with different playing strategies. The project includes a tournament system for comparing agent performance and an adaptive learning mechanism that evolves strategies based on game outcomes.

## Features

- **Complete Rummikub Implementation**: Full game rules including runs, groups, jokers, and board management
- **Multiple AI Agents**: 7 different agent strategies from aggressive to cautious
- **Tournament System**: Run head-to-head matches or round-robin tournaments
- **Machine Learning**: Adaptive agent that learns from game outcomes
- **Worm Solver Integration**: Optimal play calculation using the Worm solver

## Installation

```bash
cd Rummikub-Carden/src
pip install -r requirements.txt
```

## Requirements

- Python 3.11+
- numpy

## Quick Start

```python
from tournament import Tournament
from agent import (
    SmartWormAgent, 
    create_hoarder, 
    create_aggressive,
    create_strategic,
    create_greedy,
    create_cautious,
    create_adaptive,
)

# Create agents
agents = [
    SmartWormAgent("SmartWorm"),
    create_aggressive(),
    create_strategic(),
    create_greedy(),
]

# Run tournament
tournament = Tournament(agents, num_players=2)
tournament.run_random_matchups_parallel(num_games=200, num_workers=4)
tournament.print_results()
```

## Available Agents

| Agent | Strategy |
|-------|----------|
| **SmartWormAgent** | Baseline solver using worm logic |
| **HoarderAgent** | Accumulates tiles before playing big melds |
| **AggressiveAgent** | Plays tiles as soon as possible |
| **StrategicAgent** | Balanced play, prefers medium-large melds |
| **GreedyAgent** | Always plays the biggest meld available |
| **CautiousAgent** | Conservative play, prefers small melds |
| **AdaptiveWormAgent** | Learns weights from game outcomes |

## Project Structure

```
Rummikub-Carden/
├── src/
│   ├── tile.py              # Tile and TileSet classes
│   ├── meld.py              # Meld validation (runs and groups)
│   ├── game_state.py        # Game state management
│   ├── agent.py             # Agent base class and implementations
│   ├── tournament.py        # Tournament system
│   ├── ml_environment.py   # ML environment wrapper
│   ├── wormed.py            # Cached worm solver wrapper
│   ├── worm_integration.py  # Worm solver integration
│   └── requirements.txt     # Python dependencies
├── data/
│   ├── adaptive_weights.json # Adaptive agent learned weights
│   ├── matchup_results.csv    # Historical matchup results
│   └── analyze_results.py     # Analysis script
└── testing_files/            # Test and debug scripts
```

## Running Tournaments

### Parallel (faster, uses multiple CPU cores)
```python
tournament.run_random_matchups_parallel(num_games=200, num_workers=4)
```

### Sequential
```python
tournament.run_random_matchups(num_games=200)
```

## Game Rules

Rummikub is a tile-based game where players form sets of tiles:

- **Runs**: Sequences of 3+ consecutive numbers in the same color (e e.g., Red 4-5-6)
- **Groups**: 3 or 4 of the same number in different colors (e.g., Red 7, Blue 7, Black 7)
- **Jokers**: Wild tiles that can substitute any tile
- Players must meld (play valid sets) to go out
- The player with the lowest point total in their hand wins

## License

MIT
