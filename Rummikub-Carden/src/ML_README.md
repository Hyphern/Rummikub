# Rummikub Machine Learning System

A machine learning framework for training and evaluating Rummikub agents.

## Project Structure

```
Rummikub-Carden/
├── src/
│   ├── tile.py              # Tile and TileSet classes
│   ├── meld.py               # Meld validation (runs and groups)
│   ├── game_state.py         # Game state management
│   ├── ml_environment.py     # ML environment wrapper
│   ├── agent.py              # Agent base class and implementations
│   ├── tournament.py         # Tournament system for training/evaluation
│   ├── wormed.py            # Cached worm solver wrapper
│   ├── worm_integration.py   # Worm solver integration
│   └── requirements.txt      # Python dependencies
├── data/
│   ├── adaptive_weights.json # Adaptive agent learned weights
│   ├── matchup_results.csv  # Historical matchup results
│   └── analyze_results.py   # Analysis script
└── plots/                   # Generated plots
```

## Installation

```bash
cd src
pip install -r requirements.txt
```

## Quick Start

Run a tournament:
```bash
cd src
python tournament.py
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

## Available Agents

- **SmartWorm**: Baseline solver using worm logic
- **Hoarder**: Accumulates tiles, plays big melds
- **Aggressive**: Plays tiles as soon as possible
- **Strategic**: Balanced play, prefers medium-large melds
- **Greedy**: Always plays biggest meld available
- **Cautious**: Conservative, small melds
- **Adaptive**: Learns weights from game outcomes

## Data Files

- `adaptive_weights.json` - Learned weights for Adaptive agent
- `matchup_results.csv` - Historical head-to-head results

## Requirements

- Python 3.11+
- numpy
