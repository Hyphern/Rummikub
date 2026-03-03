"""Game configuration settings."""

# Game Rules
INCLUDE_JOKERS = True  # Set to False to disable jokers

# Logging
LOG_GAME_OUTCOMES = False  # Set to True to log game results to a text file
LOG_FILE_PATH = "game_results.log"  # Text log output path
LOG_TO_CSV = True  # Set to True to record results to matchup_results.csv
CSV_FILE_PATH = "joker_matchup_results.csv"  # CSV output filename (stored in data/)

# Game Settings
DEFAULT_NUM_PLAYERS = 2
INITIAL_HAND_SIZE = 14
MAX_TURNS = 500  # Prevent infinite games
INITIAL_MELD_MIN_VALUE = 30  # 30 is standard, 15 for faster games

# AI/Solver Settings
USE_WORM_SOLVER = True  # Whether to use worm solver for move suggestions

# Performance
NUM_CPU_CORES = 4  # Number of parallel worker processes for tournaments
