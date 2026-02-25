import copy
from itertools import combinations

# Board: 4 rows (colors) x 13 columns (numbers 1-13)
# Cell value = number of that tile in play (0, 1, or 2)
board_matrix = [
    [1, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2],
    [0, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1],
    [0, 1, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
]

COLOR_NAMES = ["Red", "Blue", "Yellow", "Black"]


def print_board(board):
    for i, row in enumerate(row for row in board):
        print(f"  {COLOR_NAMES[i]:>6}: {row}")


def is_solved(board):
    return all(cell == 0 for row in board for cell in row)


# ---------------------------------------------------------------------------
# Move application / undo
# ---------------------------------------------------------------------------

def apply_run(board, row, start, end):
    """Remove one tile per position in the run. Returns True if valid."""
    for col in range(start, end + 1):
        if board[row][col] <= 0:
            return False
    for col in range(start, end + 1):
        board[row][col] -= 1
    return True


def undo_run(board, row, start, end):
    for col in range(start, end + 1):
        board[row][col] += 1


def apply_group(board, col, colors):
    """Remove one tile per color in the group. Returns True if valid."""
    for r in colors:
        if board[r][col] <= 0:
            return False
    for r in colors:
        board[r][col] -= 1
    return True


def undo_group(board, col, colors):
    for r in colors:
        board[r][col] += 1


# ---------------------------------------------------------------------------
# Counting options for constraint analysis
# ---------------------------------------------------------------------------

def get_run_options(board, row, col):
    """Get all run start positions that include this column."""
    options = []
    # Check all windows of size 3 that include this column
    for start in range(max(0, col - 2), min(13 - 2, col + 1)):
        if all(board[row][c] > 0 for c in range(start, start + 3)):
            options.append(start)
    return options


def get_group_options(board, row, col):
    """Get all group combinations that include this tile."""
    available = [r for r in range(4) if board[r][col] > 0]
    if len(available) < 3:
        return []
    
    options = []
    for size in range(3, len(available) + 1):
        for combo in combinations(available, size):
            if row in combo:
                options.append(list(combo))
    return options


def count_run_options(board, row, col):
    """Count how many runs this tile can participate in."""
    count = 0
    for start in range(max(0, col - 2), min(13 - 2, col + 1)):
        if all(board[row][c] > 0 for c in range(start, start + 3)):
            count += 1
    return count


def count_group_options(board, row, col):
    """Count how many groups this tile can participate in."""
    available = [r for r in range(4) if board[r][col] > 0]
    if len(available) < 3:
        return 0
    
    count = 0
    for size in range(3, len(available) + 1):
        for combo in combinations(available, size):
            if row in combo:
                count += 1
    return count


def count_options(board, row, col):
    """Total valid sets this tile can join."""
    return count_run_options(board, row, col) + count_group_options(board, row, col)


# ---------------------------------------------------------------------------
# Finding definite moves (tiles with only 1 valid option)
# ---------------------------------------------------------------------------

def find_definite_moves(board):
    """Find all definite moves - tiles/sets with only one valid option.
    
    Returns list of moves that are forced (run or group).
    """
    definite = []
    
    # Find definite runs
    for row in range(4):
        for start in range(13):
            if board[row][start] <= 0:
                continue
            # Find maximal contiguous segment
            end = start
            while end + 1 < 13 and board[row][end + 1] > 0:
                end += 1
            
            seg_len = end - start + 1
            if seg_len >= 3:
                # Check each valid sub-run
                for length in range(3, seg_len + 1):
                    for s in range(start, start + seg_len - length + 1):
                        e = s + length - 1
                        # Count how many tiles in this run have only 1 option
                        constrained_tiles = sum(
                            1 for c in range(s, e + 1)
                            if count_options(board, row, c) == 1
                        )
                        if constrained_tiles > 0:
                            # Check: are the constrained tiles ONLY in this run?
                            all_forced = True
                            for c in range(s, e + 1):
                                if count_options(board, row, c) == 1:
                                    # This tile must be in a run, check if this is the only run
                                    if count_run_options(board, row, c) != 1:
                                        all_forced = False
                                        break
                            if all_forced:
                                definite.append(('run', row, s, e))
    
    # Find definite groups
    for col in range(13):
        available = [r for r in range(4) if board[r][col] > 0]
        if len(available) >= 3:
            # For each possible group size
            for size in range(3, len(available) + 1):
                for combo in combinations(available, size):
                    # Check if any tile in this combo has only this group option
                    constrained = False
                    all_forced = True
                    for r in combo:
                        if count_options(board, r, col) == 1:
                            constrained = True
                            # Verify this tile can ONLY group (not run)
                            if count_group_options(board, r, col) != 1:
                                all_forced = False
                                break
                    if constrained and all_forced:
                        definite.append(('group', col, list(combo)))
    
    # Remove duplicates
    seen = set()
    unique = []
    for move in definite:
        key = str(move)
        if key not in seen:
            seen.add(key)
            unique.append(move)
    return unique


# ---------------------------------------------------------------------------
# Constrained columns (8, 12) - special handling
# ---------------------------------------------------------------------------

def get_column_group_constraints(board, col):
    """Analyze group constraints for a specific column.
    
    Returns list of (colors_combo, constraint_score) where higher score = more forced.
    """
    available = [(r, board[r][col]) for r in range(4) if board[r][col] > 0]
    if len(available) < 3:
        return []
    
    color_indices = [r for r, _ in available]
    constraints = []
    
    for size in range(3, len(color_indices) + 1):
        for combo in combinations(color_indices, size):
            # Score: how constrained are the tiles in this group?
            score = 0
            for r in combo:
                opts = count_options(board, r, col)
                if opts == 1:
                    score += 10  # Definitely forced
                elif opts == 2:
                    score += 5   # Highly constrained
                elif opts <= 3:
                    score += 2   # Moderately constrained
            if score > 0:
                constraints.append((list(combo), score))
    
    # Sort by constraint score descending
    constraints.sort(key=lambda x: -x[1])
    return constraints


def find_constrained_column_moves(board, priority_cols=None):
    """Find moves in priority columns that are highly constrained.
    
    Default priority: columns 8 and 12 (indices 8, 12 = numbers 9 and 13)
    """
    if priority_cols is None:
        priority_cols = [8, 12]
    
    moves = []
    for col in priority_cols:
        if col >= 13:
            continue
        constraints = get_column_group_constraints(board, col)
        for colors, score in constraints:
            if score >= 10:  # At least one tile is forced
                # Verify tiles are still available
                if all(board[r][col] > 0 for r in colors):
                    moves.append(('group', col, colors, score))
    
    # Sort by constraint score
    moves.sort(key=lambda x: -x[3])
    return [(m[0], m[1], m[2]) for m in moves]  # Remove score


# ---------------------------------------------------------------------------
# General move generation (for backtracking)
# ---------------------------------------------------------------------------

def find_all_runs(board):
    """Find all valid runs (3+ consecutive tiles of same color)."""
    runs = []
    for row in range(4):
        for start in range(13):
            if board[row][start] <= 0:
                continue
            end = start
            while end + 1 < 13 and board[row][end + 1] > 0:
                end += 1
            seg_len = end - start + 1
            if seg_len >= 3:
                for length in range(3, seg_len + 1):
                    for s in range(start, start + seg_len - length + 1):
                        runs.append((row, s, s + length - 1))
    return list(set(runs))


def find_all_groups(board):
    """Find all valid groups (3+ tiles of same number, different colors)."""
    groups = []
    for col in range(13):
        available = [r for r in range(4) if board[r][col] > 0]
        if len(available) >= 3:
            for size in range(3, len(available) + 1):
                for combo in combinations(available, size):
                    groups.append((col, list(combo)))
    return groups


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def is_valid_board_state(board):
    """Check that every remaining tile can participate in at least one set."""
    for row in range(4):
        for col in range(13):
            if board[row][col] > 0:
                run_opts = count_run_options(board, row, col)
                group_opts = count_group_options(board, row, col)
                if run_opts == 0 and group_opts == 0:
                    return False
    return True


# ---------------------------------------------------------------------------
# Tiered Solver
# ---------------------------------------------------------------------------

def apply_definite_moves(board, solution_list):
    """Iteratively apply all definite moves. Returns True if progress made."""
    progress = False
    while True:
        definite = find_definite_moves(board)
        if not definite:
            break
        for move in definite:
            if move[0] == 'run':
                _, row, start, end = move
                if apply_run(board, row, start, end):
                    solution_list.append(move)
                    progress = True
            else:
                _, col, colors = move
                if apply_group(board, col, colors):
                    solution_list.append(move)
                    progress = True
    return progress


def try_constrained_columns(board, solution_list, depth=0, max_depth=10):
    """Try constrained column moves with backtracking."""
    if depth > max_depth:
        return False
    
    # First apply any definite moves
    apply_definite_moves(board, solution_list)
    
    if is_solved(board):
        return True
    
    # Try constrained column moves
    moves = find_constrained_column_moves(board)
    
    for move in moves:
        if move[0] == 'group':
            _, col, colors = move
            if not apply_group(board, col, colors):
                continue
            
            solution_list.append(move)
            
            # Recursively solve
            if try_constrained_columns(board, solution_list, depth + 1, max_depth):
                return True
            
            # Backtrack
            solution_list.pop()
            undo_group(board, col, colors)
    
    return False


def dfs_solve(board, solution_list, depth=0, max_depth=50):
    """General DFS solver for remaining tiles."""
    if is_solved(board):
        return True
    
    if depth > max_depth:
        return False
    
    # Generate all moves
    runs = find_all_runs(board)
    groups = find_all_groups(board)
    
    # Score and sort moves by constraint
    scored_moves = []
    
    for run in runs:
        row, start, end = run
        if any(board[row][c] <= 0 for c in range(start, end + 1)):
            continue
        min_opts = min(count_options(board, row, c) for c in range(start, end + 1))
        scored_moves.append((min_opts, 0, 'run', run))  # 0 priority for runs
    
    for group in groups:
        col, colors = group
        if any(board[r][col] <= 0 for r in colors):
            continue
        min_opts = min(count_options(board, r, col) for r in colors)
        scored_moves.append((min_opts, 1, 'group', group))  # 1 priority for groups
    
    # Sort: most constrained first, groups before runs
    scored_moves.sort(key=lambda x: (x[0], -x[1]))
    
    for _, _, move_type, move_data in scored_moves:
        if move_type == 'run':
            row, start, end = move_data
            if not apply_run(board, row, start, end):
                continue
            
            solution_list.append(('run', row, start, end))
            
            if is_valid_board_state(board):
                if dfs_solve(board, solution_list, depth + 1, max_depth):
                    return True
            
            solution_list.pop()
            undo_run(board, row, start, end)
        else:
            col, colors = move_data
            if not apply_group(board, col, colors):
                continue
            
            solution_list.append(('group', col, colors))
            
            if is_valid_board_state(board):
                if dfs_solve(board, solution_list, depth + 1, max_depth):
                    return True
            
            solution_list.pop()
            undo_group(board, col, colors)
    
    return False


def solve(board):
    """Main solver using tiered strategy."""
    solution = []
    
    # Tier 1: Apply definite moves
    print("Phase 1: Applying definite moves...")
    apply_definite_moves(board, solution)
    print(f"  Applied {len(solution)} definite moves")
    
    if is_solved(board):
        return solution
    
    # Tier 2: Handle constrained columns (8, 12)
    print("Phase 2: Handling constrained columns...")
    constrained_solution = []
    if try_constrained_columns(board, constrained_solution):
        solution.extend(constrained_solution)
        print(f"  Applied {len(constrained_solution)} constrained column moves")
    else:
        print("  No constrained column solution found, proceeding to DFS")
    
    if is_solved(board):
        return solution
    
    # Tier 3: General DFS
    print("Phase 3: General DFS...")
    remaining_solution = []
    if dfs_solve(board, remaining_solution):
        solution.extend(remaining_solution)
        print(f"  DFS found {len(remaining_solution)} additional moves")
    else:
        print("  DFS failed to find solution")
        return None
    
    return solution if is_solved(board) else None


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def format_move(move):
    if move[0] == 'run':
        _, row, start, end = move
        numbers = list(range(start + 1, end + 2))
        return f"Run:   {COLOR_NAMES[row]} {numbers}"
    else:
        _, col, colors = move
        number = col + 1
        color_list = [COLOR_NAMES[c] for c in colors]
        return f"Group: Number {number} [{', '.join(color_list)}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    board = copy.deepcopy(board_matrix)

    print("Starting board:")
    print_board(board)
    print()

    total_tiles = sum(cell for row in board for cell in row)
    print(f"Total tiles: {total_tiles}")
    print()

    print("Solving with tiered strategy...")
    solution = solve(board)

    if solution is None:
        print("\nNo solution found!")
        print("\nRemaining board:")
        print_board(board)
    else:
        print(f"\nSolved! {len(solution)} sets formed:\n")
        for i, move in enumerate(solution, 1):
            print(f"  {i:2d}. {format_move(move)}")

        print("\nFinal board:")
        print_board(board)
        remaining = sum(cell for row in board for cell in row)
        print(f"\nRemaining tiles: {remaining}")
        if remaining == 0:
            print("SUCCESS - Board fully decomposed!")
        else:
            print("ERROR - Tiles remain!")
