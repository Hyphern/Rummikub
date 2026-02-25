from functools import lru_cache

# Global cache for solved board states (transposition table)
_solve_cache = {}

def board_to_hashable(board_matrix):
    """Convert board to hashable form for caching."""
    return tuple(tuple(row) for row in board_matrix)

def definite_moves(board_matrix):
    moves = []
    for i in range(len(board_matrix)):
        for j in range(len(board_matrix[i])):
            if board_matrix[i][j] == 0:
                moves.append((i, j))
    return moves

def find_explicit_moves(board_matrix):
    """Find definite (explicit) moves that must be played.
    
    Based on steps.txt logic:
    - A run is definite if all tiles have only 1 run option
    - A group is definite if tiles with count 2 can only form 1 run (forcing them to group),
      and the column has exactly 3 tiles
    
    Returns list of (move_type, details) tuples.
    """
    from itertools import combinations
    
    moves = []
    rows = len(board_matrix)
    cols = len(board_matrix[0]) if rows > 0 else 0
    
    def count_run_options(row, col):
        """Count how many different runs this tile can join."""
        if board_matrix[row][col] == 0:
            return 0
        count = 0
        # Only count runs within contiguous segments
        # Find segment boundaries
        seg_start = col
        while seg_start > 0 and board_matrix[row][seg_start - 1] > 0:
            seg_start -= 1
        seg_end = col
        while seg_end + 1 < cols and board_matrix[row][seg_end + 1] > 0:
            seg_end += 1
        
        # Count valid run windows within this segment that include col
        for start in range(max(seg_start, col - 2), min(seg_end - 1, col + 1) + 1):
            if start + 2 <= seg_end and all(board_matrix[row][c] > 0 for c in range(start, start + 3)):
                count += 1
        return count
    
    def get_valid_runs(row, col):
        """Get list of all valid runs (start, end) that include this tile."""
        if board_matrix[row][col] == 0:
            return []
        runs = []
        # Find segment boundaries
        seg_start = col
        while seg_start > 0 and board_matrix[row][seg_start - 1] > 0:
            seg_start -= 1
        seg_end = col
        while seg_end + 1 < cols and board_matrix[row][seg_end + 1] > 0:
            seg_end += 1
        
        # Find all valid runs in this segment
        for start in range(seg_start, seg_end - 1):
            for end in range(start + 2, min(seg_end + 1, start + 13)):
                if all(board_matrix[row][c] > 0 for c in range(start, end + 1)):
                    if start <= col <= end:
                        runs.append((start, end))
        return runs
    
    # Find definite runs - all tiles have exactly 1 run option
    for row in range(rows):
        col = 0
        while col < cols:
            if board_matrix[row][col] == 0:
                col += 1
                continue
            
            # Find segment
            seg_start = col
            while seg_start > 0 and board_matrix[row][seg_start - 1] > 0:
                seg_start -= 1
            seg_end = col
            while seg_end + 1 < cols and board_matrix[row][seg_end + 1] > 0:
                seg_end += 1
            
            seg_len = seg_end - seg_start + 1
            if seg_len >= 3:
                # Check each possible run
                for run_start in range(seg_start, seg_end - 1):
                    for run_end in range(run_start + 2, min(seg_end + 1, run_start + 13)):
                        if not all(board_matrix[row][c] > 0 for c in range(run_start, run_end + 1)):
                            continue
                        
                        # Check if all tiles have exactly 1 run option
                        all_definite = True
                        for c in range(run_start, run_end + 1):
                            if count_run_options(row, c) != 1:
                                all_definite = False
                                break
                        
                        if all_definite:
                            moves.append(('run', row, run_start, run_end))
            
            col = seg_end + 1
    
    # Find definite groups based on constraint propagation
    # A group is definite if:
    # 1. Column has exactly 3 tiles, AND
    # 2. Either:
    #    a) A count-2 tile is "consumed" by a forced run (from a count-1 tile with 1 option),
    #       leaving it with only 1 remaining option for its second copy
    #    b) A count-1 tile has 0 run options (must use group)
    #    c) Multiple count-1 tiles are forced into different runs that share this column,
    #       forcing the count-2 tile to use a group
    for col in range(cols):
        available = [(r, board_matrix[r][col]) for r in range(rows) if board_matrix[r][col] > 0]
        if len(available) != 3:
            continue
        
        colors = [r for r, _ in available]
        count1_tiles = [(r, c) for r, c in available if board_matrix[r][c] == 1]
        count2_tiles = [(r, c) for r, c in available if board_matrix[r][c] == 2]
        
        # Check case b: any count-1 tile has 0 run options (must use group)
        if len(count1_tiles) > 0:
            has_zero_option_tile = False
            for r, _ in count1_tiles:
                if count_run_options(r, col) == 0:
                    has_zero_option_tile = True
                    break
            
            if has_zero_option_tile:
                moves.append(('group', col, colors))
                continue
        
        # Check case c: count-1 tiles with conflicting forced runs
        if len(count1_tiles) >= 2 and len(count2_tiles) >= 1:
            # Check if count-1 tiles are forced into different runs
            forced_runs_per_tile = {}
            for r, _ in count1_tiles:
                runs = get_valid_runs(r, col)
                # Filter to runs where this tile has only 1 option
                forced = [run for run in runs if count_run_options(r, col) == 1]
                if len(forced) == 1:
                    forced_runs_per_tile[r] = forced[0]
            
            # If we have forced runs for count-1 tiles, check if they conflict
            if len(forced_runs_per_tile) >= 2:
                # The count-2 tile would need to be in multiple runs simultaneously
                # which is impossible, so it must use a group
                moves.append(('group', col, colors))
                continue
        
        # Check case a: count-2 tile consumed by forced run
        for r, count in available:
            if count != 2:
                continue
            
            # Get all runs this tile can be in
            tile_runs = get_valid_runs(r, col)
            if len(tile_runs) == 0:
                continue
            
            # Check if any run is "forced" because it contains a count-1 tile with only 1 option
            forced_runs = []
            for run in tile_runs:
                run_start, run_end = run
                # Check each position in this run
                for c in range(run_start, run_end + 1):
                    if board_matrix[r][c] == 1 and count_run_options(r, c) == 1:
                        # This count-1 tile is forced into this run
                        forced_runs.append(run)
                        break
            
            # If there's a forced run, it consumes one copy
            # Check remaining options for second copy
            if len(forced_runs) > 0:
                remaining_runs = [run for run in tile_runs if run not in forced_runs]
                # After forced run, how many options remain?
                remaining_options = len(remaining_runs)
                
                # If only 1 run remains (or 0), and column has 3 tiles, group is definite
                if remaining_options <= 1:
                    moves.append(('group', col, colors))
                    break
    
    # Remove duplicates
    seen = set()
    unique = []
    for move in moves:
        key = str(move)
        if key not in seen:
            seen.add(key)
            unique.append(move)
    
    return unique

def apply_move(board_matrix, move):
    """Apply a move to the board, reducing tile counts. Returns a new board."""
    from copy import deepcopy
    new_board = deepcopy(board_matrix)
    
    if move[0] == 'run':
        _, row, start, end = move
        for col in range(start, end + 1):
            if new_board[row][col] > 0:
                new_board[row][col] -= 1
    elif move[0] == 'group':
        _, col, colors = move
        for row in colors:
            if new_board[row][col] > 0:
                new_board[row][col] -= 1
    
    return new_board


def iterate_explicit_moves(board_matrix):
    """Find and apply all explicit moves iteratively until no more can be found.
    
    Returns list of all moves found in order.
    """
    from copy import deepcopy
    
    board = deepcopy(board_matrix)
    all_moves = []
    step = 0
    
    print(f"Step {step}: Initial state")
    for row in board:
        print(f"  {row}")
    print()
    
    while True:
        moves = find_explicit_moves(board)
        if not moves:
            break
        
        for move in moves:
            step += 1
            all_moves.append(move)
            board = apply_move(board, move)
            
            print(f"Step {step}: Applied {move}")
            for row in board:
                print(f"  {row}")
            print()
    
    print(f"No more explicit moves found. Total moves: {len(all_moves)}")
    return all_moves


def debug_explicit_moves(board_matrix):
    """Debug function to see why moves are/aren't being found."""
    from itertools import combinations
    
    rows = len(board_matrix)
    cols = len(board_matrix[0]) if rows > 0 else 0
    
    def count_run_options(row, col):
        if board_matrix[row][col] == 0:
            return 0
        count = 0
        seg_start = col
        while seg_start > 0 and board_matrix[row][seg_start - 1] > 0:
            seg_start -= 1
        seg_end = col
        while seg_end + 1 < cols and board_matrix[row][seg_end + 1] > 0:
            seg_end += 1
        
        for start in range(max(seg_start, col - 2), min(seg_end - 1, col + 1) + 1):
            if start + 2 <= seg_end and all(board_matrix[row][c] > 0 for c in range(start, start + 3)):
                count += 1
        return count
    
    print("Checking each column for potential groups:")
    for col in range(cols):
        available = [(r, board_matrix[r][col]) for r in range(rows) if board_matrix[r][col] > 0]
        if len(available) >= 3:
            print(f"\nColumn {col}:")
            print(f"  Available tiles: {available}")
            print(f"  Number of tiles: {len(available)}")
            
            for r, count in available:
                run_opts = count_run_options(r, col)
                print(f"  Row {r}: count={count}, run_options={run_opts}")


@lru_cache(maxsize=None)
def get_all_runs(board_hashable):
    """Get all valid runs on the board."""
    # Convert back to list for internal use
    board_matrix = [list(row) for row in board_hashable]
    runs = []
    rows = len(board_matrix)
    cols = len(board_matrix[0]) if rows > 0 else 0
    
    for row in range(rows):
        col = 0
        while col < cols:
            if board_matrix[row][col] == 0:
                col += 1
                continue
            
            # Find segment
            seg_start = col
            while seg_start > 0 and board_matrix[row][seg_start - 1] > 0:
                seg_start -= 1
            seg_end = col
            while seg_end + 1 < cols and board_matrix[row][seg_end + 1] > 0:
                seg_end += 1
            
            seg_len = seg_end - seg_start + 1
            if seg_len >= 3:
                # Generate all valid runs in this segment
                for start in range(seg_start, seg_end - 1):
                    for end in range(start + 2, min(seg_end + 1, start + 13)):
                        if all(board_matrix[row][c] > 0 for c in range(start, end + 1)):
                            runs.append(('run', row, start, end))
            
            col = seg_end + 1
    
    return runs


@lru_cache(maxsize=None)
def get_all_groups(board_hashable):
    """Get all valid groups on the board."""
    # Convert back to list for internal use
    board_matrix = [list(row) for row in board_hashable]
    from itertools import combinations
    
    groups = []
    rows = len(board_matrix)
    cols = len(board_matrix[0]) if rows > 0 else 0
    
    for col in range(cols):
        available = [r for r in range(rows) if board_matrix[r][col] > 0]
        if len(available) >= 3:
            for size in range(3, len(available) + 1):
                for combo in combinations(available, size):
                    groups.append(('group', col, list(combo)))
    
    return groups


@lru_cache(maxsize=None)
def get_tile_moves(board_hashable, row, col):
    """Get all valid moves that include this tile."""
    # Convert back to list for internal use
    board_matrix = [list(row) for row in board_hashable]
    
    if board_matrix[row][col] == 0:
        return ()
    
    all_moves = list(get_all_runs(board_hashable)) + list(get_all_groups(board_hashable))
    tile_moves = []
    
    for move in all_moves:
        if move[0] == 'run':
            _, r, start, end = move
            if r == row and start <= col <= end:
                tile_moves.append(move)
        else:  # group
            _, c, colors = move
            if c == col and row in colors:
                tile_moves.append(move)
    
    return tile_moves


def is_solved(board_matrix):
    """Check if board is fully solved (all tiles used)."""
    return all(cell == 0 for row in board_matrix for cell in row)


def is_valid_state(board_matrix):
    """Check if current state is valid (no tile has 0 options while still present)."""
    rows = len(board_matrix)
    cols = len(board_matrix[0]) if rows > 0 else 0
    
    board_hashable = board_to_hashable(board_matrix)
    
    for row in range(rows):
        for col in range(cols):
            if board_matrix[row][col] > 0:
                moves = get_tile_moves(board_hashable, row, col)
                if len(moves) == 0:
                    return False
    return True


def solve(board_matrix, solution=None, depth=0, max_depth=100):
    """Solve the board using DFS with constraint propagation.
    
    Algorithm:
    1. Apply all definite moves
    2. If solved, return solution
    3. If invalid, backtrack
    4. Find tile with minimum options (MRV)
    5. Try each option recursively
    
    Returns solution list or None if no solution.
    """
    from copy import deepcopy
    
    if solution is None:
        solution = []
    
    if depth > max_depth:
        return None
    
    # Check transposition table first
    board_hashable = board_to_hashable(board_matrix)
    if board_hashable in _solve_cache:
        cached = _solve_cache[board_hashable]
        if cached is not None:
            # Return solution with the moves appended
            return solution + cached
        else:
            return None
    
    board = deepcopy(board_matrix)
    
    # Step 1: Apply all definite moves
    while True:
        definite = find_explicit_moves(board)
        if not definite:
            break
        for move in definite:
            solution.append(move)
            board = apply_move(board, move)
    
    board_hashable = board_to_hashable(board)
    
    # Step 2: Check if solved
    if is_solved(board):
        _solve_cache[board_hashable] = []  # Empty solution from here
        return solution
    
    # Step 3: Check validity
    if not is_valid_state(board):
        _solve_cache[board_hashable] = None  # Cache failure
        return None
    
    # Step 4: Find tile with minimum options (MRV heuristic)
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    
    min_options = float('inf')
    best_tile = None
    best_moves = []
    
    for row in range(rows):
        for col in range(cols):
            if board[row][col] > 0:
                tile_moves = get_tile_moves(board_hashable, row, col)
                num_options = len(tile_moves)
                
                if num_options < min_options:
                    min_options = num_options
                    best_tile = (row, col)
                    best_moves = tile_moves
                    
                    # If we find a tile with only 1 option, use it immediately
                    if num_options == 1:
                        break
        if min_options == 1:
            break
    
    if best_tile is None or len(best_moves) == 0:
        _solve_cache[board_hashable] = None  # Cache failure
        return None
    
    # Sort moves to try longer ones first (uses more tiles = faster resolution)
    def move_size(m):
        if m[0] == 'run':
            return -(m[3] - m[2] + 1)  # Longer runs first
        else:
            return -len(m[2])  # Larger groups first
    best_moves = sorted(best_moves, key=move_size)
    
    # Step 5: Try each option recursively
    for move in best_moves:
        new_solution = solution.copy()
        new_solution.append(move)
        new_board = apply_move(deepcopy(board), move)
        
        result = solve(new_board, new_solution, depth + 1, max_depth)
        if result is not None:
            # Cache the solution (relative to this board state)
            remaining = result[len(solution):]
            _solve_cache[board_hashable] = remaining
            return result
    
    # No solution found from this state
    _solve_cache[board_hashable] = None  # Cache failure
    return None


def format_move(move, color_names=None):
    """Convert a move tuple to human-readable format.
    
    Colors are displayed in alphabetical order.
    """
    if color_names is None:
        # Map row indices to color names
        color_names = ["Red", "Blue", "Yellow", "Black"]
    
    if move[0] == 'run':
        _, row, start, end = move
        color = color_names[row]
        # Columns correspond to numbers 1-13
        numbers = list(range(start + 1, end + 2))
        return f"Run:   {color} {numbers}"
    else:  # group
        _, col, colors = move
        number = col + 1  # Column index to number (0->1, 1->2, etc.)
        # Sort colors alphabetically
        color_list = sorted([color_names[c] for c in colors])
        return f"Group: {number} [{', '.join(color_list)}]"


def format_solution(solution, color_names=None):
    """Format the entire solution with human-readable moves."""
    if color_names is None:
        color_names = ["Black", "Blue", "Red", "Yellow"]
    
    formatted = []
    for i, move in enumerate(solution, 1):
        formatted.append(f"{i:2d}. {format_move(move, color_names)}")
    return formatted


def solved(board_matrix):
    from copy import deepcopy
    
    print("=" * 60)
    print("Solving board with DFS + Constraint Propagation:")
    print("=" * 60)
    print()
    
    print("Initial board:")
    for i, row in enumerate(board_matrix):
        print(f"  Row {i}: {row}")
    print()
    
    solution = solve(board_matrix)
    
    if solution:
        print(f"Solution found with {len(solution)} moves!")
        print()
        
        # Display formatted solution
        print("Solution:")
        formatted_moves = format_solution(solution)
        for move_str in formatted_moves:
            print(f"  {move_str}")
        print()
        
        # Verify solution
        board = deepcopy(board_matrix)
        for move in solution:
            board = apply_move(board, move)
        
        print("Final board:")
        for i, row in enumerate(board):
            print(f"  Row {i}: {row}")
        
        remaining = sum(cell for row in board for cell in row)
        print(f"\nRemaining tiles: {remaining}")
        
        if remaining == 0:
            print("SUCCESS - Board fully solved!")
        else:
            print("ERROR - Tiles remain!")
    else:
        print("No solution found.")