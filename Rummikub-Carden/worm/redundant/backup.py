board_matrix = [[1,1,1,0,1,2,2,1,1,2,2,2,2],
                [0,2,2,2,2,2,0,0,1,0,0,0,1],
                [0,1,1,1,1,2,2,2,2,2,2,1,1],
                [0,1,2,1,1,0,0,0,1,0,0,0,1]]

def matrix_print(matrix):
    for row in matrix:
        print(' '.join(str(cell) for cell in row))

def get_column(board, col_idx):
    """Get all values in a column."""
    return [board[i][col_idx] for i in range(len(board))]

def find_groups(board):
    """Find all possible groups (same number, 3+ tiles from different colors).
    Now handles colors with count > 1 by using same color multiple times."""
    groups = []
    num_cols = len(board[0])
    
    for col in range(num_cols):
        column = get_column(board, col)
        # Get all available tiles with their colors (including duplicates)
        available_tiles = []
        for color_idx, count in enumerate(column):
            for _ in range(count):
                available_tiles.append(color_idx)
        
        # Need at least 3 tiles total
        if len(available_tiles) >= 3:
            from itertools import combinations
            # Generate all combinations of 3 tiles
            for combo in combinations(range(len(available_tiles)), 3):
                colors = [available_tiles[i] for i in combo]
                groups.append((col, colors))
    
    return groups

def is_board_empty(matrix):
    for row in matrix:
        for cell in row:
            if cell != 0:
                return False
    return True

def copy_board(matrix):
    return [row[:] for row in matrix]

def apply_group(board, col_idx, colors):
    """Apply a group: remove one tile of the given number from each color.
    Handles duplicate colors (same color appearing multiple times in group)."""
    new_board = copy_board(board)
    for color in colors:
        if new_board[color][col_idx] > 0:
            new_board[color][col_idx] -= 1
    return new_board

def apply_run(board, row_idx, start, end):
    new_board = copy_board(board)
    for col in range(start, end + 1):
        if new_board[row_idx][col] > 0:
            new_board[row_idx][col] -= 1
    return new_board

def can_form_run(board, row_idx, col_idx):
    """Check if tile can be part of any run."""
    row = board[row_idx]
    
    if col_idx >= 2 and row[col_idx-2] > 0 and row[col_idx-1] > 0:
        return True
    if col_idx >= 1 and col_idx <= len(row) - 2 and row[col_idx-1] > 0 and row[col_idx+1] > 0:
        return True
    if col_idx <= len(row) - 3 and row[col_idx+1] > 0 and row[col_idx+2] > 0:
        return True
    
    return False

def can_form_group(board, row_idx, col_idx):
    """Check if tile can be part of any group."""
    column = get_column(board, col_idx)
    available = sum(1 for count in column if count > 0)
    return available >= 3

def find_all_runs(board):
    """Find all valid runs of 3+ consecutive tiles."""
    runs = []
    
    for row_idx in range(len(board)):
        row = board[row_idx]
        
        seq_start = None
        for j in range(len(row)):
            if row[j] > 0:
                if seq_start is None:
                    seq_start = j
            else:
                if seq_start is not None:
                    seq_end = j - 1
                    
                    for run_start in range(seq_start, seq_end - 1):
                        for run_end in range(run_start + 2, min(seq_end + 1, run_start + 5)):
                            runs.append((row_idx, run_start, run_end))
                    
                    seq_start = None
        
        if seq_start is not None:
            seq_end = len(row) - 1
            for run_start in range(seq_start, seq_end - 1):
                for run_end in range(run_start + 2, min(seq_end + 1, run_start + 5)):
                    runs.append((row_idx, run_start, run_end))
    
    return runs

def find_definite_moves(board):
    """Find tiles with limited options."""
    moves = []
    
    for row_idx in range(len(board)):
        for col_idx in range(len(board[row_idx])):
            if board[row_idx][col_idx] == 0:
                continue
            
            can_run = can_form_run(board, row_idx, col_idx)
            can_group = can_form_group(board, row_idx, col_idx)
            
            if can_group and not can_run:
                moves.append(('only_group', row_idx, col_idx))
            elif can_run and not can_group:
                moves.append(('only_run', row_idx, col_idx))
    
    return moves

def count_tiles(board):
    """Count total tiles remaining."""
    return sum(sum(row) for row in board)

def is_valid_board_state(board):
    """Check if board is in a potentially solvable state.
    Returns False if there are tiles that can never be placed."""
    for row_idx in range(len(board)):
        for col_idx in range(len(board[row_idx])):
            if board[row_idx][col_idx] == 0:
                continue
            
            # Check if this tile can ever form a valid set
            can_run = can_form_run(board, row_idx, col_idx)
            can_group = can_form_group(board, row_idx, col_idx)
            
            if not can_run and not can_group:
                return False
    
    return True

def find_stranded_tiles(board):
    """Find tiles that can only form groups (no run option).
    Only returns tiles that genuinely cannot participate in any run."""
    stranded = []
    
    for row_idx in range(len(board)):
        for col_idx in range(len(board[row_idx])):
            if board[row_idx][col_idx] == 0:
                continue
            
            can_run = can_form_run(board, row_idx, col_idx)
            can_group = can_form_group(board, row_idx, col_idx)
            
            # Tile can only group, not run
            if can_group and not can_run:
                stranded.append(('group_only', row_idx, col_idx))
    
    return stranded

def find_maximal_run(board, row_idx, start):
    """From a starting position, find the longest consecutive run."""
    row = board[row_idx]
    end = start
    while end + 1 < len(row) and row[end + 1] > 0:
        end += 1
    return end

def solve_greedy(board):
    """Solve using greedy approach: 
    1. Stranded tiles (can only group) 
    2. Definite runs (tiles that can only run)
    3. Any remaining groups
    4. Any remaining runs"""
    moves = []
    current_board = copy_board(board)
    max_iterations = 100
    iteration = 0
    
    while not is_board_empty(current_board) and iteration < max_iterations:
        iteration += 1
        made_progress = False
        
        # 0. Handle stranded tiles first (can only group, can't run)
        stranded = find_stranded_tiles(current_board)
        
        for strand_type, row_idx, col_idx in stranded:
            column = get_column(current_board, col_idx)
            available_colors = [i for i, count in enumerate(column) if count > 0]
            
            if len(available_colors) >= 3 and row_idx in available_colors:
                other_colors = [c for c in available_colors if c != row_idx][:2]
                group_colors = [row_idx] + other_colors
                
                new_board = apply_group(current_board, col_idx, group_colors)
                if is_valid_board_state(new_board):
                    current_board = new_board
                    moves.append(f"Group (stranded): number {col_idx+1}, colors {group_colors}")
                    made_progress = True
                    break
        
        if made_progress:
            continue
        
        # 0.5 Find "isolated runs" — consecutive sequences in a row that are
        # bounded on both sides by 0 (or board edge). These MUST be used as runs.
        # Process most constrained rows first (fewest total tiles).
        row_tile_counts = [(sum(current_board[i]), i) for i in range(len(current_board))]
        row_tile_counts.sort()  # fewest tiles first
        
        for _, row_idx in row_tile_counts:
            row = current_board[row_idx]
            seq_start = None
            
            for j in range(len(row) + 1):
                val = row[j] if j < len(row) else 0
                
                if val > 0:
                    if seq_start is None:
                        seq_start = j
                else:
                    if seq_start is not None:
                        seq_end = j - 1
                        seq_len = seq_end - seq_start + 1
                        
                        if seq_len >= 3:
                            new_board = apply_run(current_board, row_idx, seq_start, seq_end)
                            if is_valid_board_state(new_board):
                                current_board = new_board
                                moves.append(f"Run (isolated): color {row_idx}, numbers {seq_start+1}-{seq_end+1}")
                                made_progress = True
                                break
                        
                        seq_start = None
            
            if made_progress:
                break
        
        if made_progress:
            continue
        
        # 1. Find tiles that can only form runs (not groups) — use longest run
        definite = find_definite_moves(current_board)
        
        for move_type, row_idx, col_idx in definite:
            if move_type == 'only_run':
                row = current_board[row_idx]
                
                # Find all possible runs containing this tile, prefer longer ones
                possible_runs = []
                for start in range(max(0, col_idx - 12), col_idx + 1):
                    if start >= 0 and row[start] > 0:
                        end = find_maximal_run(current_board, row_idx, start)
                        if end >= col_idx and end - start >= 2:
                            possible_runs.append((start, end))
                
                # Sort by length (longest first), deduplicate
                seen = set()
                unique_runs = []
                for r in possible_runs:
                    if r not in seen:
                        seen.add(r)
                        unique_runs.append(r)
                unique_runs.sort(key=lambda x: x[1] - x[0], reverse=True)
                
                for start, end in unique_runs:
                    new_board = apply_run(current_board, row_idx, start, end)
                    if is_valid_board_state(new_board):
                        current_board = new_board
                        moves.append(f"Run (only option): color {row_idx}, numbers {start+1}-{end+1}")
                        made_progress = True
                        break
                
                if made_progress:
                    break
            
            elif move_type == 'only_group':
                column = get_column(current_board, col_idx)
                available_colors = [i for i, count in enumerate(column) if count > 0]
                
                if len(available_colors) >= 3 and row_idx in available_colors:
                    other_colors = [c for c in available_colors if c != row_idx][:2]
                    group_colors = [row_idx] + other_colors
                    
                    new_board = apply_group(current_board, col_idx, group_colors)
                    if is_valid_board_state(new_board):
                        current_board = new_board
                        moves.append(f"Group (only option): number {col_idx+1}, colors {group_colors}")
                        made_progress = True
                        break
        
        if made_progress:
            continue
        
        # 2. Try runs — most constrained row first, then longest run
        all_runs = find_all_runs(current_board)
        if all_runs:
            # Score each run: longer is better, and rows with fewer options are better
            def run_score(r):
                row_idx, start, end = r
                length = end - start
                # Count how many options tiles in this run have (fewer = more constrained = higher priority)
                total_options = 0
                for col in range(start, end + 1):
                    if can_form_group(current_board, row_idx, col):
                        total_options += 1
                    # Count how many other runs this tile could be part of
                    if can_form_run(current_board, row_idx, col):
                        total_options += 1
                # Lower options = higher priority (negate), longer = higher priority (negate)
                return (total_options, -length)
            
            all_runs.sort(key=run_score)
            
            for row_idx, start, end in all_runs:
                new_board = apply_run(current_board, row_idx, start, end)
                if is_valid_board_state(new_board):
                    current_board = new_board
                    moves.append(f"Run: color {row_idx}, numbers {start+1}-{end+1}")
                    made_progress = True
                    break
            
            if made_progress:
                continue
        
        # 3. Try groups
        groups = find_groups(current_board)
        if groups:
            col_idx, colors = groups[0]
            new_board = apply_group(current_board, col_idx, colors)
            if is_valid_board_state(new_board):
                current_board = new_board
                moves.append(f"Group: number {col_idx+1}, colors {colors}")
                made_progress = True
                continue
        
        if not made_progress:
            return False, moves, current_board
    
    return is_board_empty(current_board), moves, current_board

def worm(board):
    print("Initial board:")
    matrix_print(board)
    print()
    
    success, moves, final_board = solve_greedy(board)
    
    print(f"Made {len(moves)} moves:")
    for i, move in enumerate(moves, 1):
        print(f"  {i}. {move}")
    
    print("\nFinal board:")
    matrix_print(final_board)
    
    if success:
        print("\nBoard cleared successfully!")
    else:
        print("\nGot stuck - couldn't clear board")
    
    return success

worm(board_matrix)
