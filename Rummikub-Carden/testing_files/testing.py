import wormed

board_matrix = [[2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2]]

hand_matrix =[[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]

new_matrices = []

for i in range(len(hand_matrix)):
    for j in range(len(hand_matrix[i])):
        if hand_matrix[i][j] != 0:
            new_board = [row[:] for row in board_matrix]  # Deep copy of the board
            new_board[i][j] = hand_matrix[i][j] + board_matrix[i][j]  # Add hand tile to board
            new_matrices.append(new_board)
            if wormed.solved(new_board):
                board_matrix = new_board
                hand_matrix[i][j] = 0
            


