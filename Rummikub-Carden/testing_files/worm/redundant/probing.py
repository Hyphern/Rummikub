import pandas as pd
from copy import deepcopy

board_matrix = [[1,1,1,0,1,2,2,1,1,2,2,2,2],
                [0,2,2,2,2,2,0,0,1,0,0,0,1],
                [0,1,1,1,1,2,2,2,2,2,2,1,1],
                [0,1,2,1,1,0,0,0,1,0,0,0,1]]

step_matrices = [[[1, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2],[0, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1],[0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1],[0, 1, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]],[[0, 0, 0, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2],[0, 2, 2, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1],[0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1],[0, 1, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]],[[0, 0, 0, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2],
[0, 2, 1, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1],
[0, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1],
[0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]],[[0, 0, 0, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2],
[0, 1, 1, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1],
[0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1],
[0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]],[[0, 0, 0, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2],
[0, 1, 1, 2, 2, 2, 0, 0, 1, 0, 0, 0, 1],
[0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
]]


def unitary_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 2:
                matrix[i][j] = 1
            else:
                matrix[i][j] = matrix[i][j]
    return matrix

def weighted_matrix(matrix):
    b_matrix = deepcopy(matrix)    
    u_matrix = unitary_matrix(b_matrix)
    w_matrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j])
            if matrix[i][j] == 0:
                continue
            else:
                tile_num = matrix[i][j]
                column_sum = sum(matrix[k][j] for k in range(len(matrix)) if k != i)
                neighbour_sum = sum(u_matrix[i][max(0, j-u_matrix[i][:j][::-1].index(0))-1:j] if 0 in u_matrix[i][:j][::-1] else u_matrix[i][:j]) + sum(matrix[i][j+1:j+1+matrix[i][j+1:].index(0)] if 0 in matrix[i][j+1:] else matrix[i][j+1:])
                w_matrix[i][j] = (column_sum,neighbour_sum,tile_num)

    return w_matrix

# hope = weighted_matrix(board_matrix)

# for row in hope:
#     print(row)
all_data = []
for step in step_matrices:
    print(f"Step: {step_matrices.index(step)+1}")
    weighted = weighted_matrix(step)
    for row in weighted:
        print(row)
    all_data.extend(weighted)

pd.DataFrame(all_data).to_csv("all_steps_weighted.csv", index=False)
