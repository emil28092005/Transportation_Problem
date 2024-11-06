import numpy as np
from typing import Optional
from enum import Enum

# M constant
M = 1_000_000


class State(Enum):
    SOLVED = 0
    UNSOLVED = 1
    UNAPPLICABLE = 2


class Result:
    solved: State
    objective_function_value: Optional[np.int64]
    solution: Optional[np.array]

    def __init__(self,
                 solved: State,
                 objective_function_value: Optional[np.array] = None,
                 solution: np.int64 = None):
        self.solved = solved
        self.objective_function_value = objective_function_value
        self.solution = solution


def NorthwestCorner(
        S: np.array,
        C: np.array,
        D: np.array) -> Result:
    # TODO Northwest corner method
    pass



S = np.array([1, 2, 3, 4])

C = np.array([
    [16, 16, 13, 22, 17], 
    [14, 14, 13, 19, 15], 
    [19, 19, 20, 23, M ], 
    [M,  0,  M,  0,  0]])

D = np.array([30, 20, 70, 30, 60])


def Vogel(
        S: np.array,
        C: np.array,
        D: np.array) -> Result:
    
    C_length = len(C[0])
    C_height = len(C)
    
    #print(C_length)
    #print(C_height)
    
    RowD = np.array
    ColD = np.array
    

    RowD = np.resize(RowD, C_height)
    ColD = np.resize(ColD, C_length)
    
    #print(sorted(C[0]))
    #print(sorted(C[0])[0])
    #print(sorted(C[0])[1])
    

    #Finding differences
    for y in range(C_height):
        RowD[y] = abs(sorted(C[y])[0] - sorted(C[y])[1])
    for x in range(C_length):   
        ColD[x] = abs(sorted(C.T[x])[0] - sorted(C.T[x])[1])
    
    #print(RowD)
    #print(ColD)
    #Maximum difference
    maxD = max(np.concatenate((ColD, RowD)))
    
    
    
    #print(maxD)
    
    # TODO Vogel's method
    pass

Vogel(S,C,D)

def Russell(
        S: np.array,
        C: np.array,
        D: np.array) -> Result:
    selected = np.zeros(C.shape)
    remaining_rows = np.ones(C.shape[0], dtype=bool)
    remaining_cols = np.ones(C.shape[1], dtype=bool)
    x_0 = np.zeros(C.shape)
    while True:

        mask = np.outer(remaining_rows, remaining_cols)

        u = np.max(np.where(mask, C, -M), axis=1)
        v = np.max(np.where(mask, C, -M), axis=0)
        d = np.zeros(C.shape, dtype=np.int64)
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if selected[i][j]:
                    d[i][j] = M
                else:
                    d[i][j] = C[i][j] - u[i] - v[j]

        i, j = np.unravel_index(np.argmin(d, axis=None), d.shape)

        if (D[j] == 0):
            break

        if (S[i] >= D[j]):
            x_0[i][j] = D[j]
            S[i] -= D[j]
            D[j] = 0
            remaining_cols[j] = 0
        else:
            x_0[i][j] = S[i]
            D[j] -= S[i]
            S[i] = 0
            remaining_rows[i] = 0
        selected[i][j] = 1
    return x_0


def print_problem_statement(S, C, D) -> None:
    # TODO print table
    pass


def solve(
        S: np.array,
        C: np.array,
        D: np.array) -> int:

    print_problem_statement(S, C, D)

    if (np.sum(S) != np.sum(D)):
        print("The problem is not balanced!")
        return 1

    result1 = NorthwestCorner(S, C, D)
    result2 = Vogel(S, C, D)
    result3 = Russell(S, C, D)

    # TODO check for state (unappicable?)

    print(result1.solution, result2.solution, result3.solution)
    return 0


if __name__ == "__main__":
    C = np.array([
        [16, 16, 13, 22, 17],
        [14, 14, 13, 19, 15],
        [19, 19, 20, 23, M],
        [M, 0, M, 0, 0]
    ], dtype=np.int64)

    S = np.array([
        50, 60, 50, 50
    ])

    D = np.array([
        30, 20, 70, 30, 60
    ])

    print(Russell(S, C, D))
