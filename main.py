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


def NorthwestCorner(S: np.array, 
                    C: np.array, 
                    D: np.array) -> Result:
    num_rows, num_cols = len(S), len(D)
    solution = [[0] * num_cols for _ in range(num_rows)]
    
    i, j = 0, 0
    while i < num_rows and j < num_cols:
        quantity = min(S[i], D[j])
        solution[i][j] = quantity
        S[i] -= quantity
        D[j] -= quantity
        
        if S[i] == 0:
            i += 1
        elif D[j] == 0:
            j += 1

    if sum(S) == 0 and sum(D) == 0:
        objective_function_value = np.sum(solution * C)
        return Result(State.SOLVED, objective_function_value, solution)
    else:
        return Result(State.UNAPPLICABLE)



S = np.array([50, 60, 50, 50])

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
    
    print(C)
    iteration = 0
    while(len(C[0]) > 1 and  len(C) > 1):
        iteration += 1
        C_map = dict()
        
        for y in range(len(C)):
            for x in range(len(C[0])):
                C_map[x,y] = C[y][x]
        
        C_length = len(C[0])
        C_height = len(C)
        
        print(f"C_length: {C_length}")
        print(f"C_height: {C_height}")
        
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
        
        print(f"S: {S}")
        print(f"D: {D}")
        
        print(f"RowD: {RowD}")
        print(f"ColD: {ColD}")
        #Maximum difference
        maxD = max(np.concatenate((ColD, RowD)))
        
        
        target_array = np.array([])
        target_number = None
        row_index_to_eleminate = None
        column_index_to_eleminate = None
        
        print(f"maxD: {maxD}")
        
        '''if maxD in RowD:
            target_array = C[np.where(RowD == maxD)[0]][0]
            target_number = min(target_array)
            row_index_to_eleminate = np.where(target_array == target_number)[0][0]
            print(f"row {row_index_to_eleminate}")
            C = np.delete(C, row_index_to_eleminate, 0)
            S = np.delete(S, row_index_to_eleminate, 0)'''
        
        if maxD in RowD:
            print("in RowD")
            y = np.where(RowD == maxD)[0][0]
            target_array = C[np.where(RowD == maxD)[0]][0]
            target_number = min(target_array)
            print(f"target_number: {target_number}")
            x = np.where(target_array == target_number)[0][0]
            print(f"(x,y): {(x,y)}")
            print(f"D[x]:{D[x]}, S[y]:{S[y]}")
                
            if (D[x] >= S[y]): #TODO ?
                print("D[x] > S[y]")
                row_index_to_eleminate = y
                print(f"row_index_to_eleminate {row_index_to_eleminate}")
                selected_value = S[y]
                
                D[x] -= selected_value 
                
                C = np.delete(C, row_index_to_eleminate, 0)
                S = np.delete(S, row_index_to_eleminate, 0)
                
            else:
                print("D[x] <= S[y]")
                column_index_to_eleminate = x
                selected_value = D[x]
                
                S[y] -= selected_value 
                
                C = np.delete(C, column_index_to_eleminate, 1)
                D = np.delete(D, column_index_to_eleminate, 0)
        
        if maxD in ColD:
            print("in ColD")
            x = np.where(ColD == maxD)[0][0]
            target_array = C.T[np.where(ColD == maxD)[0]][0]
            target_number = min(target_array)
            print(f"target_number: {target_number}")
            y = np.where(target_array == target_number)[0][0]
            print(f"(x,y): {(x,y)}")
            print(f"D[x]:{D[x]}, S[y]:{S[y]}")
            
            if (D[x] >= S[y]): #TODO ?
                print("D[x] > S[y]")
                row_index_to_eleminate = y
                selected_value = S[y]
                
                D[x] -= selected_value 
                
                C = np.delete(C, row_index_to_eleminate, 0)
                S = np.delete(S, row_index_to_eleminate, 0)
                
            else:
                print("D[x] <= S[y]")
                column_index_to_eleminate = x
                selected_value = D[x]
                
                S[y] -= selected_value 
                
                C = np.delete(C, column_index_to_eleminate, 1)
                D = np.delete(D, column_index_to_eleminate, 0)
        print(f"target_array: {target_array}")
        print(f"selected_value: {selected_value}")
        
        
        
        print(C)
        
        #print(target_array)
        #print(maxD)
    
    
    
    
    
    # TODO Vogel's method
    pass

#Vogel(S,C,D)

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


'''if __name__ == "__main__":
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
'''
