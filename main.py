import numpy as np
from typing import Optional
from enum import Enum

# M constant
M = 1_000_000


class State(Enum):
    SOLVED = 0
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
    solution = np.zeros((num_rows, num_cols), dtype=np.int64)
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


def Vogel(
        S: np.array,
        C: np.array,
        D: np.array) -> Result:
    remaining_rows = np.ones(C.shape[0], dtype=bool)
    remaining_cols = np.ones(C.shape[1], dtype=bool)
    x_0 = np.zeros(C.shape, dtype=np.int64)
    iteration = 0

    while True:
        iteration += 1
        mask = np.outer(remaining_rows, remaining_cols)

        if iteration > 1000:
            return Result(State.UNAPPLICABLE)

        # Finding differences
        _C = np.sort(np.where(mask, C.copy(), M*M))
        RowD = _C[:, 1] - _C[:, 0]
        _C = np.sort(np.where(mask.T, C.copy().T, M*M))
        ColD = _C[:, 1] - _C[:, 0]

        # Maximum difference
        maxD = max(np.concatenate((ColD, RowD)))

        if maxD in RowD:
            x = np.argmax(RowD, axis=0)
            y = np.argmin(np.where(mask, C.copy(), M)[x], axis=0)

            if (D[y] == 0):
                break

            if (D[y] >= S[x]):
                selected_value = S[x]
                D[y] -= selected_value
                S[x] = 0
                remaining_rows[x] = 0
            else:
                selected_value = D[y]
                S[x] -= selected_value
                D[y] = 0
                remaining_cols[y] = 0
            x_0[x][y] = selected_value
        elif maxD in ColD:
            y = np.argmax(ColD, axis=0)
            x = np.argmin(np.where(mask, C.copy(), M)[:, y], axis=0)

            if (D[y] == 0):
                break

            if (D[y] >= S[x]):
                selected_value = S[x]
                D[y] -= selected_value
                S[x] = 0
                remaining_rows[x] = 0
            else:
                selected_value = D[y]
                S[x] -= selected_value
                D[y] = 0
                remaining_cols[y] = 0
            x_0[x][y] = selected_value
    return Result(State.SOLVED, np.sum(C * x_0), x_0)


def Russell(
        S: np.array,
        C: np.array,
        D: np.array) -> Result:
    selected = np.zeros(C.shape)
    remaining_rows = np.ones(C.shape[0], dtype=bool)
    remaining_cols = np.ones(C.shape[1], dtype=bool)
    x_0 = np.zeros(C.shape, dtype=np.int64)

    it_count = 0

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

        it_count += 1
        if (it_count > 1000):
            return Result(State.UNAPPLICABLE)
    return Result(State.SOLVED, np.sum(C * x_0), x_0)


def print_problem_statement(
        S: np.array,
        C: np.array,
        D: np.array) -> None:
    S = S.astype(object)
    S = np.append(S, "_").reshape(-1, 1)
    matrix = np.append(C, [D], axis=0)
    matrix = np.hstack((matrix, S))
    matrix = matrix.astype(object)
    matrix[matrix == M] = "M"
    table = ""
    for y in range(len(matrix)):
        row = ""
        for x in range(len(matrix[0])):
            if matrix[y][x] != "_":
                if (x == len(matrix[0]) - 1):
                    row += f" |{matrix[y][x]}"
                else:
                    row += f" {matrix[y][x]}"
                if len(str(matrix[y][x])) == 1:
                    row += " "
        if (y == len(matrix)-1):
            table += "\n" + "_ " * ((len(matrix[0])-1) * 2)
        table += f"\n{row}"
    print(table)


def solve(
        S: np.array,
        C: np.array,
        D: np.array,
        NWExpected: np.array,
        VogelExpected: np.array,
        RussellExpected: np.array,
        ) -> int:

    print_problem_statement(S, C, D)

    if (np.sum(S) != np.sum(D)):
        print("The problem is not balanced!")
        return 1

    result1 = NorthwestCorner(S.copy(), C.copy(), D.copy())
    result2 = Vogel(S.copy(), C.copy(), D.copy())
    result3 = Russell(S.copy(), C.copy(), D.copy())

    if (any([result1.solved == State.UNAPPLICABLE,
            result2.solved == State.UNAPPLICABLE,
            result3.solved == State.UNAPPLICABLE])):
        print("The method is not applicable!")
        return 1
    if (not np.all(NWExpected == result1.solution)):
        print("Incorrect initial basic feasible solution for North-West.\n",
              f"Got:\n{result1.solution}.\n Expected:\n{NWExpected}.")
        return 0
    if (not np.all(VogelExpected == result2.solution)):
        print("Incorrect initial basic feasible solution for Vogel's approximation.\n",
              f"Got:\n{result2.solution}.\n Expected:\n{VogelExpected}.")
        return 0
    if (not np.all(RussellExpected == result3.solution)):
        print("Incorrect initial basic feasible solution for Russell's approximation.\n",
              f"Got:\n{result3.solution}.\nExpected:\n{RussellExpected}.")
        return 0

    print("North-West initial basic feasible solution:\n", result1.solution,
          "\nVogel's approximation intial basic feasible solution:\n", result2.solution,
          "\nRussell's approximation initial basic feasible solution:\n", result3.solution)
    return 1


def TEST_CASE_1():
    print("----------------------RUNNING_TEST_CASE_1----------------------")
    C = np.array([
        [4, 8, 6, 5],
        [3, 2, 7, 4],
        [6, 5, 3, 9],
    ], dtype=np.int64)

    S = np.array([
        150, 200, 100
    ], dtype=np.int64)

    D = np.array([
        80, 120, 100, 150
    ], dtype=np.int64)

    NWExpected = np.array([
        [80, 70, 0, 0],
        [0, 50, 100, 50],
        [0, 0, 0, 100]
    ], dtype=np.int64)

    VogelExpected = np.array([
        [80, 0, 0, 70],
        [0, 120, 0, 80],
        [0, 0, 100, 0]
    ], dtype=np.int64)

    RussellExpected = np.array([
        [0, 0, 0, 150],
        [80, 120, 0, 0],
        [0, 0, 100, 0]
    ], dtype=np.int64)

    return solve(S, C, D, NWExpected, VogelExpected, RussellExpected)


def TEST_CASE_2():
    print("----------------------RUNNING_TEST_CASE_2----------------------")
    C = np.array([[7, 3, 8, 6],
                  [4, 9, 5, 3],
                  [2, 6, 7, 4]], dtype=np.int64)

    S = np.array([180, 160, 140], dtype=np.int64)

    D = np.array([100, 110, 90, 180], dtype=np.int64)

    NWExpected = np.array([
        [100, 80, 0, 0],
        [0, 30, 90, 40],
        [0, 0, 0, 140]
    ], dtype=np.int64)

    VogelExpected = np.array([
        [0, 110, 0, 70],
        [0, 0, 90, 70],
        [100, 0, 0, 40]
    ], dtype=np.int64)

    RussellExpected = np.array([
        [0, 110, 70, 0],
        [0, 0, 20, 140],
        [100, 0, 0, 40]
    ], dtype=np.int64)

    return solve(S, C, D, NWExpected, VogelExpected, RussellExpected)


def TEST_CASE_3():
    print("----------------------RUNNING_TEST_CASE_3----------------------")
    C = np.array([
        [5, 7, 4, 8],
        [3, 6, 5, 2],
        [8, 4, 7, 3],
    ], dtype=np.int64)

    S = np.array([
        130, 170, 150
    ], dtype=np.int64)

    D = np.array([
        90, 80, 140, 140
    ], dtype=np.int64)

    NWExpected = np.array([
        [90, 40, 0, 0],
        [0, 40, 130, 0],
        [0, 0, 10, 140]
    ], dtype=np.int64)

    VogelExpected = np.array([
        [0, 0, 130, 0],
        [90, 0, 0, 80],
        [0, 80, 10, 60]
    ], dtype=np.int64)

    RussellExpected = np.array([
        [0, 0, 130, 0],
        [90, 70, 10, 0],
        [0, 10, 0, 140]
    ], dtype=np.int64)

    return solve(S, C, D, NWExpected, VogelExpected, RussellExpected)


if __name__ == "__main__":
    tests = [TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]
    tests_passed = 0
    for test in tests:
        tests_passed += test()
    print("----------------------RESULTS----------------------")
    print(f"Total number of tests: {len(tests)}")
    print(f"Total number of passed tests: {tests_passed}")
