from main import Vogel
import numpy as np

C = np.array([
    [7, 8, 1, 2],
    [4, 5, 9, 8],
    [9, 2, 3, 6],
], dtype=np.int64)

S = np.array([
    160, 140, 170
], dtype=np.int64)

D = np.array([
    120, 50, 190, 110
], dtype=np.int64)

VogelExpected = np.array([
    [0, 0, 50, 110],
    [120, 20, 0, 0],
    [0, 30, 140, 0],
], dtype=np.int64)

print(Vogel(S, C, D).solution)
