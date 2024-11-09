from main import Vogel
import numpy as np

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

print(Vogel(S, C, D).solution)
