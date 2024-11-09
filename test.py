from main import Vogel, M
import numpy as np

C = np.array([
    [16, 16, 13, 22, 17],
    [14, 14, 13, 19, 15],
    [19, 19, 20, 23, M],
    [M, 0, M, 0, 0]
], dtype=np.int64)

S = np.array([
    50, 60, 50, 50
], dtype=np.int64)

D = np.array([
    30, 20, 70, 30, 60
], dtype=np.int64)

print(Vogel(S, C, D).solution)
