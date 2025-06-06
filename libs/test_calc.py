import numpy as np
from scipy import linalg
import time

start = time.time()
A = np.random.randn(2048, 2048)
B = np.random.randn(2048, 2048)
C = np.load("test_matrix.npy")
# print(A.shape)
print(A.dtype)

res, _ = linalg.sqrtm(C, disp=False)

print(res)
print(f"time cost is {(time.time() - start)/ 60} min")
