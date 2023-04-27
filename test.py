import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j
import time

# from diatomic import build_hamiltonians, SrFConstants


# func = lambda N, Nprime: np.sqrt(2*N+1)*np.sqrt(2*Nprime+1)*wigner_3j(N, 1, Nprime, -1, 0, 1)*wigner_3j(N, 1, Nprime, 1, 0, -1)

def func (N, Nprime, mN, mNprime):
    # return np.sqrt(2*N+1)*np.sqrt(2*Nprime+1)*wigner_3j(N, 1, Nprime, -1, 0, 1)*wigner_3j(N, 1, Nprime, 1, 0, -1)
    return wigner_3j(N, 1, Nprime, -mN, 0, mNprime)

vfunc = np.vectorize(func)

Nmax=2
shapeN = np.sum([2*N+1 for N in range(Nmax+1)])
N_list = np.array([])
mN_list = np.array([])
for N in range(Nmax+1):
    # N_list = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, ...])
    # N_list = np.array([0, -1, 0, 1, -2, -1, 0, 1, 2, ...])
    N_list = np.append(N_list, [N]*(2*N+1))
    mN_list = np.append(mN_list, np.arange(-N, N+1))

column_N = np.tile(N_list, shapeN).reshape((shapeN, shapeN))
column_mN = np.tile(mN_list, shapeN).reshape((shapeN, shapeN))
row_N = np.repeat(N_list, shapeN).reshape((shapeN, shapeN))
row_mN = np.repeat(mN_list, shapeN).reshape((shapeN, shapeN))

t0 = time.time()
print(vfunc(column_N, row_N, column_mN, row_mN))
print("")
print(time.time()-t0)

# t0 = time.time()
# a = np.zeros((shapeN, shapeN))
# for i in range(shapeN):
#     for j in range(shapeN):
#         a[i, j] = func(column_N[i, j], row_N[i, j], column_mN[i, j], row_mN[i, j])
# print(a)
# print("")
# print(time.time()-t0)