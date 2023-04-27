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

Nmax=1
shapeN = int(np.sum([2*x+1 for x in range(Nmax+1)]))
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

D_func = lambda N, mN, Nprime, mNprime, p: np.sqrt(2*N+1)*np.sqrt(2*Nprime+1)*((-1)**mN)*wigner_3j(N, 1, Nprime, -mN, p, mNprime)*wigner_3j(N, 1, Nprime, 0, 0, 0)
D_minus_1_func = lambda N, mN, Nprime, mNprime: D_func(N, mN, Nprime, mNprime, -1)
D_0_func = lambda N, mN, Nprime, mNprime: D_func(N, mN, Nprime, mNprime, 0)
D_plus_1_func = lambda N, mN, Nprime, mNprime: D_func(N, mN, Nprime, mNprime, 1)

D_minus_1_vfunc = np.vectorize(D_minus_1_func)
D_0_vfunc = np.vectorize(D_0_func)
D_plus_1_vfunc = np.vectorize(D_plus_1_func)

D_minus_1 = D_minus_1_vfunc(row_N, row_mN, column_N, column_mN)
D_0 = D_0_vfunc(row_N, row_mN, column_N, column_mN)
D_plus_1 = D_plus_1_vfunc(row_N, row_mN, column_N, column_mN)

print(D_minus_1)
print(D_0)
print(D_plus_1)


D_minus_1 = np.zeros((shapeN, shapeN))
D_0 = np.zeros((shapeN, shapeN))
D_plus_1 = np.zeros((shapeN, shapeN))
for i in range(shapeN):
    for j in range(shapeN):
        D_minus_1[i, j] = D_minus_1_func(N_list[i], mN_list[i], N_list[j], mN_list[j])
        D_0[i, j] = D_0_func(N_list[i], mN_list[i], N_list[j], mN_list[j])
        D_plus_1[i, j] = D_plus_1_func(N_list[i], mN_list[i], N_list[j], mN_list[j])
        
print(D_minus_1)
print(D_0)
print(D_plus_1)