import numpy as np
from numpy.linalg import eigh, eig, eigvalsh
import matplotlib.pyplot as plt
import time

from diatomic import build_hamiltonians, SrFConstants
Nmax = 5
H0, _, _, _ = build_hamiltonians(Nmax, SrFConstants)
H0 = H0[0:16, 0:16]

energies, states = eigh(H0)
# print(H0[0:15, 0:15])
# print("")
print(np.sort(energies/1e6))
