import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import time

from diatomic import build_hamiltonians, SrFConstants

Nmax = 10
H0, _, _, _ = build_hamiltonians(Nmax, SrFConstants)
energies, states = eigh(H0)
# print(H0)
# print("")
print(energies[0:15]/1e6)