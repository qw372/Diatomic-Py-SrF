import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import time

from diatomic import build_hamiltonians, SrFConstants


Nmax = 0
H0, Hz, Hdc, Hac = build_hamiltonians(Nmax, SrFConstants)
print(H0)

energies, states = eigh(H0)
print(energies/1e6)