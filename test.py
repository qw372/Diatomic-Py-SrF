import numpy as np
from numpy.linalg import eigh, eig, eigvalsh
import matplotlib.pyplot as plt
import time

from diatomic import hyperfine_hamiltonian_no_field, Zeeman_hamiltonian, SrFConstants


Nmax = 5
H0 = hyperfine_hamiltonian_no_field(Nmax, SrFConstants)

for Bz in np.linspace(0, 10, 10):
    Hz = Zeeman_hamiltonian(Nmax, SrFConstants, Bfield=np.array([0, 0, Bz]))
    H = H0 + Hz
    H = H[0:16, 0:16]

    energies, states = eigh(H)
    # print(H0[0:15, 0:15])
    # print("")
    print(energies[3]/1e6)
