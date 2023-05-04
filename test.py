import numpy as np
from numpy.linalg import eigh, eig, eigvalsh
import matplotlib.pyplot as plt
import time, pandas

from diatomic import hyperfine_hamiltonian_no_field, Zeeman_hamiltonian, SrFConstants


Nmax = 5
H0 = hyperfine_hamiltonian_no_field(Nmax, SrFConstants)

energy_list = np.array([])
Bz_list = np.linspace(0, 100, 100)
for Bz in Bz_list:
    Hz = Zeeman_hamiltonian(Nmax, SrFConstants, Bfield=np.array([0, 0, Bz]))
    H = H0 + Hz
    H = H[0:16, 0:16]

    energies, states = eigh(H)
    # print(H0[0:15, 0:15])
    # print("")

    energy_list = np.append(energy_list, energies[4:16]/1e6)

energy_list -= energy_list[0]
energy_list = energy_list.reshape((-1, 12))

plt.plot(Bz_list, energy_list)

df = pandas.read_csv('Default Dataset (5).csv')
df.columns =['Bfield', 'energy']
# print(df)
df['energy'] -= 63.41595833
df['energy'] += 170.9331
plt.plot(df['Bfield'], df['energy'], 'o')



plt.grid()
plt.show()

# print(energy_list)
