import numpy as np
from numpy.linalg import eigh, eig, eigvalsh
import matplotlib.pyplot as plt
import time, pandas

from diatomic import hyperfine_hamiltonian_no_field, Stark_dc_hamiltonian, SrFConstants


Nmax = 5
H0 = hyperfine_hamiltonian_no_field(Nmax, SrFConstants)

energy_list = np.array([])
Ez_list = np.linspace(0, 100, 10)
for Ez in Ez_list:
    Hdc = Stark_dc_hamiltonian(Nmax, SrFConstants, Efield=np.array([0, 0, Ez]))
    H = H0 + Hdc

    energies, states = eigh(H)

    energy_list = np.append(energy_list, energies[0:36]/1e6)

energy_list -= energy_list[0]
energy_list = energy_list.reshape((-1, 36))

plt.plot(Ez_list, energy_list)
plt.ylim(-15e3, 66e3)

# df = pandas.read_csv('Default Dataset (5).csv')
# df.columns =['Efield', 'energy']
# # print(df)
# df['energy'] -= 63.41595833
# df['energy'] += 170.9331
# plt.plot(df['Efield'], df['energy'], 'o')


plt.grid()
plt.show()

# print(energy_list)
