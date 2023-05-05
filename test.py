import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
import time, pandas

from diatomic import hyperfine_hamiltonian_no_field, Stark_dc_hamiltonian, SrFConstants, calc_electric_dipole_moment, sort_eigenstates

Nmax = 5 # Electric field mixes rotational levels, so we need to include up to sufficiently high Nmax to converge the result
H0 = hyperfine_hamiltonian_no_field(Nmax, SrFConstants) # generate field-free hamiltonian
Hdc = Stark_dc_hamiltonian(Nmax, SrFConstants, Efield=np.array([0, 0, 1])) # generate dc Stark hamiltonian for unit electric field

Ez_list = np.linspace(0, 2, 10)
energies_list = np.empty((len(Ez_list), 16), dtype=np.complex_)
states_list = np.empty((len(Ez_list), H0.shape[0], 16), dtype=np.complex_)
for i, Ez in enumerate(Ez_list):
    H = H0 + Hdc * Ez
    energies, states = eigh(H)
    energies_list[i, :] = energies[0:16]
    states_list[i, :, 0:16] = states[:, 0:16]

energies_list, states_list = sort_eigenstates(energies_list, states_list)
dipole_moment_list = calc_electric_dipole_moment(Nmax=Nmax, states=np.hstack(states_list), consts=SrFConstants) # calc_electric_dipole_moment only takes 2D array for argument states
dipole_moment_list = dipole_moment_list[:, 2] # z component only
dipole_moment_list = dipole_moment_list.reshape((-1, 16))

fig = plt.figure(figsize=(8, 6), dpi=150)
ax1 = fig.add_subplot(1,1,1)
ax1.plot(Ez_list, dipole_moment_list[:, 0:16])
# ax1.plot(Ez_list, dipole_moment_list[:, 0:4], color="C0", label="$N=0, mN=0$")
# ax1.plot(Ez_list, dipole_moment_list[:, 4:12], color="C1", label="$N=1, mN=\pm1$")
# ax1.plot(Ez_list, dipole_moment_list[:, 12:16], color="C2", label="$N=1, mN=0$")

# handles, labels = ax1.get_legend_handles_labels()
# display = (0,4,12)

ax1.set_ylabel("Dipole moment (Debye)")
ax1.set_xlabel("Electric Field (kV/cm)")
ax1.set_title("Induced lab-frame dipole moment of SrF molecule ($X^2\Sigma,\ v=0$)")
ax1.grid(True)
# ax1.legend([handle for i,handle in enumerate(handles) if i in display],
#             [label for i,label in enumerate(labels) if i in display], loc='upper left')

ax2 = ax1.twinx()
ymin, ymax = ax1.get_ylim()
ax2.set_ylim(ymin/SrFConstants.DipoleMoment_d, ymax/SrFConstants.DipoleMoment_d)
ax2.set_ylabel('Ratio to body-frame dipole moment')

plt.show()