import numpy as np
from .constants import MolecularConstants
from .hamiltonians import _generate_vecs

__all__ = ['calc_electric_dipole_moment', 'sort_eigenstates']

def calc_electric_dipole_moment(Nmax: int, states: np.ndarray, consts: MolecularConstants) -> np.ndarray:
    '''
    Returns the induced lab-frame electric dipole moments of each eigenstate
    
    Args:
        states (np.ndarray): matrix for eigenstates of problem output from np.linalg.eigh, states[:, i] is the i-th normalized eigenstate
        Nmax (int): Maximum rotational quantum number in original calculations
        consts (MolecularConstants): Dictionary of constants for the molecular to be calculated
        
    Returns:
        d (np.ndarray): dipole moment, in order of [[d1x, d1y, d1z], [d2x, d2y, d2z], ...]
    '''

    dim = int((Nmax+1)**2*(2*consts.ElectronSpin_S+1)*(2*consts.NuclearSpin_I+1))
    assert states.shape[0] == dim
    assert len(states.shape) == 2
    
    dipole_moment_list = np.array([])
    N_vec, S_vec, I_vec, n_vec= _generate_vecs(Nmax=Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I)
    for i in range(states.shape[1]):
        dx = consts.DipoleMoment_d * np.matmul(np.conjugate(states[:, i]), np.matmul(n_vec[0], states[:, i].reshape((-1, 1))))
        dy = consts.DipoleMoment_d * np.matmul(np.conjugate(states[:, i]), np.matmul(n_vec[1], states[:, i].reshape((-1, 1))))
        dz = consts.DipoleMoment_d * np.matmul(np.conjugate(states[:, i]), np.matmul(n_vec[2], states[:, i].reshape((-1, 1))))

        dipole_moment_list = np.append(dipole_moment_list, np.real(np.array([dx, dy, dz])))

    return dipole_moment_list.reshape((-1, 3))


def sort_eigenstates(energies: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ''' 
    Sort states to remove false avoided crossings.

    This is a function to ensure that all eigenstates plotted change
    adiabatically, it does this by assuming that step to step the eigenstates
    should vary by only a small amount (i.e. that the  step size is fine) and
    arranging states to maximise the overlap one step to the next.

    Args:
        energies (np.ndarray): 2D np array containing the eigenenergies, each row is a 1D array of eigenenergies under one condition
        states (np.ndarray): 3D np array containing the states, states[x, :, i] corresponds to energies[x, i]
    Returns:
        energies (np.ndarray): 2D np array containing the eigenenergies, each row is a 1D array of eigenenergies under one condition but the order in each row is sorted
        states (np.ndarray): 3D np array containing the states, in the same order as energies E[x, j] -> States[x, :, j]
    '''

    assert len(energies.shape) == 2 # assert energies is 1D
    assert len(states.shape) == 3 # assert states is 2D
    assert energies.shape[0] == states.shape[0] # assert the number of energies is the same as number of states
    assert energies.shape[1] == states.shape[2] # assert the number of energies is the same as number of states

    ls = np.arange(states.shape[2], dtype="int")
    number_iterations = energies.shape[0]

    # This loop sorts the eigenstates such that they maintain some continuity. 
    # Each eigenstate should be chosen to maximise the overlap with the previous.
    for i in range(1, number_iterations):

        #calculate the overlap of the ith and jth eigenstates
        overlaps = np.einsum('ij,ik->jk', np.conjugate(states[i-1, :, :]), states[i, :, :])
        orig_states = states[i, :, :].copy()
        orig_energies = energies[i, :].copy()

        #insert location of maximums into array ls
        np.argmax(np.abs(overlaps), axis=1, out=ls)

        print(np.unique(ls).shape)
        print(ls)
        assert np.unique(ls).shape == ls.shape # assert there's no duplicates in ls

        # assert no overlap in ls numbers

        for k, l in enumerate(ls):
            if l!=k:
                energies[i, k] = orig_energies[l].copy()
                states[i, :, k] = orig_states[:, l].copy()

    return (energies, states)
