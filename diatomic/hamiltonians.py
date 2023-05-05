import numpy as np
from scipy.linalg import block_diag
from sympy.physics.wigner import wigner_3j
from .constants import MolecularConstants

'''
This module contains the main code to calculate the hyperfine structure of 
molecules in external electric and magnetic fields. 

Here we use uncoupled basis |N, mN>|S, mS>|I, mI>, and mJ (J is N, S, or I) is in descending order of J, J-1, ..., -J+1, -J.

Energies in the Hamiltonian have units of Hz (defined as E/h). 

Example:
    Basic usage of this module is for accessing the eigenstates and
    eigenvalues of the molecule in question. This is most easily done
    by combining this module with the uses favourite linear algebra module.
    For instance to find the zero-field hyperfine states of Molecule::

        $ from diatom import Hamiltonian
        $ from numpy import linalg as la
        $ H0,Hz,HDC,HAC = Hamiltonian.Build_Hamiltonians(5,Molecule)
        $ ev,es = la.eigh(H0)
'''

__all__ = ['hyperfine_hamiltonian_no_field', 'Zeeman_hamiltonian', 'Stark_dc_hamiltonian', 'Stark_ac_hamiltonian']

def _raising_operator(J: float) -> np.ndarray:
    ''' 
    Creates the matrix representation of angular momentum raising operator for J, in |J, mJ> basis.
    Note that this is different from spherical tensor operator J_{+1}

    Args:
        J (float) : value of the angular momentum

    Returns:
        J+ (np.ndarray) : Array representing the operator J+, has shape ((2J+1),(2J+1))

    '''

    assert float(2*J+1).is_integer()
    assert J >= 0

    mJ_list = np.arange(-J, J)
    elements = np.sqrt(J*(J+1)-mJ_list*(mJ_list+1)) # assume the basis |J, mJ> is in order J, J-1, ..., -J+1, -J
    J_plus = np.diag(elements, 1)

    return J_plus

def _x_operator(J: float) -> np.ndarray:
    ''' 
    Creates the Cartesian operator Jx for a given J (x component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jx (np.ndarray) : 2J+1 square numpy array
    '''

    J_plus = _raising_operator(J)

    return 0.5 * (J_plus + J_plus.T) # J_plus.T is lowering operator J_minus

def _y_operator(J: float) -> np.ndarray:
    ''' 
    Creates the Cartesian operator Jy for a given J (y component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jy (np.ndarray) : 2J+1 square numpy array
    '''

    J_plus = _raising_operator(J)

    return 0.5j * (J_plus.T - J_plus) # J_plus.T is lowering operator J_minus

def _z_operator(J: float) -> np.ndarray:
    ''' 
    Creates the Cartesian operator Jz for a given J (z component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jz (np.ndarray) : 2J+1 square numpy array
    '''

    assert float(2*J+1).is_integer()
    assert J >= 0

    return np.diag(np.arange(J, -J-1, -1))

def _generate_vecs(Nmax: int, S: float, I: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' 
    Build N, S, I angular momentum vectors

    Generate the vectors of the angular momentum operators which we need
    to produce the Hamiltonian

    Args:
        Nmax (int): maximum rotational level to include in calculations
        S (float): electronic spin
        I (float): Nuclear spin, assume only one nucleus has non-zero spin
    Returns:
        N_vec, S_vec, I_vec, n_vec (np.ndarray): length-3 list of (2Nmax+1)*(2S+1)*(2I+1) square numpy arrays
    '''

    assert isinstance(Nmax, int)
    assert Nmax >= 0

    assert (2*S+1).is_integer()
    assert S > 0
    assert (2*I+1).is_integer()
    assert I > 0

    shapeN = int(np.sum([2*x+1 for x in range(Nmax+1)]))
    shapeS = int(2*S+1)
    shapeI = int(2*I+1)

    Nx = np.array([[]])
    Ny = np.array([[]])
    Nz = np.array([[]])

    for N in range(Nmax+1):
        Nx = block_diag(Nx, _x_operator(N))
        Ny = block_diag(Ny, _y_operator(N))
        Nz = block_diag(Nz, _z_operator(N))

    # remove the first element of the N vectors, which is empty
    Nx = Nx[1:,:]
    Ny = Ny[1:,:]
    Nz = Nz[1:,:]

    # Each of the following corresponds to the product [N x S x I]
    # This gives the operators for N in the full hyperfine space.

    # numpy.kron is the function for the Kronecker product, often also called
    # the tensor product.

    N_vec = np.array([np.kron(Nx, np.kron(np.identity(shapeS), np.identity(shapeI))),
                        np.kron(Ny, np.kron(np.identity(shapeS), np.identity(shapeI))),
                        np.kron(Nz, np.kron(np.identity(shapeS), np.identity(shapeI)))])

    # we also have to repeat for the electronic and nuclear spins
    S_vec = np.array([np.kron(np.identity(shapeN), np.kron(_x_operator(S), np.identity(shapeI))),
                        np.kron(np.identity(shapeN), np.kron(_y_operator(S),np.identity(shapeI))),
                        np.kron(np.identity(shapeN), np.kron(_z_operator(S),np.identity(shapeI)))])

    I_vec = np.array([np.kron(np.identity(shapeN), np.kron(np.identity(shapeS),_x_operator(I))),
                        np.kron(np.identity(shapeN), np.kron(np.identity(shapeS),_y_operator(I))),
                        np.kron(np.identity(shapeN), np.kron(np.identity(shapeS),_z_operator(I)))])
    
    # Below we'll calculate matrix representatoin of vector n, the unit vector represents the orientation of internuclear axis, under |N, mN> basis
    N_list = np.array([])
    mN_list = np.array([])
    for N in range(Nmax+1):
        # N_list = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, ...])
        # mN_list = np.array([0, 1, 0, -1, 2, 1, 0, -1, -2, ...])
        N_list = np.append(N_list, [N]*(2*N+1))
        mN_list = np.append(mN_list, np.arange(N, -N-1, -1))

    # We first define functions to calculate matrix presentation of Wigner matrix D^{1*}_{p0}, for p = -1, 0, 1
    D_func = lambda N, mN, Nprime, mNprime, p: float(np.sqrt(2*N+1)*np.sqrt(2*Nprime+1)*((-1)**mN)*wigner_3j(N, 1, Nprime, -mN, p, mNprime)*wigner_3j(N, 1, Nprime, 0, 0, 0))
    D_minus_1_func = lambda N, mN, Nprime, mNprime: D_func(N, mN, Nprime, mNprime, -1)
    D_0_func = lambda N, mN, Nprime, mNprime: D_func(N, mN, Nprime, mNprime, 0)
    D_plus_1_func = lambda N, mN, Nprime, mNprime: D_func(N, mN, Nprime, mNprime, 1)

    # # Vectorized method to calculate matrix representation of vector n
    # # Given small Nmax here (usually < 10), it's not significant faster than for loop
    # column_N = np.tile(N_list, shapeN).reshape((shapeN, shapeN))
    # column_mN = np.tile(mN_list, shapeN).reshape((shapeN, shapeN))
    # row_N = np.repeat(N_list, shapeN).reshape((shapeN, shapeN))
    # row_mN = np.repeat(mN_list, shapeN).reshape((shapeN, shapeN))

    # D_minus_1_vfunc = np.vectorize(D_minus_1_func)
    # D_0_vfunc = np.vectorize(D_0_func)
    # D_plus_1_vfunc = np.vectorize(D_plus_1_func)

    # D_minus_1 = D_minus_1_vfunc(row_N, row_mN, column_N, column_mN)
    # D_0 = D_0_vfunc(row_N, row_mN, column_N, column_mN)
    # D_plus_1 = D_plus_1_vfunc(row_N, row_mN, column_N, column_mN)

    # for loop method to calculate matrix representation of vector n
    D_minus_1 = np.zeros((shapeN, shapeN))
    D_0 = np.zeros((shapeN, shapeN))
    D_plus_1 = np.zeros((shapeN, shapeN))
    for i in range(shapeN):
        for j in range(shapeN):
            D_minus_1[i, j] = D_minus_1_func(N_list[i], mN_list[i], N_list[j], mN_list[j])
            D_0[i, j] = D_0_func(N_list[i], mN_list[i], N_list[j], mN_list[j])
            D_plus_1[i, j] = D_plus_1_func(N_list[i], mN_list[i], N_list[j], mN_list[j])

    # matrix representation of vector n under |N, mN> subspace
    nx = (D_minus_1 - D_plus_1) / np.sqrt(2)
    ny = (D_minus_1 + D_plus_1) / np.sqrt(2) * 1j
    nz = D_0

    # matrix representation of vector n under |N, mN, S, mS, I, mI> full space
    n_vec = np.array([np.kron(nx, np.kron(np.identity(shapeS), np.identity(shapeI))),
                        np.kron(ny, np.kron(np.identity(shapeS), np.identity(shapeI))),
                        np.kron(nz, np.kron(np.identity(shapeS), np.identity(shapeI)))])

    return (N_vec, S_vec, I_vec, n_vec)

def _generate_coefficients(Nmax: int, S: float, I: float, DunhamSeries: np.ndarray) -> np.ndarray:
    ''' 
    Coupling coefficients (spin-rotational, hyperfine, etc) generally have rovibrational state (v, N) dependence. Such dependence is described by Dunham model. 
    Here we generate a matrix (in |N, mN>|S, mS>|I, mI> uncoupled basis) whose diagonal terms represent coupling coefficiets for each rovibrational level. 
    This matrix can be multiplied by vecs to generate hamiltonians.

    Args:
        Nmax (int): maximum rotational level to include in calculations
        S, I (float): electronic and nuclear spin, assuming only one nucleus has non-zero spin
        DunhamSeries (np.ndarray): Dunham series (see John Barry's series scetion 2.4, 2.5 for details) in order of [[X_00, X_01, X_02, ...], [X_10, X_11, X_12, ...], ...]

    Returns:
        Dunham_matrix (np.ndarray): see above 
    '''

    assert isinstance(Nmax, int)
    assert Nmax >= 0

    assert (2*S+1).is_integer()
    assert S > 0
    assert (2*I+1).is_integer()
    assert I > 0

    assert isinstance(DunhamSeries, np.ndarray)
    assert len(DunhamSeries.shape) == 2 # make sure it's a 2D array

    shapeS = int(2*S+1)
    shapeI = int(2*I+1)

    v = 0 # here we only consider v=0 vibrational levels

    Dunham_matrix = np.array([[]]) 

    for N in range(Nmax+1):
        X_sum = 0
        for l, X_list in enumerate(DunhamSeries):
            for j, X in enumerate(X_list):
                X_sum += X*((v+1/2)**l)*((N*(N+1))**j)

        Dunham_matrix = block_diag(Dunham_matrix, np.identity(2*N+1)*X_sum)

    Dunham_matrix = Dunham_matrix[1:,:] # remove the first element of the N vectors, which is empty

    Dunham_matrix = np.kron(Dunham_matrix, np.kron(np.identity(shapeS), np.identity(shapeI)))

    return Dunham_matrix

# From here the functions will calculate individual terms in the Hamiltonian

def _rotational(B: np.ndarray) -> np.ndarray:
    ''' 
    Generates the hyperfine-free hamiltonian for the rotational levels of molecules.

    Args:
        B (np.ndarray): Rotational constant coefficient, return of _generate_coefficients()

    Returns:
        H (np.ndarray): hamiltonian for rotation in the N,MN basis
    '''

    B -= B[0][0]*np.identity(B.shape[0]) # remove energy offset so N=0 state have energy zero

    return B

def _spin_rotational_coupling(gamma: np.ndarray, S_vec: np.ndarray, N_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the spin-rotational coupling term

    Args:
        gamma (np.ndarray): spin-rotational coupling coefficient, return of _generate_coefficients()
        S_vec, N_vec (np.ndarray): Angular momentum vectors

    Returns:
        H (np.ndarray): Hamiltonian for spin-spin interaction
    '''

    return np.matmul(gamma, np.matmul(S_vec, N_vec).sum(axis=0))

def _hyperfine(b: np.ndarray, I_vec: np.ndarray, S_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the hyperfine interaction term

    Args:
        b (np.ndarray): hyperfine coefficient, return of _generate_coefficients()
        I_vec, S_vec (np.ndarray): Angular momentum vectors

    Returns:
        H (np.ndarray): Hamiltonian for spin-spin interaction
    '''

    return np.matmul(b, np.matmul(I_vec, S_vec).sum(axis=0))

def _spin_dipole_dipole_coupling(c: np.ndarray, I_vec: np.ndarray, S_vec: np.ndarray, n_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the spin dipoile-dipole coupling term

    Args:
        c (np.ndarray): spin-rotational coupling coefficient, return of _generate_coefficients()
        I_vec, S_vec, n_vec (np.ndarray): Angular momentum vectors

    Returns:
        H (np.ndarray): Hamiltonian for spin-spin interaction
    '''

    return np.matmul(c, np.matmul(np.matmul(S_vec, n_vec).sum(axis=0), np.matmul(I_vec, n_vec).sum(axis=0)))

def _nuclear_spin_rotational_coupling(C: np.ndarray, I_vec: np.ndarray, N_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the nuclear spin-rotational coupling term

    Args:
        C (np.ndarray): nuclear spin-rotational coupling coefficient, return of _generate_coefficients()
        I_vec, N_vec (np.ndarray): Angular momentum vectors

    Returns:
        H (np.ndarray): Hamiltonian for spin-spin interaction
    '''

    return np.matmul(C, np.matmul(I_vec, N_vec).sum(axis=0))

def hyperfine_hamiltonian_no_field(Nmax: int, consts: MolecularConstants) -> np.ndarray:
    '''
    Calculate the field-free Hyperfine hamiltonian

    Args:
        Nmax (int) - Maximum rotational level to include
        Consts (MolecularConstants): class of molecular constants
    Returns:
        H : Hamiltonian for the hyperfine structure
    '''

    assert isinstance(Nmax, int)
    assert Nmax >= 0

    assert (2*consts.ElectronSpin_S+1).is_integer()
    assert consts.ElectronSpin_S > 0
    assert (2*consts.NuclearSpin_I+1).is_integer()
    assert consts.NuclearSpin_I > 0

    dim = int((Nmax+1)**2*(2*consts.ElectronSpin_S+1)*(2*consts.NuclearSpin_I+1))

    # to calculate spin dipole-dipole coupling, we use <N_i|n^2|N_j > = \sum_{N_k} <N_i|n|N_k><N_k|n|N_j>. And to be accurate, we need to include up to Nk = max{Ni, Nj}+1
    N_vec_1, S_vec_1, I_vec_1, n_vec_1= _generate_vecs(Nmax=Nmax+1, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I)
    N_vec = N_vec_1[:, 0:dim, 0:dim]
    S_vec = S_vec_1[:, 0:dim, 0:dim]
    I_vec = I_vec_1[:, 0:dim, 0:dim]

    B_matrix = _generate_coefficients(Nmax=Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I, DunhamSeries=consts.RotationalConstant_B)
    gamma_matrix = _generate_coefficients(Nmax=Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I, DunhamSeries=consts.SpinRotationalCoupling_gamma)
    b_matrix = _generate_coefficients(Nmax=Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I, DunhamSeries=consts.HyperfineCoupling_b)
    c_matrix = _generate_coefficients(Nmax=Nmax+1, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I, DunhamSeries=consts.DipoleDipoleCoupling_c)
    C_matrix = _generate_coefficients(Nmax=Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I, DunhamSeries=consts.NuclearSpinRotationalCoupling_C)

    H = _rotational(B=B_matrix) + \
        _spin_rotational_coupling(gamma=gamma_matrix, S_vec=S_vec, N_vec=N_vec) + \
        _hyperfine(b=b_matrix, I_vec=I_vec, S_vec=S_vec) + \
        _spin_dipole_dipole_coupling(c=c_matrix, I_vec=I_vec_1, S_vec=S_vec_1, n_vec=n_vec_1)[0:dim, 0:dim] + \
        _nuclear_spin_rotational_coupling(C=C_matrix, I_vec=I_vec, N_vec=N_vec)
    
    return H

def Zeeman_hamiltonian(Nmax: int, consts: MolecularConstants, Bfield: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    '''
    Calculate the Zeeman shift hamiltonian
    See John Barry's thesis Eq. 2.20 for details

    TODO: John's thesis Eq. 2.20 is an incomplete version of the Zeeman shift, need to add more terms. 
    Reference:
        1. J. Mol. Spectrosc. 317 (2015) 1-9
        2. arXiv:2302.14687v1
        3. PhysRevResearch.2.013251
        4. Eric's thesis
        5. Brown & Carrington

    Args:
        Nmax (int) - Maximum rotational level to include
        Consts (MolecularConstants): class of molecular constants
        Bfield (np.ndarray): Magnetic field vector in unit of Gauss
    Returns:
        H : Hamiltonian for Zeeman shift
    '''

    assert np.shape(Bfield) == (3,) # Bfield must be a 1D vector of length 3

    N_vec, S_vec, I_vec, n_vec= _generate_vecs(Nmax=Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I)

    Hz = - consts.EletronGFactor_gs * consts.BohrMagneton_muB * np.sum([Bfield[i]*S_vec[i] for i in range(len(Bfield))], axis=0) # Electron spin Zeeman effect
    Hz += 0 # electron orbital Zeeman shift, for sigma state molecule it's zero
    Hz += - consts.NuclearGFactor_gI * consts.NuclearMagneton_muN * np.sum([Bfield[i]*I_vec[i] for i in range(len(Bfield))], axis=0) # Nuclear Zeeman effect
  
    return Hz

def Stark_dc_hamiltonian(Nmax: int, consts: MolecularConstants, Efield: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
    '''
    Calculate the dc Stark shift hamiltonian -\vec{E}\cdot\vec{d}, where \vec{d} is dipole moment operator.
    \vec{d} is can also be written as \vec{d} = d\vec{n}, where \vec{n} is the unit vector of diatomic molecule internuclear axis and d is the molecule-frame dipole moment magnitude.

    Args:
        Nmax (int) - Maximum rotational level to include
        Consts (MolecularConstants): class of molecular constants
        Efield (np.ndarray): Electric field vector in unit of kV/cm
    Returns:
        H : Hamiltonian for dc Stark shift
    '''

    assert np.shape(Efield) == (3,) # Bfield must be a 1D vector of length 3

    N_vec, S_vec, I_vec, n_vec= _generate_vecs(Nmax=Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I)

    Hdc = - consts.DebyeToHzcmperkV * consts.DipoleMoment_d * np.sum([Efield[i]*n_vec[i] for i in range(len(Efield))], axis=0) # in unit of Hz

    return Hdc

def Stark_ac_hamiltonian():
    pass
