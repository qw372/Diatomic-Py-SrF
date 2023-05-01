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

__all__ = ['build_hamiltonians']

def _raising_operator(J: float) -> np.ndarray:
    ''' 
    Creates the matrix representation of angular momentum raising operator for J, in |J, mJ> basis.
    Note that this is different from spherical tensor operator J_{+1}

    Args:
        J (float) : value of the angular momentum

    Returns:
        J+ (numpy.ndarray) : Array representing the operator J+, has shape ((2J+1),(2J+1))

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
        Jx (numpy.ndarray) : 2J+1 square numpy array
    '''

    J_plus = _raising_operator(J)

    return 0.5 * (J_plus + J_plus.T) # J_plus.T is lowering operator J_minus

def _y_operator(J: float) -> np.ndarray:
    ''' 
    Creates the Cartesian operator Jy for a given J (y component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jy (numpy.ndarray) : 2J+1 square numpy array
    '''

    J_plus = _raising_operator(J)

    return 0.5j * (J_plus.T - J_plus) # J_plus.T is lowering operator J_minus

def _z_operator(J: float) -> np.ndarray:
    ''' 
    Creates the Cartesian operator Jz for a given J (z component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jz (numpy.ndarray) : 2J+1 square numpy array
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
        Nmax (float): maximum rotational level to include in calculations
        S (float): electronic spin
        I (float): Nuclear spin, assume only one nucleus has non-zero spin
    Returns:
        N_vec, S_vec, I_vec, n_vec (numpy.ndarray): length-3 list of (2Nmax+1)*(2S+1)*(2I+1) square numpy arrays
    '''

    assert isinstance(Nmax, int)
    assert Nmax >=0

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

# From here the functions will calculate individual terms in the Hamiltonian

def _rotational(Nmax: int, Brot: np.ndarray, S: float, I: float) -> np.ndarray:
    ''' 
    Rotational structure

    Generates the hyperfine-free hamiltonian for the rotational levels of
    molecules, including higher-order distortion terms (Dunham series, see John Barry thesis section 2.4 for details).

    Matrix is returned in the N,mN basis.

    Args:
        Nmax: max rotational levels to include 
        Brot (np.ndarray) - Rotational constant coefficient expressed in Dunham series in order of [[Y_00, Y_01, Y_02, ...], [Y_10, Y_11, Y_12, ...], ...]
        S, I: electronic and nuclear spin

    Returns:
        Hrot (numpy.ndarray) - hamiltonian for rotation in the N,MN basis
    '''

    assert isinstance(Nmax, int)
    assert Nmax >=0
    assert isinstance(Brot, np.ndarray)
    assert len(Brot.shape) == 2 # assert Brot is a 2D array

    Hrot = np.array([[]])
    v = 0 # here we only consider v=0 vibrational levels

    for N in range(Nmax+1):
        Y_sum = 0
        for l, Y_list in enumerate(Brot):
            for m, Y in enumerate(Y_list):
                Y_sum += ((v+1/2)**l)*((N*(N+1))**m)

        Hrot = block_diag(Hrot, np.identity(2*N+1)*Y_sum)

    # remove the first element of the N vectors, which is empty
    Hrot = Hrot[1:,:]

    return Hrot

def _spin_rotational_coupling(gamma: float, S_vec: np.ndarray, N_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the spin-rotational coupling term

    Args:
        gamma (float) - spin-rotational coupling coefficient
        S_vec, N_vec (numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return gamma*np.matmul(S_vec, N_vec).sum(axis=0)

def _hyperfine(b: float, I_vec: np.ndarray, S_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the hyperfine term

    Args:
        b (float) - hyperfine coefficient
        I_vec, S_vec (numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return b*np.matmul(I_vec, S_vec).sum(axis=0)

def _spin_dipole_dipole_coupling(c: float, I_vec: np.ndarray, S_vec: np.ndarray, n_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the spin dipoile-dipole coupling term

    Args:
        gamma (float) - spin-rotational coupling coefficient
        I_vec, S_vec, n_vec (numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return c*np.matmul(np.matmul(S_vec, n_vec).sum(axis=0), np.matmul(I_vec, n_vec).sum(axis=0)) 

def _nuclear_spin_rotational_coupling(C: float, I_vec: np.ndarray, N_vec: np.ndarray) -> np.ndarray:
    ''' 
    Calculate the nuclear spin-rotational coupling term

    Args:
        C (float) - nuclear spin-rotational coupling coefficient
        I_vec, N_vec (numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return C*np.matmul(I_vec, N_vec).sum(axis=0)

def _hamiltonian_no_field(Nmax: int, consts: MolecularConstants) -> np.ndarray:
    '''
    Calculate the field-free Hyperfine hamiltonian

    Args:
        Nmax (int) - Maximum rotational level to include
        S, I (float) - electronic and nuclear spins
        Consts (MolecularConstants): class of molecular constants
    Returns:
        H : Hamiltonian for the hyperfine structure
    '''

    N_vec, S_vec, I_vec, n_vec= _generate_vecs(Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I)
    H = _rotational(Nmax=Nmax, Brot=consts.RotationalConstant_B, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I) + \
        _spin_rotational_coupling(consts.SpinRotationalCoupling_gamma, S_vec=S_vec, N_vec=N_vec) + \
        _hyperfine(consts.HyperfineCoupling_b, I_vec=I_vec, S_vec=S_vec) + \
        _spin_dipole_dipole_coupling(consts.DipoleDipoleCoupling_c, I_vec=I_vec, S_vec=S_vec, n_vec=n_vec) + \
        _nuclear_spin_rotational_coupling(consts.NuclearSpinRotationalCoupling_C, I_vec=I_vec, N_vec=N_vec)
    
    return H

def zeeman(Cz,J):
    pass

def dc(Nmax,d0,I1,I2):
    pass

def ac_iso(Nmax,a0,I1,I2):
    pass

def ac_aniso(Nmax,a2,Beta,I1,I2):
    pass

def zeeman_ham(Nmax, consts):
    '''Assembles the Zeeman term and generates operator vectors

        Calculates the Zeeman effect for a magnetic field on a singlet-sigma molecule.
        There is no electronic term and the magnetic field is fixed to be along the z axis.

        Args:
            Nmax (int) - Maximum rotational level to include
            I1_mag,I2_mag (float) - magnitude of the nuclear spins
            Consts (Dictionary): Dict of molecular constants

        Returns:
            Hz (numpy.ndarray): Hamiltonian for the zeeman effect
    '''
    # N,I1,I2 = _generate_vecs(Nmax,I1_mag,I2_mag)
    # H = zeeman(consts['Mu1'],I1)+zeeman(consts['Mu2'],I2)+\
    #             zeeman(consts['MuN'],N)
    # return H

    pass

# This is the main build function and one that the user will actually have to
# use.
def build_hamiltonians(Nmax: int, consts: MolecularConstants, Zeeman: bool = False, Edc: bool = False, Eac: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' 
    Return the hyperfine hamiltonian.

    This function builds the hamiltonian matrices for evaluation so that
    the user doesn't have to rebuild them every time and we can benefit from
    numpy's ability to do distributed multiplication.

    Args:
        Nmax (int) - Maximum rotational level to include
        Consts (MolecularConstants) - class of molecular constants
        Zeeman, Edc, Aac (Boolean) - Switches for turning off parts of the total Hamiltonian 
        can save significant time on calculations where DC and AC fields are not required 

    Returns:
        H0, Hz, Hdc, Hac (numpy.ndarray): Each of the terms in the Hamiltonian.
    '''

    H0 = _hamiltonian_no_field(Nmax, consts)

    # if Zeeman:
    #     Hz = zeeman_ham(Nmax, consts)
    # else:
    #     Hz = 0

    # if Edc:
    #     Hdc = dc(Nmax,consts['d0'], I1, I2)
    # else:
    #     Hdc =0

    # if ac:
    #     Hac = (1./(2*eps0*c))*(ac_iso(Nmax,consts['a0'],I1,I2)+\
    #     ac_aniso(Nmax,consts['a2'],consts['Beta'],I1,I2))
    # else:
    #     Hac =0

    return (H0, 0, 0, 0)
    # return H0, Hz, Hdc, Hac
