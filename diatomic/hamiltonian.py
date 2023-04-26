import numpy as np
from sympy.physics.wigner import wigner_3j
from scipy.linalg import block_diag
import scipy.constants
from constants import MolecularConstants

'''
This module contains the main code to calculate the hyperfine structure of 
molecules in external electric and magnetic fields. In usual circumstances most of the functions within
are not user-oriented.

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


###############################################################################
# Start by definining constants that are needed for the code                  #
###############################################################################

'''
    Important note!

    Elements in the Hamiltonian have units of Hz (defined as E/h). 
'''

h = scipy.constants.h
muN = scipy.constants.physical_constants['nuclear magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
eps0 = scipy.constants.epsilon_0
c = scipy.constants.c
pi = np.pi

DebyeSI = 3.33564e-30 # Conversion factor from debyes to J/V/m

###############################################################################
# Functions for the calculations to use                                       #
###############################################################################

# first functions are mathematical and used to generate the structures that we
# will need to use

def raising_operator(J):
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
    elements = np.sqrt(J*(J+1)-mJ_list*(mJ_list+1))
    J_plus = np.diag(elements, 1)

    return J_plus

def x_operator(J):
    ''' 
    Creates the Cartesian operator Jx for a given J (x component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jx (numpy.ndarray) : 2J+1 square numpy array
    '''

    J_plus = raising_operator(J)

    return 0.5 * (J_plus + J_plus.T) # J_plus.T is lowering operator J_minus

def y_operator(J):
    ''' 
    Creates the Cartesian operator Jy for a given J (y component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jy (numpy.ndarray) : 2J+1 square numpy array
    '''

    J_plus = raising_operator(J)

    return 0.5j * (J_plus.T - J_plus) # J_plus.T is lowering operator J_minus

def z_operator(J):
    ''' 
    Creates the Cartesian operator Jz for a given J (z component of J)

    Args:
        J (float): Magnitude of angular momentum
    Returns:
        Jz (numpy.ndarray) : 2J+1 square numpy array
    '''

    assert float(2*J+1).is_integer()
    assert J >= 0

    return np.diag(np.arange(-J, J+1))

def generate_vecs(Nmax, S, I):
    ''' 
    Build N, S, I angular momentum vectors

    Generate the vectors of the angular momentum operators which we need
    to produce the Hamiltonian

    Args:
        Nmax (float): maximum rotational level to include in calculations
        S (float): electronic spin
        I (float): Nuclear spin, assume only one nucleus has non-zero spin
    Returns:
        N_vec, S_vec, I_vec (list of numpy.ndarray): length-3 list of (2Nmax+1)*(2S+1)*(2I+1) square numpy arrays
    '''

    assert isinstance(Nmax, int)
    assert Nmax >=0

    shapeN = int(np.sum([2*x+1 for x in range(Nmax+1)]))
    shapeS = int(2*S+1)
    shapeI = int(2*I+1)

    Nx = np.array([[]])
    Ny = np.array([[]])
    Nz = np.array([[]])

    for n in range(Nmax+1):
        Nx = block_diag(Nx, x_operator(n))
        Ny = block_diag(Ny, y_operator(n))
        Nz = block_diag(Nz, z_operator(n))

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
    S_vec = np.array([np.kron(np.identity(shapeN), np.kron(x_operator(S), np.identity(shapeI))),
                        np.kron(np.identity(shapeN), np.kron(y_operator(S),np.identity(shapeI))),
                        np.kron(np.identity(shapeN), np.kron(z_operator(S),np.identity(shapeI)))])

    I_vec = np.array([np.kron(np.identity(shapeN), np.kron(np.identity(shapeS),x_operator(I))),
                        np.kron(np.identity(shapeN), np.kron(np.identity(shapeS),y_operator(I))),
                        np.kron(np.identity(shapeN), np.kron(np.identity(shapeS),z_operator(I)))])

    return N_vec, S_vec, I_vec

# From here the functions will calculate individual terms in the Hamiltonian

def rotational(N_vec, Brot, Drot):
    ''' 
    Rigid rotor rotational structure

    Generates the hyperfine-free hamiltonian for the rotational levels of
    a rigid-rotor like molecule. Includes the centrifugal distortion term.

    Matrix is returned in the N,mN basis.

    Args:
        N_vec (list of numpy.ndarray) - length 3 list representing the Angular momentum vector for rotation
        Brot (float) - Rotational constant coefficient
        Drot (float) - Centrifugal distortion coefficient

    Returns:
        Hrot (numpy.ndarray) - hamiltonian for rotation in the N,MN basis
    '''

    N_squared = np.matmul(N_vec, N_vec).sum(axis=0) # np.matmul calculates Nx*Nx, Ny*Ny, Nz*Nz separately and then we sum them up
    N_squared_squared = np.matmul(N_squared, N_squared)

    return Brot*N_squared - Drot*N_squared_squared

def spin_rotational_coupling(gamma, S_vec, N_vec):
    ''' 
    Calculate the spin-rotational coupling term

    Args:
        gamma (float) - spin-rotational coupling coefficient
        S_vec, N_vec (list of numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return gamma*np.matmul(S_vec, N_vec).sum(axis=0)

def hyperfine(b, I_vec, S_vec):
    ''' 
    Calculate the hyperfine term

    Args:
        b (float) - hyperfine coefficient
        I_vec, S_vec (list of numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return b*np.matmul(I_vec, S_vec).sum(axis=0)

def spin_dipole_dipole_coupling(c, I_vec, S_vec):
    ''' 
    Calculate the spin dipoile-dipole coupling term

    Args:
        gamma (float) - spin-rotational coupling coefficient
        I_vec, S_vec (list of numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return c*np.matmul(I_vec[2], S_vec[2]) # index-2 is the z-component

def nuclear_spin_rotational_coupling(C, I_vec, N_vec):
    ''' 
    Calculate the nuclear spin-rotational coupling term

    Args:
        C (float) - nuclear spin-rotational coupling coefficient
        I_vec, N_vec (list of numpy.ndarray) - Angular momentum vectors

    Returns:
        H (numpy.ndarray) - Hamiltonian for spin-spin interaction
    '''

    return C*np.matmul(I_vec, N_vec).sum(axis=0)

def hamiltonian_no_field(Nmax, consts:MolecularConstants):
    '''
    Calculate the field-free Hyperfine hamiltonian

    Args:
        Nmax (int) - Maximum rotational level to include
        S, I (float) - electronic and nuclear spins
        Consts (MolecularConstants): class of molecular constants
    Returns:
        H : Hamiltonian for the hyperfine structure
    '''

    N_vec, S_vec, I_vec = generate_vecs(Nmax, S=consts.ElectronSpin_S, I=consts.NuclearSpin_I)
    H = rotational(Nmax=Nmax, Brot=consts.RotationalConstant_B, Drot=consts.CentrifugalDistortion_D) + \
        spin_rotational_coupling(consts.SpinRotationalCoupling_gamma, S_vec=S_vec, N_vec=N_vec) + \
        hyperfine(consts.HyperfineCoupling_b, I_vec=I_vec, S_vec=S_vec) + \
        spin_dipole_dipole_coupling(consts.SpinDipoleDipoleCoupling_c, I_vec=I_vec, S_vec=S_vec) + \
        nuclear_spin_rotational_coupling(consts.NuclearSpinRotationalCoupling_C, I_vec=I_vec, N_vec=N_vec)

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
    # N,I1,I2 = generate_vecs(Nmax,I1_mag,I2_mag)
    # H = zeeman(consts['Mu1'],I1)+zeeman(consts['Mu2'],I2)+\
    #             zeeman(consts['MuN'],N)
    # return H

    pass

# This is the main build function and one that the user will actually have to
# use.
def build_hamiltonians(Nmax, consts, Zeeman=False, Edc=False, Eac=False):
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

    H0 = hamiltonian_no_field(Nmax, consts)

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

    return H0, 0, 0, 0
    # return H0, Hz, Hdc, Hac
