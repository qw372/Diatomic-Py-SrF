import scipy.constants
from numpy import pi
from dataclasses import dataclass


###############################################################################
# Molecular Constants
# Check up to date if precision needed!
###############################################################################


h = scipy.constants.h
muN = scipy.constants.physical_constants['nuclear magneton'][0]
bohr = scipy.constants.physical_constants['Bohr radius'][0]
eps0 = scipy.constants.epsilon_0
c = scipy.constants.c
DebyeSI = 3.33564e-30

# Most recent Rb87Cs133 Constants are given in the supplementary 
#of Gregory et al., Nat. Phys. 17, 1149-1153 (2021)
#https://www.nature.com/articles/s41567-021-01328-7
# Polarisabilities are for 1064 nm reported 
#in Blackmore et al., PRA 102, 053316 (2020)
#https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.053316
Rb87Cs133 = {"I1":1.5,
            "I2":3.5,
            "d0":1.225*DebyeSI,
            "binding":114268135.25e6*h,
            "Brot":490.173994326310e6*h,
            "Drot":207.3*h,
            "Q1":-809.29e3*h,
            "Q2":59.98e3*h,
            "C1":98.4*h,
            "C2":194.2*h,
            "C3":192.4*h,
            "C4":19.0189557e3*h,
            "MuN":0.0062*muN,
            "Mu1":1.8295*muN,
            "Mu2":0.7331*muN,
            "a0":2020*4*pi*eps0*bohr**3, #1064nm
            "a2":1997*4*pi*eps0*bohr**3, #1064nm
            "Beta":0}

@dataclass
class MolecularConstants:
    """ 
    Define constants for double-sigma molecule (Hund's case (b))
    See John Barry's thesis chapter 2.4, 2.5 for details.
    """

    NuclearSpin_I: float # nuclear spin, assume only one nucleus has non-zero spin
    ElectronSpin_S: float # electronic spin
    DipoleMoment_d: float # Debye, body-frame dipole moment
    RotationalConstant_B: float # Hz, defined as E/h
    CentrifugalDistortion_D: float # Hz, defined as E/h
    SpinRotationalCoupling_gamma: float # Hz, defined as E/h
    HyperfineCoupling_b: float # Hz, defined as E/h
    SpinDipoleDipoleCoupling_c: float # Hz, defined as E/h
    NuclearSpinRotationalCoupling_C: float # Hz, defined as E/h

# See John Barry's thesis chapter 2.4, 2.5 for details
SrFConstants = MolecularConstants(NuclearSpin_I=1/2,
                                  ElectronSpin_S=1/2,
                                  DipoleMoment_d=3.4963,
                                  RotationalConstant_B=7.5108318e9,
                                  CentrifugalDistortion_D=7482.4,
                                  SpinRotationalCoupling_gamma=75.02249e6,
                                  HyperfineCoupling_b=97.6670e6,
                                  SpinDipoleDipoleCoupling_c=29.846e6,
                                  NuclearSpinRotationalCoupling_C=0.00230e6)