import scipy.constants
from dataclasses import dataclass

__all__ = ['SrFConstants']

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
    DipoleDipoleCoupling_c: float # Hz, defined as E/h
    NuclearSpinRotationalCoupling_C: float # Hz, defined as E/h


# See John Barry's thesis chapter 2.4, 2.5 for details
SrFConstants = MolecularConstants(NuclearSpin_I=1/2,
                                  ElectronSpin_S=1/2,
                                  DipoleMoment_d=3.4963,
                                  RotationalConstant_B=7.5108318e9,
                                  CentrifugalDistortion_D=7482.4,
                                  SpinRotationalCoupling_gamma=75.02249e6,
                                  HyperfineCoupling_b=97.6670e6,
                                  DipoleDipoleCoupling_c=29.846e6,
                                  NuclearSpinRotationalCoupling_C=0.00230e6)