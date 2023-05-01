import numpy as np
from scipy.constants import speed_of_light
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
    RotationalConstant_B: np.ndarray # Hz, defined as E/h, expressed in Dunham series in order of [[Y_00, Y_01, Y_02, ...], [Y_10, Y_11, Y_12, ...], ...]
    SpinRotationalCoupling_gamma: float # Hz, defined as E/h
    HyperfineCoupling_b: float # Hz, defined as E/h
    DipoleDipoleCoupling_c: float # Hz, defined as E/h
    NuclearSpinRotationalCoupling_C: float # Hz, defined as E/h


# See John Barry's thesis chapter 2.4, 2.5 for details
SrFConstants = MolecularConstants(NuclearSpin_I = 1/2,
                                  ElectronSpin_S = 1/2,
                                  DipoleMoment_d = 3.4963,
                                  RotationalConstant_B = np.array([[0, 0.250534383, -2.49586e-7, -3.30e-14]
                                                                   [501.96496, -1.551101e-3, -2.423e-10],
                                                                   [-2.204617, 2.1850e-6, 1.029e-11],
                                                                   [5.2815e-3, 1.518e-8]])*1e2*speed_of_light,
                                  SpinRotationalCoupling_gamma = 75.02249e6,
                                  HyperfineCoupling_b = 97.6670e6,
                                  DipoleDipoleCoupling_c = 29.846e6,
                                  NuclearSpinRotationalCoupling_C = 0.00230e6)