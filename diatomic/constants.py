import numpy as np
from scipy.constants import physical_constants
from dataclasses import dataclass

__all__ = ['SrFConstants']

speed_of_light = physical_constants["speed of light in vacuum"][0]
electron_g_factor = physical_constants["electron g factor"][0] # -2.00231930436256
Bohr_magneton = physical_constants["Bohr magneton in Hz/T"][0]/1e4 # 1399624.49361 Hz/Gauss
nuclear_magneton = physical_constants["nuclear magneton in MHz/T"][0]*1e6/1e4 # 762.25932291 Hz/Gauss

@dataclass
class MolecularConstants:
    """ 
    Define constants for double-sigma molecule (Hund's case (b))
    See John Barry's thesis chapter 2.4, 2.5, 2.10 for details.
    """

    NuclearSpin_I: float # nuclear spin, assume only one nucleus has non-zero spin
    ElectronSpin_S: float # electronic spin
    DipoleMoment_d: float # Debye, body-frame dipole moment
    RotationalConstant_B: np.ndarray # Hz, defined as E/h, expressed in Dunham series in order of [[Y_00, Y_01, Y_02, ...], [Y_10, Y_11, Y_12, ...], ...]
    SpinRotationalCoupling_gamma: np.ndarray # Hz, defined as E/h, expressed in Dunham series in order of [[X_00, X_01, X_02, ...], [X_10, X_11, X_12, ...], ...]
    HyperfineCoupling_b: np.ndarray # Hz, defined as E/h, expressed in Dunham series in order of [[X_00, X_01, X_02, ...], [X_10, X_11, X_12, ...], ...]
    DipoleDipoleCoupling_c: np.ndarray # Hz, defined as E/h, expressed in Dunham series in order of [[X_00, X_01, X_02, ...], [X_10, X_11, X_12, ...], ...]
    NuclearSpinRotationalCoupling_C: np.ndarray # Hz, defined as E/h, expressed in Dunham series in order of [[X_00, X_01, X_02, ...], [X_10, X_11, X_12, ...], ...]
    EletronGFactor_gs: float = electron_g_factor# electron g factor
    ElectronOrbitalGFactor_gL: float = -1 # electron orbital g factor
    NuclearGFactor_gI: float = 0 # nuclear g factor, assume only one nucleus has non-zero spin
    BohrMagneton_muB: float = Bohr_magneton # Hz/Gauss
    NuclearMagneton_muN: float = nuclear_magneton # Hz/Gauss
    DebyeToHzcmperkV: float = 3.33564e-25/physical_constants['Planck constant'][0] # conversion factor from Debye to Hz/(kV/cm)


# See John Barry's thesis chapter 2.4, 2.5, 2.10 for details
SrFConstants = MolecularConstants(NuclearSpin_I = 1/2,
                                  
                                  ElectronSpin_S = 1/2,
                                  
                                  DipoleMoment_d = 3.4963,

                                  RotationalConstant_B = np.array([[0, 0.250534383, -2.49586e-7, -3.30e-14],
                                                                   [501.96496, -1.551101e-3, -2.423e-10, 0],
                                                                   [-2.204617, 2.1850e-6, 1.029e-11, 0],
                                                                   [5.2815e-3, 1.518e-8, 0, 0]])*1e2*speed_of_light, # make every row have the same length, it's easier to use later.

                                  SpinRotationalCoupling_gamma = np.array([[75.02249e6, 5.938e1, -6.3e-4],
                                                                           [-0.45528e6, -3.37, 0]]), # make every row have the same length, it's easier to use later.

                                  HyperfineCoupling_b = np.array([[97.6670e6, -3.300e2],
                                                                  [-1.1672e6, 0]]), # make every row have the same length, it's easier to use later.

                                  DipoleDipoleCoupling_c = np.array([[29.846e6],
                                                                     [0.843e6]]), # make every row have the same length, it's easier to use later.

                                  NuclearSpinRotationalCoupling_C = np.array([[0.00230e6]]), # make every row have the same length, it's easier to use later.
                                  
                                  NuclearGFactor_gI = 5.253736, # John Barry's thesis section 2.10 says g_I is 5.585, but wikipedia (Nuclear magnetic moment) says it's 5.253736 for 19F.
                                  
                                  ) 
