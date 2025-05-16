from core import calculate_overlap
import numpy as np
from input import molecule_state


def lowdin_orthonormalization(Phi):
    if molecule_state == 'ground':
        res = lowdin_orthonormalization_ground(Phi)
    return res


def lowdin_orthonormalization_ground(Phi):
    sigma, U = np.linalg.eigh(calculate_overlap(Phi, Phi))
    Sm5 = U @ np.diag(sigma**(-0.5)) @ U.T
    return Sm5 @ Phi
