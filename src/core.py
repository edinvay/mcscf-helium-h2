#####################
# One passes N_orbitals:
#
import argparse

parser = argparse.ArgumentParser(description="Run MCSCF with specified number of orbitals.")
parser.add_argument("N_orbitals", type=int, help="Number of orbitals")

args = parser.parse_args()
N_orbitals = args.N_orbitals
#
#####################



from vampyr import vampyr3d as vp3
import numpy as np
import scipy
import os as os_functions
import itertools

import pickle
from datetime import datetime


def calculate_overlap(Bra, Ket):
    r"""
    Compute the overlap matrix between two sets of functions.

    Computes a matrix \( S \) with entries
    \[
    S_{ij} = \langle \text{Bra}_i | \text{Ket}_j \rangle
    \]
    where \(\langle \cdot | \cdot \rangle\) is the inner product.
    """
    S = np.empty((len(Bra), len(Ket)))
    for i in range(len(Bra)):
        for j in range(len(Ket)):
            S[i, j] = vp3.dot(Bra[i], Ket[j])
    return S

# Löwdin orthonormalization S^{-1/2} = U * Sigma^{-1/2} * U^T
def lowdin_orthonormalization(Phi):
    sigma, U = np.linalg.eigh(calculate_overlap( Phi, Phi ))
    Sm5 = U @ np.diag(sigma**(-0.5)) @ U.transpose()
    return Sm5 @ Phi

def inner_product_vector(Phi, Psi):
    res = np.zeros_like(Phi)
    for i in range(N_orbitals):
        res[i] = vp3.dot(Phi[i], Psi[i])
    return res


