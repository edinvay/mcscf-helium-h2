from vampyr import vampyr3d as vp3
import numpy as np
import scipy
import os as os_functions
import itertools
import pickle
from datetime import datetime


N_orbitals = None
experiments_directory = 'experiments/'

polynomial_order = 9
computational_domain_radius = 20

equilibrium_internuclear_distance_h2 = 1.4010784


mra = vp3.MultiResolutionAnalysis(order = polynomial_order, box = [-computational_domain_radius, computational_domain_radius]) # Computational domain in a.u.


def get_equilibrium_internuclear_distance(molecule_name):
    equilibrium_internuclear_distance = None
    if molecule_name == 'h2':
        equilibrium_internuclear_distance = equilibrium_internuclear_distance_h2
    return equilibrium_internuclear_distance


def name_solution_file(
    directory_name,
    file_name
):
    if not os_functions.path.exists(directory_name):
        os_functions.makedirs(directory_name)
    file_name = os_functions.path.join(directory_name, file_name)
    return file_name


def calculate_overlap(Bra, Ket):
    r"""
    Compute the overlap matrix between two sets of functions.

    Computes a matrix :math:`S` with entries:

    .. math::

        S_{ij} = \langle \text{Bra}_i \mid \text{Ket}_j \rangle

    where :math:`\langle \cdot \mid \cdot \rangle` is the inner product.
    """
    S = np.empty((len(Bra), len(Ket)))
    for i in range(len(Bra)):
        for j in range(len(Ket)):
            S[i, j] = vp3.dot(Bra[i], Ket[j])
    return S

def lowdin_orthonormalization(Phi):
    sigma, U = np.linalg.eigh(calculate_overlap(Phi, Phi))
    Sm5 = U @ np.diag(sigma**(-0.5)) @ U.T
    return Sm5 @ Phi

def inner_product_vector(Phi, Psi):
    res = np.zeros_like(Phi)
    for i in range(N_orbitals):
        res[i] = vp3.dot(Phi[i], Psi[i])
    return res

# Only parse arguments when running this file directly
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MCSCF with specified number of orbitals.")
    parser.add_argument("N_orbitals", type=int, help="Number of orbitals")
    args = parser.parse_args()
    N_orbitals = args.N_orbitals

    print(f"Running with N_orbitals = {N_orbitals}")
