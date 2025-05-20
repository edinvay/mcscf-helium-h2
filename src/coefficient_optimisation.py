from core import calculate_overlap
import numpy as np
from input import molecule_state
from core import vp3


from operators import Laplace
from operators import Poisson



class CoefficientOptimiser(object):
    def __init__(self, V, precision):
        self.V = V
        self.precision = precision
        if molecule_state == 'excited':
            from lowdin import Ground_orbital, Ground_coeff
            self.Ground_orbital = Ground_orbital
            self.Ground_coeff = Ground_coeff



    def calculate_energy(self, Phi):
        supplementary_data = self.calculate_energy_data(Phi)
        H_matrix = supplementary_data[-1]
        if molecule_state == 'ground':
            H_eigenvalue, H_eigenvector = self.solve_eigenvalue_problem(H_matrix)
        else:
            P = self.calculate_excited_projector(Phi)
            H_eigenvalue, H_eigenvector = self.solve_eigenvalue_problem(P @ H_matrix @ P)
        return H_eigenvalue, H_eigenvector, supplementary_data


    def solve_eigenvalue_problem(self, H_matrix):
        H_eigenvalue, H_eigenvector = np.linalg.eigh(H_matrix)
        H_eigenvalue = H_eigenvalue[0]
        H_eigenvector = H_eigenvector.T[0]
        H_eigenvector *= np.sign(H_eigenvector[0])
        return H_eigenvalue, H_eigenvector


    def calculate_energy_data(self, Phi):
        N_orbitals = len(Phi)
        mult = np.empty((N_orbitals, N_orbitals), dtype=Phi.dtype)
        for i in range(N_orbitals):
            for j in range(i, N_orbitals):  # Loop only over the upper triangle
                value = Phi[i] * Phi[j]
                mult[i, j] = value
                mult[j, i] = value  # Use symmetry to fill the lower triangle
        
        conv = np.empty((N_orbitals, N_orbitals), dtype=Phi.dtype)
        for i in range(N_orbitals):
            for j in range(i, N_orbitals):  # Loop only over the upper triangle
                value = 4 * np.pi * Poisson(mult[i, j]).crop(self.precision)
                conv[i, j] = value
                conv[j, i] = value  # Use symmetry to fill the lower triangle

        h_vector = np.array([ -0.5 * Laplace(phi) + self.V * phi for phi in Phi ])
        
        h_matrix = np.eye(N_orbitals)
        for i in range(N_orbitals):
            for j in range(i, N_orbitals):  # Loop only over the upper triangle
                value = vp3.dot(h_vector[i], Phi[j])
                h_matrix[i, j] = value
                h_matrix[j, i] = value  # Use symmetry to fill the lower triangle

        H_matrix = np.eye(N_orbitals)
        for k in range(N_orbitals):
            for m in range(k, N_orbitals):
                value = vp3.dot(mult[k, m], conv[k, m])
                H_matrix[k, m] = value
                H_matrix[m, k] = value  # Use symmetry to fill the lower triangle
        for k in range(N_orbitals):
            H_matrix[k, k] += 2.0 * h_matrix[k, k]

        del mult
        supplementary_data = [ conv, h_vector, h_matrix, H_matrix ]
        return supplementary_data


    def calculate_excited_projector(self, Phi):
        overlap = calculate_overlap(Phi, self.Ground_orbital)
        v = overlap ** 2 @ self.Ground_coeff
        P = np.outer(v, v)
        P /= np.trace(P)
        P = np.eye(len(Phi)) - P
        return P


