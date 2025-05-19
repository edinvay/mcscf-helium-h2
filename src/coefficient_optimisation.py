#from core import calculate_overlap
import numpy as np
#from input import molecule_state
from core import vp3


from operators import Laplace
from operators import Poisson



class CoefficientOptimiser(object):
    def __init__(self, V, precision):
        self.V = V
        self.precision = precision

    def calculate_energy(self, Phi):
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

        H_eigenvalue, H_eigenvector = np.linalg.eigh(H_matrix)
        H_eigenvalue = H_eigenvalue[0]
        H_eigenvector = H_eigenvector.T[0]
        H_eigenvector *= np.sign(H_eigenvector[0])
        
        del mult
        
        supplementary_data = [ conv, h_vector, h_matrix, H_matrix ]
        
        return H_eigenvalue, H_eigenvector, supplementary_data
