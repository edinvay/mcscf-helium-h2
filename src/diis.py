import numpy as np
from core import vp3


class DIIS(object):
    def __init__(self, mra, x0, MAX_HISTORY_SCF):
        self.mra = mra
        self.x_iterations = [ x0 ]
        self.f_iterations = []
        self.g_iterations = []
        self.MAX_HISTORY_SCF = MAX_HISTORY_SCF



    def norm_SCF(self, x):
        # change:
        return np.sqrt( sum([ psi.squaredNorm() for psi in x ]))

    def dot_SCF(self, x, y):
        # change:
        return sum([ vp3.dot(psi, phi) for psi, phi in zip(x, y) ])

    def form_B_matrix(self, X):
        n = len(X)
        B = np.empty((n, n))
        for j in range(n):
            for k in range(n):
                B[j][k] = self.dot_SCF(X[j], X[k])
        return B

    def form_DIIS_matrix(self, X):
        n = len(X)
        B = self.form_B_matrix(X)
        column_vector = np.ones((n, 1))
        res = np.hstack((B, column_vector))
        row_vector = np.ones((1, n+1))
        res = np.vstack((res, row_vector))
        res[-1][-1] = 0
        return res

    def ell(self, n):
        max_history = self.MAX_HISTORY_SCF - 1
        return max(0, n - max_history)

    def linear_combination_SCF(self, c, X):
        # change:
        res0 = np.array([vp3.FunctionTree(self.mra).setZero() for _ in range(len(X[0]))])
        for ind, x in enumerate(X):
            res0 += c[ind] * x        
        return res0

    def remove_old_history(self, x):
        while len(x) > self.MAX_HISTORY_SCF:
            del x[0]


    def append_f_g(self, f, g):
        self.f_iterations.append(f)
        self.g_iterations.append(g)


    def calculate_norm_f(self):
        return self.norm_SCF(self.f_iterations[-1])

    def calculate_norm_g(self):
        return self.norm_SCF(self.g_iterations[-1])

    def run(self):
        self.remove_old_history(self.x_iterations)
        self.remove_old_history(self.f_iterations)
        self.remove_old_history(self.g_iterations)

        ell_n0 = self.ell(len(self.x_iterations) - 1)
        for ell_n in range(ell_n0, len(self.x_iterations)):
            DIIS_matrix = self.form_DIIS_matrix(self.g_iterations[ell_n : ])
            DIIS_vector = np.zeros(np.shape(DIIS_matrix)[1])
            DIIS_vector[-1] = 1.0
            try:
                c = np.linalg.solve(DIIS_matrix, DIIS_vector)[:-1]
                x = self.linear_combination_SCF(c, self.f_iterations[ell_n : ])
                self.x_iterations.append(x)
                break
            except np.linalg.LinAlgError:
                print("DIIS matrix is singular and ell_n is optimised.")


class ExtendedDIIS(DIIS):
    def __init__(self, mra, x0, MAX_HISTORY_SCF, N_orbitals):
        super(ExtendedDIIS, self).__init__(mra, x0, MAX_HISTORY_SCF)
        self.N_orbitals = N_orbitals
