import numpy as np
import scipy


def radius(x, x0):
    return np.sqrt( (x[0] - x0[0])**2 + (x[1] - x0[1])**2 + (x[2] - x0[2])**2 )


def U(r):
    return (scipy.special.erf(r) / r) + (np.exp(-r**2) / np.sqrt(np.pi))


def V_xi(r, xi):
    return - (1.0 / xi) * U(r / xi)


def smoothed_singularity(x, x0, xi):
    r"""
    Returns :math:`-1/|x - x_0|`.

    Singularity :math:`-1/|x - x_0|` is smoothed by :math:`\xi`

    """
    return V_xi(radius(x, x0), xi)