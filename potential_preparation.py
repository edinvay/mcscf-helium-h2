from vampyr import vampyr3d as vp3
import numpy as np
from scipy.special import erf as scipy_erf
import os as os_functions


precision = 1.0e-4                 
XI = 0.001


Z = 2.0
x0 = (0., 0., 0.)
L = 0.5

polynomial_order = 9


def name_solution_file(
    directory_name = 'experiments/helium',
    file_name = 'test'
):
    if not os_functions.path.exists(directory_name):
        os_functions.makedirs(directory_name)
    file_name = os_functions.path.join(directory_name, file_name)
    return file_name


def radius(x):
    return np.sqrt( (x[0] - x0[0])**2 + (x[1] - x0[1])**2 + (x[2] - x0[2])**2 )

def VL(x):
    r = radius(x)
    return - 2 * L * Z / r

def f_phi_exact_0(x):
    r = Z * 2 * L * radius(x)
    return np.exp(-r)

def f_phi_exact_1(x):
    minus_r_over_2 = - L * Z * radius(x)
    return np.exp(minus_r_over_2) * (1 + minus_r_over_2)


def U(r):
    return (scipy_erf(r) / r) + (np.exp(-r**2) / np.sqrt(np.pi))

def V_xi(r, L, xi):
    Z = 2
    return - (2 * L * Z / xi) * U(r / xi)

def VLxi(x):
    r = radius(x)
    xi = XI
    return V_xi(r, L, xi)


mra = vp3.MultiResolutionAnalysis(order = polynomial_order, box = [-20, 20]) # Computational domain in a.u.
print(mra)

P_mra = vp3.ScalingProjector(mra, precision)

V = P_mra(VLxi)

file_name = 'potential'
name = name_solution_file(
    file_name = file_name
)
V.saveTree( name ) 
print(V)
