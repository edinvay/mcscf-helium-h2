#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vampyr import vampyr3d as vp3
import numpy as np
import scipy
from scipy.special import factorial
import os as os_functions
import itertools


precision = 1.0e-4                 
ZERO = 1.0e-10
x0 = (0., 0., 0.)

equilibrium_internuclear_distance = 1.4010784

x1 = (-0.5 * equilibrium_internuclear_distance, 0., 0.)
x2 = ( 0.5 * equilibrium_internuclear_distance, 0., 0.)



# Define molecule
N_orbitals = 2
molecule_name = 'h2'


polynomial_order = 9


def name_solution_file(
    directory_name = 'experiments/' + molecule_name,
    file_name = 'test'
):
    if not os_functions.path.exists(directory_name):
        os_functions.makedirs(directory_name)
    file_name = os_functions.path.join(directory_name, file_name)
    return file_name


# In[2]:


def calculate_overlap(Bra, Ket):
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


# In[3]:


mra = vp3.MultiResolutionAnalysis(order = polynomial_order, box = [-20, 20]) # Computational domain in a.u.
Derivative = vp3.ABGVDerivative(mra, 0.5, 0.5)
print(mra)

P_mra = vp3.ScalingProjector(mra, precision)
Poisson = vp3.PoissonOperator(mra, precision)

def Laplace(f_tree):
    return Derivative(Derivative(f_tree, 0), 0) + Derivative(Derivative(f_tree, 1), 1) + Derivative(Derivative(f_tree, 2), 2)




# \begin{equation*}
#     A_m(x, y)
#     =
#     \sum_{p = 0}^m
#     \begin{pmatrix}
#         m
#         \\
#         p
#     \end{pmatrix}
#     x^p y^{m - p}
#     \cos
#     \left(
#         (m - p) \frac {\pi}2
#     \right)
# \end{equation*}
# 
# 
# \begin{equation*}
#     B_m(x, y)
#     =
#     \sum_{p = 0}^m
#     \begin{pmatrix}
#         m
#         \\
#         p
#     \end{pmatrix}
#     x^p y^{m - p}
#     \sin
#     \left(
#         (m - p) \frac {\pi}2
#     \right)
# \end{equation*}
# 
# 
# \begin{equation*}
#     \Pi_l^m(z)
#     =
#     \left(
#         \frac{ (l - m)! }{ (l + m)! }
#     \right)
#     ^{1/2}
#     \sum_{k = 0}^{ \lfloor (l - m) / 2 \rfloor }
#     (-1)^k
#     2^{-l}
#     \begin{pmatrix}
#         l
#         \\
#         k
#     \end{pmatrix}
#     \begin{pmatrix}
#         2l - 2k
#         \\
#         l
#     \end{pmatrix}
#     \frac{ (l - 2k)! }{ (l - 2k - m)! }
#     r^{2k}
#     z^{l - 2k - m}
# \end{equation*}
# 
# 
# Then for $m > 0$ we have
# \begin{equation*}
#     r^l
#     \begin{pmatrix}
#         Y_{l,m}
#         \\
#         Y_{l,-m}
#     \end{pmatrix}
#     =
#     \sqrt{ \frac{2l + 1}{2\pi} }
#     \Pi_l^m(z)
#     \begin{pmatrix}
#         A_m
#         \\
#         B_m
#     \end{pmatrix}
# \end{equation*}
# and for $m = 0$:
# \begin{equation*}
#     r^l Y_{l,0}
#     =
#     \sqrt{ \frac{2l + 1}{4\pi} }
#     \Pi_l^0(z)
# \end{equation*}
# 

# In[5]:


def A_m(x, y, m):
    """Compute the Cartesian polynomial A_m(x, y)."""
    return sum(
        factorial(m) / (factorial(p) * factorial(m - p)) * x**p * y**(m - p) *
        np.cos((m - p) * np.pi / 2)
        for p in range(m + 1)
    )

def B_m(x, y, m):
    """Compute the Cartesian polynomial B_m(x, y)."""
    return sum(
        factorial(m) / (factorial(p) * factorial(m - p)) * x**p * y**(m - p) *
        np.sin((m - p) * np.pi / 2)
        for p in range(m + 1)
    )

def Pi_lm(z, r, l, m):
    """Compute Π̄_l^m(z) as a function of (x, y, z) via r."""
    prefactor = np.sqrt(factorial(l - m) / factorial(l + m))
    sum_term = sum(
        (-1)**k * 2**(-l) *
        factorial(l) / (factorial(k) * factorial(l - k) * factorial(l - 2*k - m)) *
        r**(2*k) * z**(l - 2*k - m)
        for k in range((l - m) // 2 + 1)
    )
    return prefactor * sum_term

def solid_real_spherical_harmonic(l, m, x, y, z):
    """Compute r^l * real spherical harmonics Y_lm(x, y, z) avoiding division by r."""
    r = np.sqrt(x**2 + y**2 + z**2)

    # Compute Π̄_l^m(z)
    Pi = Pi_lm(z, r, l, abs(m))

    # Compute real spherical harmonics using Cartesian polynomials
    prefactor = np.sqrt((2 * l + 1) / (4 * np.pi))
    if m > 0:
        return prefactor * Pi * A_m(x, y, m)
    elif m < 0:
        return prefactor * Pi * B_m(x, y, -m)
    else:
        return prefactor * Pi  # m = 0 case

def hydrogen_orbital(n, l, m, x, y, z, Z=1):
    """Compute hydrogen-like atomic orbitals Ψ_nlm(x, y, z) in atomic units."""
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Radial part (Laguerre polynomials)
    rho = 2 * Z * r / n
    radial_part = np.sqrt((2 * Z / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) \
                  * np.exp(-rho / 2) * scipy.special.assoc_laguerre(rho, n - l - 1, 2 * l + 1)

    # Real spherical harmonics
    angular_part = solid_real_spherical_harmonic(l, m, x, y, z)

    return radial_part * angular_part




# In[7]:


def create_n_l_m_Z(n_max, Z):
    n_l_m_Z = []
    for n in range(1, n_max + 1):
        for l in range(n):
            for m in range(-l, l + 1):
                n_l_m_Z.append([ n, l, m, Z ])
    return n_l_m_Z


n_max = 2
Z = 2

n_l_m_Z_list = create_n_l_m_Z(n_max, Z)
n_l_m_Z_list[0][-1] = 1
print(n_l_m_Z_list)

Position = [["_left", x1], ["_right", x2]]

# In[8]:


for position in Position:
    for n_l_m_Z in n_l_m_Z_list:
        n = n_l_m_Z[0]
        l = n_l_m_Z[1]
        m = n_l_m_Z[2]
        Z = n_l_m_Z[3]
        formatted_str = f"n={n}_l={l}_m={m}_Z={Z}"
        file_name = 'guess_' + formatted_str + position[0]
        print(file_name)
        def f(x):
            return hydrogen_orbital(n, l, m, x[0] - position[1][0], x[1], x[2], Z)

        print("projecting...")
        guess = P_mra(f)
        guess.normalize()
        name = name_solution_file(
            file_name = file_name
        )
        guess.saveTree( name )





