from scipy.special import factorial
import numpy as np
import scipy



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






def create_n_l_m_Z(n_max, Z):
    n_l_m_Z = []
    for n in range(1, n_max + 1):
        for l in range(n):
            for m in range(-l, l + 1):
                n_l_m_Z.append([ n, l, m, Z ])
    return n_l_m_Z

