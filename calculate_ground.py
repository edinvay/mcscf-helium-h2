#!/usr/bin/env python
# coding: utf-8

# # General case (constrained coefficients)
# 
# 
# The wave function is represented as a sum of $M + 1$ closed shell determinants
# $$
#     \Psi = \sum_{m = 0}^M c_m \left| m \overline{m} \right \rangle 
# $$
# Then the total energy takes the quadratic form
# $$
#     E
#     =
#     \left \langle
#         \Psi
#         \left| \widehat H \right|
#         \Psi
#     \right \rangle
#     =
#     \sum_{k, m = 0}^M
#     H_{km} c_k c_m
#     ,
# $$
# where the energy matrix elements are
# $$
#     H_{km}
#     =
#     \left \langle
#         k \overline{k}
#         \left| \widehat H \right|
#         m \overline{m}
#     \right \rangle
#     =
#     2 \delta_{km} (k|h|m)
#     +
#     (km|km)
# $$
# 
# Introduce the Lagrangian
# $$
#     \mathcal L
#     =
#     \frac 14 E
#     -
#     \frac 14 \varepsilon \left( \sum_{m = 0}^M c_m^2 - 1 \right)
#     -
#     \frac 12 \sum_{i, j  = 0}^M \varepsilon_{ij} \left( \int \varphi_i \varphi_j - \delta_{ij} \right)
# $$
# with symmetric matrix coefficients $\varepsilon_{ij}$.
# 

# In[1]:

import sys
sys.path.append("src/")
from core import *
N_orbitals = int(sys.argv[1])
molecule_name = str(sys.argv[2])

start_calculations = datetime.now()
print("Day YYYY-MM-DD and Time HH:MM:SS:")
print(start_calculations.strftime("%Y-%m-%d %H:%M:%S"))
print(" ")
print(" ")



from input import precision
from input import ZERO


from input import MAX_HISTORY_SCF

from input import outer_max
from input import inner_max
from input import trust_radius
from input import epsilon_matrix_correction_max

#from input import molecule_state
from lowdin import lowdin_orthonormalization



print(" ")
print("N_orbitals = ", N_orbitals)
print(" ")


########################################################################################
########################################################################################
#
# Helium:
#HF_total_energy = -2.8616677431287076
#
# H2:
#HF_total_energy = -1.84735656
#
########################################################################################
########################################################################################

equilibrium_internuclear_distance = get_equilibrium_internuclear_distance(molecule_name)
print(" ")
print("equilibrium_internuclear_distance:")
print(equilibrium_internuclear_distance)
print(" ")
#print("HF_total_energy:")
#print(HF_total_energy)
print(" ")










# In[3]:

Derivative = vp3.ABGVDerivative(mra, 0.5, 0.5)
print(mra)

P_mra = vp3.ScalingProjector(mra, precision)
Poisson = vp3.PoissonOperator(mra, precision)

def Laplace(f_tree):
    return Derivative(Derivative(f_tree, 0), 0) + Derivative(Derivative(f_tree, 1), 1) + Derivative(Derivative(f_tree, 2), 2)


# In[4]:


file_name = 'potential'

name = name_solution_file(
    directory_name = experiments_directory + molecule_name,
    file_name = file_name
)
V = vp3.ZeroTree(mra)
V.loadTree( name ) 
V.setName( 'potential' ) 

print(V)



# In[5]:





Guess_orbital = []

for n in range(N_orbitals):
    file_name = 'guess_orbital_' + str(n)
    print(file_name)
    name = name_solution_file(
        directory_name = experiments_directory + molecule_name,
        file_name = file_name
    )
    guess = vp3.ZeroTree(mra)
    guess.loadTree( name )
    guess.setName( file_name ) 
    print(guess)
    Guess_orbital.append(guess)


Guess_orbital = np.array(Guess_orbital)
Guess_orbital, coeff = lowdin_orthonormalization(Guess_orbital, 1.0)



H_EIGENVALUE = []
EPSILON = []
COEFF = []
ORBITAL_ENERGY = []
DIIS_ITERATIONS = []
F_NORM = []



# Consider equation of the form
# \begin{equation}
#     -
#     \frac 12 \Delta \varphi(x)
#     -
#     \lambda \varphi(x)
#     =
#     \text{RHS}
# \end{equation}
# 
# 
# For example, the helium Hartree-Fock equation
# \begin{equation}
#     -
#     \frac 12 \Delta \varphi(x)
#     -
#     \varepsilon \varphi(x)
#     =
#     -
#     \left(
#         V_{\text{nuc}}(x) \varphi(x)
#         +
#         \int_{\mathbb R^3}
#         \frac{\varphi^2(y)}{|x - y|} dy \varphi(x)
#     \right)
# \end{equation}
# 
# We define operator $H_{\lambda}(\text{RHS}, \varphi)$ as
# \begin{equation}
#     H_{\lambda}(\text{RHS}, \varphi)
#     =
#     \left \{
#     \begin{aligned}
#         &
#         ( - \Delta - 2 \varepsilon )^{-1}
#         (\text{RHS})
#         , \quad
#         &
#         \varepsilon \leqslant 0
#         \\
#         &
#         ( - \Delta + 1 )^{-1}
#         (
#             \text{RHS} + (\varepsilon + 1/2) \varphi
#         )
#         , \quad
#         &
#         \varepsilon > 0
#     \end{aligned}
#     \right.
# \end{equation}
# 
# Therefore, the main equation takes the form
# \begin{equation}
#     \varphi
#     =
#     2 H_{\lambda}(\text{RHS}, \varphi)
# \end{equation}

# In[9]:


class HelmholtzOperator(object):
    """
    lamb : mu = sqrt(-2*lamb)
    """
    def __init__(self, mra, lamb, prec):
        self.mra = mra
        self.lamb = lamb
        self.prec = prec
        self.operator = None
        self.setup()

    def setup(self):
        if self.lamb < - ZERO:
            self.operator = vp3.HelmholtzOperator(mra=self.mra, exp=np.sqrt(-2.0*self.lamb), prec=self.prec)
        elif self.lamb < ZERO:
            self.operator = Poisson
        else:
            self.operator = vp3.HelmholtzOperator(mra=self.mra, exp=1.0, prec=self.prec)

    def __call__(self, RHS, psi):
        res = None
        if self.lamb < ZERO:
            res = self.operator(RHS)
        else:
            res = RHS + (0.5 + self.lamb) * psi
            res = self.operator(res)
        return res


# $$
#     \begin{pmatrix}
#         0 & c_0 & \cdots & c_M
#         \\
#         c_0 & & &
#         \\
#         \vdots  & & \varepsilon - H &
#         \\
#         c_M & & & 
#     \end{pmatrix}
#     \begin{pmatrix}
#         \delta \varepsilon
#         \\
#         \delta c_0
#         \\
#         \vdots
#         \\
#         \delta c_M
#     \end{pmatrix}
#     =
#     \begin{pmatrix}
#         - \frac 12 \left( \sum_{m = 0}^M c_m^2 - 1 \right)
#         \\
#         f
#     \end{pmatrix}
# $$
# 
# $$
#     \begin{pmatrix}
#         0 & \mathbf{c}^T
#         \\
#         \mathbf{c} & \varepsilon - H
#     \end{pmatrix}
#     \begin{pmatrix}
#         \delta \varepsilon
#         \\
#         \delta \mathbf c
#     \end{pmatrix}
#     =
#     \begin{pmatrix}
#         - \frac 12 \left( \sum_{m = 0}^M c_m^2 - 1 \right)
#         \\
#         f
#     \end{pmatrix}
# $$
# 
# where:
# $\mathbf{c} = (c_0, c_1, \dots, c_M)^T$ is a column vector,
# $\varepsilon - H$ is an $(M+1) \times (M+1)$ matrix.
# Here
# $$
#     f_k
#     =
#     \sum_{m = 0}^M
#     H_{km} c_m
#     -
#     \varepsilon c_k
#     +
#     4 c_k (\delta k | h | k )
#     +
#     2 \sum_{m = 0}^M
#     c_m ( \delta k m | k m )
#     +
#     2 \sum_{m = 0}^M
#     c_m ( k \delta m | k m )
#     , \quad
#     k = 0, \ldots, M
# $$
# 
# 
# 

# In[10]:


def build_coefficient_matrix(c, H, epsilon):
    """
    Constructs the matrix:
    
        (  0     c^T  )
        (  c  ε - H )
    
    Parameters:
    c : (M+1,) array_like
    H : (M+1, M+1) array_like
        Square matrix H.
    epsilon : float
    
    Returns:
    numpy.ndarray
        Constructed (M+2, M+2) matrix.
    """
    c = np.asarray(c).reshape(-1, 1)  # Ensure c is a column vector
    H = np.asarray(H)                 # Ensure H is an array
    M = H.shape[0]                    # Determine M from H
    
    if c.shape[0] != M:
        raise ValueError("Dimension mismatch: c must have the same length as the size of H")
    
    # Construct the (M+1, M+1) matrix
    matrix = np.zeros((M+1, M+1))
    matrix[0, 1:] = c.T  # First row
    matrix[1:, 0] = c[:, 0]  # First column
    matrix[1:, 1:] = epsilon * np.eye(M) - H  # Bottom-right block
    
    return matrix


# The equations for the orbital energy updates $\delta \varepsilon_{kj}$
# have the following matrix form
# $$
#     X + \mathcal E Y = F
#     ,
# $$
# where $\mathcal E = ( \varepsilon_{kj} )$,
# $X = ( \delta \varepsilon_{kj} )$ is symmetric
# and $Y = \left( \int \delta \varphi_j \varphi_k \right)$ is antisymmetric.
# 
# 
# ### Solving the Equation $X + E Y = F$
# 
# We want to solve the equation:
# 
# $$ X + E Y = F $$
# 
# where:
# - $X$ is symmetric: $X^T = X$,
# - $Y$ is antisymmetric: $Y^T = -Y$.
# 
# #### Step 1: Split into Symmetric and Antisymmetric Parts
# 
# Taking the transpose of both sides:
# 
# $$ X^T + Y^T E^T = F^T. $$
# 
# Using the properties of $X$ and $Y$, we rewrite:
# 
# $$ X + (-Y) E^T = F^T. $$
# 
# Now, we add and subtract the original equation:
# 
# $$ (X + X) + E Y - Y E^T = F + F^T, $$
# 
# $$ (X - X) + E Y + Y E^T = F - F^T. $$
# 
# These simplify to two separate equations:
# 
# $$ 2X = (F + F^T) - E Y + Y E^T, $$
# 
# $$ E Y + Y E^T = (F - F^T). $$
# 
# #### Step 2: Solve for $Y$
# 
# The equation for $Y$:
# 
# $$ E Y + Y E^T = (F - F^T) $$
# 
# is a **Sylvester equation**, which can be solved using `scipy.linalg.solve_sylvester`.
# 
# #### Step 3: Solve for $X$
# 
# Once $Y$ is found, we solve for $X$:
# 
# $$ X = \frac{1}{2} (F + F^T - E Y + Y E^T). $$
# 
# This ensures that $X$ remains symmetric and $Y$ remains antisymmetric.
# 

# In[11]:


def solve_symmetric_antisymmetric(E, F):
    # Ensure F is a square matrix
    assert E.shape == F.shape, "E and F must be square matrices of the same size"
    
    # Compute antisymmetric matrix Y
    Y = scipy.linalg.solve_sylvester(E, E.T, F - F.T)
    
    # Compute symmetric matrix X
    X = 0.5 * (F + F.T - E @ Y + Y @ E.T)
    
    return X, Y


# In[12]:


def F_SCF(delta_Phi, w, w_data):
    Phi            = w[0]
    epsilon        = w[1]
    coeff          = w[2]
    epsilon_matrix = w[3]
    conv      = w_data[0]
    h_vector  = w_data[1]
    h_matrix  = w_data[2]
    H_matrix  = w_data[3]
    Helmholtz = w_data[4]
    coefficient_matrix = w_data[5]
    
    delta_mult = np.empty( (N_orbitals, N_orbitals), dtype = delta_Phi.dtype )
    for i in range(N_orbitals):
        for j in range(N_orbitals):
            delta_mult[i, j] = delta_Phi[i] * Phi[j]

    delta_h_matrix = np.eye(N_orbitals)
    for i in range(N_orbitals):
        for j in range(N_orbitals):
            delta_h_matrix[i, j] = vp3.dot( delta_Phi[i], h_vector[j])

    f_vector = H_matrix @ coeff - epsilon * coeff + 2.0 * coeff * delta_h_matrix.diagonal()
    for k in range(N_orbitals):
        for m in range(N_orbitals):
            f_vector[k] += 2.0 * coeff[m] * vp3.dot( delta_mult[k, m] + delta_mult[m, k], conv[k, m] )

    first_entry = -0.5 * (np.sum(coeff**2) - 1)
    RHS = np.hstack((first_entry, f_vector))
    delta_epsilon_coeff = scipy.linalg.solve(coefficient_matrix, RHS, assume_a="sym")
    delta_coeff = delta_epsilon_coeff[1:]
    
    energy_RHS = np.eye(N_orbitals)
    for k in range(N_orbitals):
        for j in range(N_orbitals):
            energy_RHS[k, j] = coeff[k]**2 * ( delta_h_matrix[k, j] + h_matrix[k, j] ) - epsilon_matrix[k, j]
            for m in range(N_orbitals):
                energy_RHS[k, j] += coeff[k] * coeff[m] * vp3.dot( delta_mult[k, m] + delta_mult[m, k], conv[j, m] )
                temp = coeff[k] * coeff[m] * delta_Phi[m]
                temp += ( coeff[k] * delta_coeff[m] + delta_coeff[k] * coeff[m] + coeff[k] * coeff[m] ) * Phi[m]
                temp *= Phi[j]
                energy_RHS[k, j] += vp3.dot( temp, conv[k, m] )

                
    delta_epsilon_matrix = solve_symmetric_antisymmetric(epsilon_matrix, energy_RHS)[0]

    mathfrak_F = 2.0 * delta_coeff / coeff * epsilon_matrix.diagonal() - delta_epsilon_matrix.diagonal()
    mathfrak_F = mathfrak_F * Phi
    temp = ( coeff**2 + 2.0 * coeff * delta_coeff ) * Phi + coeff**2 * delta_Phi
    mathfrak_F += V * temp
    for k in range(N_orbitals):
        for m in itertools.chain(range(k), range(k + 1, N_orbitals)):    
            mathfrak_F[k] -= epsilon_matrix[k, m] * delta_Phi[m]
            mathfrak_F[k] -= ( delta_epsilon_matrix[k, m] + epsilon_matrix[k, m] ) * Phi[m]
    for k in range(N_orbitals):
        for m in range(N_orbitals):
            temp = ( delta_coeff[k] * coeff[m] + coeff[k] * delta_coeff[m] + coeff[k] * coeff[m] ) * Phi[m]
            temp += coeff[k] * coeff[m] * delta_Phi[m]
            mathfrak_F[k] += conv[k, m] * temp
    for k in range(N_orbitals):
        for m in range(N_orbitals):
            temp = 4 * np.pi * Poisson( delta_mult[k, m] + delta_mult[m, k] ).crop(precision)
            mathfrak_F[k] += coeff[k] * coeff[m] * temp * Phi[m]
        mathfrak_F[k].crop(precision)
    
    new_delta_Phi = ( - 1.0 - 2.0 * delta_coeff / coeff ) * Phi
    for k in range(N_orbitals):
        new_delta_Phi[k] -= 2.0 / coeff[k]**2 * Helmholtz[k](mathfrak_F[k], delta_Phi[k])
        new_delta_Phi[k].crop(precision, True)
    
    return new_delta_Phi, [ delta_epsilon_coeff[0], delta_coeff, delta_epsilon_matrix ]


# In[13]:


#MAX_HISTORY_SCF = 3 #9

def f_g_SCF(x, w, w_data):
    psi, delta_epsilon_coeff_epsilon_matrix = F_SCF(x, w, w_data)
    return psi, x - psi, delta_epsilon_coeff_epsilon_matrix

def norm_SCF(x):
    return np.sqrt( sum([ psi.squaredNorm() for psi in x ]))

def dot_SCF(x, y):
    return sum([ vp3.dot(psi, phi) for psi, phi in zip(x, y) ])

def form_B_matrix(X):
    n = len(X)
    B = np.empty((n, n))
    for j in range(n):
        for k in range(n):
            B[j][k] = dot_SCF(X[j], X[k])
    return B

def form_DIIS_matrix(X):
    n = len(X)
    B = form_B_matrix(X)
    column_vector = np.ones((n, 1))
    res = np.hstack((B, column_vector))
    row_vector = np.ones((1, n+1))
    res = np.vstack((res, row_vector))
    res[-1][-1] = 0
    return res

def ell(n):
    max_history = MAX_HISTORY_SCF - 1
    return max(0, n - max_history)

def linear_combination_SCF(c, X):
    res0 = np.array([vp3.FunctionTree(mra).setZero()] * len(X[0]))
    res1 = 0
    for ind, x in enumerate(X):
        res0 += c[ind] * x        
    return res0

def remove_old_history(x):
    while len(x) > MAX_HISTORY_SCF:
        del x[0]


# ## Newton Algorithm
# 
# - Prepare initial guess $w$
# - Outer loop:
#   - Initialise all operators and parameters depending only on $w$
#   - Set $\delta \varphi_1 = \delta \varphi_2 = 0$ for an iterative solution of Newton's linear system
#   - Inner loop:
#     - Solve the linear system by iterating over $\delta \varphi_1, \delta \varphi_2$
#   - Update $w = w + \delta w$
# 


def calculate_energy(Phi):
    mult = np.empty((N_orbitals, N_orbitals), dtype=Phi.dtype)
    for i in range(N_orbitals):
        for j in range(i, N_orbitals):  # Loop only over the upper triangle
            value = Phi[i] * Phi[j]
            mult[i, j] = value
            mult[j, i] = value  # Use symmetry to fill the lower triangle
    
    conv = np.empty((N_orbitals, N_orbitals), dtype=Phi.dtype)
    for i in range(N_orbitals):
        for j in range(i, N_orbitals):  # Loop only over the upper triangle
            value = 4 * np.pi * Poisson(mult[i, j]).crop(precision)
            conv[i, j] = value
            conv[j, i] = value  # Use symmetry to fill the lower triangle

    h_vector = np.array([ -0.5 * Laplace(phi) + V * phi for phi in Phi ])
    
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
    # Optional: force immediate cleanup
    #import gc
    #gc.collect()
    
    supplementary_data = [ conv, h_vector, h_matrix, H_matrix ]
    
    return H_eigenvalue, H_eigenvector, supplementary_data

# In[14]:


# Initial guess:
Phi = Guess_orbital
epsilon = 0.0

coeff = None
epsilon_matrix = - np.eye(N_orbitals)



# In[15]:



tolerance = np.sqrt(N_orbitals) * precision



for outer_index in range(outer_max):
    print(f"outer_index = {outer_index}")
    
    H_eigenvalue, H_eigenvector, supplementary_data = calculate_energy(Phi)
    if outer_index > 0 and H_eigenvalue > H_EIGENVALUE[-1] + ZERO:
        print("H_eigenvalue > H_EIGENVALUE[-1]")
        Phi = previous_Phi
        H_eigenvalue = H_EIGENVALUE[-1]
        H_eigenvector = COEFF[-1]
        trust_radius *= 0.5
        for ind in range(N_orbitals):
            epsilon_matrix[ind, ind] *= 10.0
            if epsilon_matrix[ind, ind] >= 0:
                print("POSITIVE ORBITAL ENERGY:", epsilon_matrix[ind, ind])
                epsilon_matrix[ind, ind] *= -1
    # Too expensive to recalculate:
    conv = supplementary_data[0]
    h_vector = supplementary_data[1]
    h_matrix = supplementary_data[2]
    H_matrix = supplementary_data[3]
    
    H_EIGENVALUE.append(H_eigenvalue)
    EPSILON.append(epsilon)
    COEFF.append(H_eigenvector)

    print("H_eigenvalue: ", H_eigenvalue)
    print("epsilon:      ", epsilon)
    epsilon = H_eigenvalue
    coeff = H_eigenvector
        
    coefficient_matrix = build_coefficient_matrix(coeff, H_matrix, epsilon)    
        

    helmholtz_lambda = epsilon_matrix.diagonal() / coeff**2
    Helmholtz = [ HelmholtzOperator(mra, lambda_1, precision) for lambda_1 in helmholtz_lambda ]

    w = [ Phi, epsilon, coeff, epsilon_matrix ]
    w_data = [ conv, h_vector, h_matrix, H_matrix, Helmholtz, coefficient_matrix ]
    
    delta_Phi = np.array([ vp3.ZeroTree(mra) for i in range(N_orbitals) ])
    # It would be itnteresting to try instead something like:
    #delta_Phi = np.array([ 0.1* Phi[1], 0.1* Phi[0] ])    
    
    

    x_iterations = [ delta_Phi ]
    f_iterations = []
    g_iterations = []

    DIIS_ITERATIONS.append(0)
    F_NORM.append(-1.0)

    for inner_index in range(inner_max):
        print(f"inner_index = {inner_index}")
        DIIS_ITERATIONS[-1] = inner_index

        f, g, delta_epsilon_coeff_epsilon_matrix = f_g_SCF(x_iterations[-1], w, w_data)
        f_iterations.append(f)
        g_iterations.append(g)

        norm_f = norm_SCF(f_iterations[-1])
        F_NORM[-1] = norm_f
        if norm_f > trust_radius:
            print("Outside trust region with norm of delta_Phi = ", norm_f)
            f_iterations[-1] *= trust_radius / norm_f
            for ind in range(N_orbitals):
                epsilon_matrix[ind, ind] *= 10.0
                if epsilon_matrix[ind, ind] >= 0:
                    print("POSITIVE ORBITAL ENERGY:", epsilon_matrix[ind, ind])
                    epsilon_matrix[ind, ind] *= -1
            break
        
        norm_g = norm_SCF(g_iterations[-1])
        if norm_g < tolerance:
            print("Precision is achieved at inner_index =", inner_index)
            break

        remove_old_history(x_iterations)
        remove_old_history(f_iterations)
        remove_old_history(g_iterations)

        ell_n0 = ell(len(x_iterations) - 1)
        for ell_n in range(ell_n0, len(x_iterations)):
            DIIS_matrix = form_DIIS_matrix(g_iterations[ell_n : ])
            DIIS_vector = np.zeros(np.shape(DIIS_matrix)[1])
            DIIS_vector[-1] = 1.0
            try:
                c = np.linalg.solve(DIIS_matrix, DIIS_vector)[:-1]
                x = linear_combination_SCF(c, f_iterations[ell_n : ])
                x_iterations.append(x)
                break
            except np.linalg.LinAlgError:
                print("DIIS matrix is singular and ell_n is optimised.")

        print("norm(f) = ", norm_f)
        
        
    delta_Phi = f_iterations[-1]
        
    previous_Phi = Phi
    Phi = Phi + delta_Phi
    epsilon += delta_epsilon_coeff_epsilon_matrix[0]
    coeff   += delta_epsilon_coeff_epsilon_matrix[1]
    epsilon_matrix += delta_epsilon_coeff_epsilon_matrix[2]
    Phi, coeff = lowdin_orthonormalization(Phi, coeff)
    for ind in range(N_orbitals):
        if epsilon_matrix[ind, ind] >= 0:
            print("POSITIVE ORBITAL ENERGY:", epsilon_matrix[ind, ind])
            epsilon_matrix[ind, ind] *= -1

    ORBITAL_ENERGY.append( epsilon_matrix.diagonal() )
    
    if outer_index < epsilon_matrix_correction_max:
        epsilon_matrix = - np.eye(N_orbitals)

    print("coeff = ", coeff)

    if norm_SCF(delta_Phi) < tolerance:
        break



# In[20]:

H_eigenvalue, coeff, supplementary_data = calculate_energy(Phi)

H_EIGENVALUE.append(H_eigenvalue)
EPSILON.append(epsilon)
COEFF.append(coeff)

print("epsilon_matrix:")
print(epsilon_matrix)
print("H_eigenvalue:")
print(H_eigenvalue)
print("epsilon:")
print(epsilon)
print("coeff:")
print(coeff)




print(" ")
end_calculations = datetime.now()
print("Day YYYY-MM-DD and Time HH:MM:SS:")
print(end_calculations.strftime("%Y-%m-%d %H:%M:%S"))
print(" ")
print(f"Elapsed time: {end_calculations - start_calculations}")
print(" ")


# In[33]:


improvement = {
    'molecule_name' : molecule_name,
    'coeff' :  coeff,
    'epsilon_matrix' :  epsilon_matrix,
    'H_eigenvalue' :  H_eigenvalue,
    'epsilon' :  epsilon,
    #'HF_total_energy' :  HF_total_energy,
    'equilibrium_internuclear_distance' : equilibrium_internuclear_distance,
    

    'H_EIGENVALUE' : H_EIGENVALUE,          
    'EPSILON' : EPSILON,                    
    'COEFF' : COEFF,                        
    'ORBITAL_ENERGY' : ORBITAL_ENERGY,       #collected after solving Newton's equation

    'outer_max' : outer_max,
    'inner_max' : inner_max,
    'trust_radius' : trust_radius,
    'epsilon_matrix_correction_max' : epsilon_matrix_correction_max,
    'tolerance' : tolerance,
    'precision' : precision,
    'polynomial_order' : polynomial_order,
    'computational_domain_radius' : computational_domain_radius,
    'MAX_HISTORY_SCF' : MAX_HISTORY_SCF,
    'DIIS_ITERATIONS' : DIIS_ITERATIONS,
    'F_NORM' : F_NORM,
    'elapsed_time' : end_calculations - start_calculations
}

file_name = f"general_{N_orbitals}_orbital"
name = name_solution_file(
    directory_name = experiments_directory + molecule_name,
    file_name = file_name
)

for index, phi in enumerate( Phi ):
    print(index)
    print(phi)
    phi.saveTree( name + '_phi_' + str(index) )

with open(name + '.pkl', 'wb') as file:
    pickle.dump(improvement, file)



print(" ")
end_calculations = datetime.now()
print("Day YYYY-MM-DD and Time HH:MM:SS:")
print(end_calculations.strftime("%Y-%m-%d %H:%M:%S"))
print(" ")
print(f"Elapsed time: {end_calculations - start_calculations}")
print(" ")


print("FINISHED")

