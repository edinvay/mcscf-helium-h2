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

from input import molecule_state
if molecule_state == 'excited':
    import input
    input.set_ground_directory(molecule_name)
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











file_name = 'potential'

name = name_solution_file(
    directory_name = experiments_directory + molecule_name,
    file_name = file_name
)
V = vp3.ZeroTree(mra)
V.loadTree( name ) 
V.setName( 'potential' ) 

print(V)








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
coeff = np.array( [1.0] * len(Guess_orbital) )
Guess_orbital, coeff = lowdin_orthonormalization(Guess_orbital, coeff)



H_EIGENVALUE = []
EPSILON = []
COEFF = []
ORBITAL_ENERGY = []
DIIS_ITERATIONS = []
F_NORM = []






from operators import HelmholtzOperator

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




from functions import build_coefficient_matrix
from functions import solve_symmetric_antisymmetric



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

    if molecule_state == 'excited':
        lamb = w[4]
        v = w_data[6]
    
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
    if molecule_state == 'excited':
        overlap_Phi = calculate_overlap(Phi, CI_optimiser.Ground_orbital)
        overlap_delta_Phi = calculate_overlap(delta_Phi, CI_optimiser.Ground_orbital)
        temp = overlap_Phi * overlap_delta_Phi
        temp = temp @ CI_optimiser.Ground_coeff
        temp = v + 2.0 * temp
        first_entry = np.hstack(( first_entry, - np.dot(coeff, temp) ))
        f_vector -= lamb * temp
    RHS = np.hstack((first_entry, f_vector))
    delta_epsilon_coeff = scipy.linalg.solve(coefficient_matrix, RHS, assume_a="sym")
    delta_coeff = delta_epsilon_coeff[-N_orbitals:]
    
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
    if molecule_state == 'excited':
        delta_lamb = delta_epsilon_coeff[1]
        temp = overlap_Phi * CI_optimiser.Ground_coeff
        temp = temp.T
        energy_RHS -= np.diag(lamb * coeff) @ overlap_delta_Phi @ temp
        energy_RHS -= np.diag(lamb * delta_coeff + delta_lamb * coeff + lamb * coeff) @ overlap_Phi @ temp
                
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
    
    if molecule_state == 'excited':
        M_1 = np.diag(lamb * coeff) @ (overlap_delta_Phi * CI_optimiser.Ground_coeff)
        M_2 = np.diag(lamb * delta_coeff + delta_lamb * coeff + lamb * coeff) @ (overlap_Phi * CI_optimiser.Ground_coeff)
        mathfrak_F -= (M_1 + M_2) @ CI_optimiser.Ground_orbital
        for k in range(N_orbitals):
            mathfrak_F[k].crop(precision)
    
    new_delta_Phi = ( - 1.0 - 2.0 * delta_coeff / coeff ) * Phi
    for k in range(N_orbitals):
        new_delta_Phi[k] -= 2.0 / coeff[k]**2 * Helmholtz[k](mathfrak_F[k], delta_Phi[k])
        new_delta_Phi[k].crop(precision, True)
    
    delta_epsilon_coeff_epsilon_matrix = [ delta_epsilon_coeff[0], delta_coeff, delta_epsilon_matrix ]
    if molecule_state == 'excited':
        delta_epsilon_coeff_epsilon_matrix.append(delta_lamb)

    return new_delta_Phi, delta_epsilon_coeff_epsilon_matrix




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

from operators import Laplace
from operators import Poisson

from coefficient_optimisation import CoefficientOptimiser
CI_optimiser = CoefficientOptimiser(V, precision)

# Initial guess:
Phi = Guess_orbital
epsilon = 0.0

coeff = None
epsilon_matrix = - np.eye(N_orbitals)

if molecule_state == 'excited':
    lamb = 0.0



tolerance = np.sqrt(N_orbitals) * precision



for outer_index in range(outer_max):
    print(f"outer_index = {outer_index}")
    
    H_eigenvalue, H_eigenvector, supplementary_data = CI_optimiser.calculate_energy(Phi)
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
        
    v = None
    if molecule_state == 'excited':
        v = CI_optimiser.ground_vector(Phi)
        

    coefficient_matrix = build_coefficient_matrix(coeff, H_matrix, epsilon, v)
        

    helmholtz_lambda = epsilon_matrix.diagonal() / coeff**2
    Helmholtz = [ HelmholtzOperator(mra, lambda_1, precision) for lambda_1 in helmholtz_lambda ]

    w = [ Phi, epsilon, coeff, epsilon_matrix ]
    w_data = [ conv, h_vector, h_matrix, H_matrix, Helmholtz, coefficient_matrix ]
    
    if molecule_state == 'excited':
        w.append(lamb)
        w_data.append(v)
    
    delta_Phi = np.array([ vp3.ZeroTree(mra) for i in range(N_orbitals) ])
    
    

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
    if molecule_state == 'excited':
        lamb += delta_epsilon_coeff_epsilon_matrix[3]
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




H_eigenvalue, coeff, supplementary_data = CI_optimiser.calculate_energy(Phi)

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




improvement = {
    'molecule_name' : molecule_name,
    'coeff' :  coeff,
    'epsilon_matrix' :  epsilon_matrix,
    'H_eigenvalue' :  H_eigenvalue,
    'epsilon' :  epsilon,
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

