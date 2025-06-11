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

from input import molecule_state
if molecule_state == 'excited':
    print("Attention: simple ground Newton in use!")
    #raise ValueError(f"Invalid molecule_state: '{molecule_state}'. Must be 'ground', since 'excited' is not implemented yet.")


#from input import precision
precision = 1.0e-4
projection_precision = 1.0e-3
# DIIS:
Nmax = 9
tolerance = precision


from input import ZERO


from input import MAX_HISTORY_SCF


from input import molecule_state
if molecule_state == 'excited':
    import input
    input.set_ground_directory(molecule_name)
from lowdin import lowdin_orthonormalization



print(" ")
print("N_orbitals = ", N_orbitals)
print(" ")

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

##########################################################################################
##########################################################################################

def multi_indices(d):
    """
    Generate 3D multi-indices in graded lexicographic order:
    - Sorted by total degree (n_x + n_y + n_z)
    - Then lexicographically within each degree
    """
    indices = []
    for total_deg in range(d + 1):
        for n in itertools.product(range(total_deg + 1), repeat=3):
            if sum(n) == total_deg:
                indices.append(n)
    return indices

from math import comb

def degree_for_truncation(N):
    """
    Find the smallest d such that binomial(d + 3, 3) >= N
    """
    d = 0
    while comb(d + 3, 3) < N:
        d += 1
    return d

def get_multi_indices(N_orbitals):
    return multi_indices(degree_for_truncation(N_orbitals))[ : N_orbitals]


print(get_multi_indices(N_orbitals))




def hermite_3d_eval(n, x, y, z):
    """
    Evaluate the 3D Hermite polynomial H_{n_x}(x) * H_{n_y}(y) * H_{n_z}(z)
    at points (x, y, z).
    
    Inputs:
        n : tuple (n_x, n_y, n_z) of nonnegative integers
        x, y, z : floats or numpy arrays (must be broadcastable)
    
    Returns:
        H_n(x, y, z) = H_{n_x}(x) * H_{n_y}(y) * H_{n_z}(z)
    """
    nx, ny, nz = n
    return scipy.special.eval_hermite(nx, x) * scipy.special.eval_hermite(ny, y) * scipy.special.eval_hermite(nz, z)


def orthogonal_hermite_3d(multi_index, x):
    return np.exp( (- x[0]*x[0] - x[1]*x[1] - x[2]*x[2]) * 0.5 ) * hermite_3d_eval(multi_index, x[0], x[1], x[2])

##########################################################################################
##########################################################################################

def generate_gaussian_centers(N_orbitals, radius=5.0, min_distance=1.0):
    """
    Generate N_orbitals Gaussian centers in 3D.
    The first center is at the origin.
    The rest are randomly placed with a minimum pairwise distance.
    """
    centers = [np.zeros(3)]  # First Gaussian at origin

    while len(centers) < N_orbitals:
        candidate = np.random.uniform(-radius, radius, 3)

        if all(np.linalg.norm(candidate - c) >= min_distance for c in centers):
            centers.append(candidate)

    return np.array(centers)

centers = generate_gaussian_centers(N_orbitals, radius=4.0, min_distance=1.0)
for i, c in enumerate(centers):
    print(f"Center {i}: {c}")

Gauss = []
for i, c in enumerate(centers):
    Gauss.append( vp3.GaussFunc(beta = 1.0, position = c) )

print(mra)
P_mra = vp3.ScalingProjector(mra, projection_precision)

"""
Guess_orbital = []

for ind, multi_index in enumerate(get_multi_indices(N_orbitals)):
    print(ind)
    print(multi_index)
    def f(x):
        return orthogonal_hermite_3d(multi_index, x)
    print("projecting...")
    guess = P_mra(f)
    guess.normalize()
    print(guess)
    Guess_orbital.append(guess)

Guess_orbital = np.array(Guess_orbital)
print(np.linalg.norm(calculate_overlap(Guess_orbital, Guess_orbital) - np.eye(N_orbitals)))
"""

Guess_orbital = []

for ind, gauss in enumerate(Gauss):
    print(ind)
#    def f(x):
#        return orthogonal_hermite_3d(multi_index, x)
    print("projecting...")
    guess = P_mra(gauss)
    guess.normalize()
    print(guess)
    Guess_orbital.append(guess)

Guess_orbital = np.array(Guess_orbital)
print(np.linalg.norm(calculate_overlap(Guess_orbital, Guess_orbital) - np.eye(N_orbitals)))
coeff = np.array( [1.0] * N_orbitals )
Guess_orbital, coeff = lowdin_orthonormalization(Guess_orbital, coeff)


from operators import Laplace
from operators import HelmholtzOperator
from diis import DIIS


def F_SCF(Phi):
    N_orbitals = len(Phi)

    h_vector = np.array([ -0.5 * Laplace(phi) + V * phi for phi in Phi ])
    
    h_matrix = np.eye(N_orbitals)
    for i in range(N_orbitals):
        for j in range(i, N_orbitals):  # Loop only over the upper triangle
            value = vp3.dot(h_vector[i], Phi[j])
            h_matrix[i, j] = value
            h_matrix[j, i] = value  # Use symmetry to fill the lower triangle
    
    mathfrak_F = V * Phi
    for k in range(N_orbitals):
        for m in itertools.chain(range(k), range(k + 1, N_orbitals)):    
            mathfrak_F[k] -= h_matrix[k, m] * Phi[m]

    helmholtz_lambda = np.copy(h_matrix.diagonal())
    for ind in range(N_orbitals):
        if helmholtz_lambda[ind] >= 0:
            print("POSITIVE ORBITAL ENERGY:", helmholtz_lambda[ind])
            helmholtz_lambda[ind] *= -1    
    Helmholtz = [ HelmholtzOperator(mra, lambda_1, precision) for lambda_1 in helmholtz_lambda ]
    
    new_Phi = []
    for k, frak in enumerate(mathfrak_F):
        new_Phi.append( - 2.0 * Helmholtz[k](frak, Phi[k]) )
        new_Phi[-1].crop(precision, True)
    
    new_Phi = np.array(new_Phi)

    coeff = np.array( [1.0] * N_orbitals )
    new_Phi, coeff = lowdin_orthonormalization(new_Phi, coeff)

    
    return new_Phi, h_matrix


def f_g_SCF(x):
    psi, epsilon_matrix = F_SCF(x)
    return psi, x - psi, epsilon_matrix



single_electron_diis = DIIS(mra, Guess_orbital, MAX_HISTORY_SCF)



for n in range(Nmax):
    print(f"For n = {n} we have:")

    f, g, epsilon_matrix = f_g_SCF(single_electron_diis.x_iterations[-1])
    single_electron_diis.append_f_g(f, g)

    norm_f = single_electron_diis.calculate_norm_f()

    norm_g = single_electron_diis.calculate_norm_g()
    print("norm(g) = ", norm_g)
    if norm_g < tolerance:
        print("Precision is achieved at n =", n)
        break

    single_electron_diis.run()

    print("norm(f) = ", norm_f)


Phi = single_electron_diis.f_iterations[-1]

print("Orbital energies:")
print(epsilon_matrix.diagonal())

file_name = f"single_electron_{N_orbitals}_orbital"
name = name_solution_file(
    directory_name = experiments_directory + molecule_name,
    file_name = file_name
)

for index, phi in enumerate( Phi ):
    print(index)
    print(phi)
    phi.saveTree( name + '_phi_' + str(index) )

#with open(name + '.pkl', 'wb') as file:
#    pickle.dump(improvement, file)



print(" ")
end_calculations = datetime.now()
print("Day YYYY-MM-DD and Time HH:MM:SS:")
print(end_calculations.strftime("%Y-%m-%d %H:%M:%S"))
print(" ")
print(f"Elapsed time: {end_calculations - start_calculations}")
print(" ")


print("FINISHED")

