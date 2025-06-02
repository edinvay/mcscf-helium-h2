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



#from input import precision
precision = 1.0e-3
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



print(mra)
P_mra = vp3.ScalingProjector(mra, precision)


Guess_orbital = []

for ind, multi_index in enumerate(get_multi_indices(N_orbitals)):
    print(ind)
    print(multi_index)
    def f(x):
        return np.exp( (- x[0]*x[0] - x[1]*x[1] - x[2]*x[2]) * 0.5 ) * hermite_3d_eval(multi_index, x[0], x[1], x[2])
    print("projecting...")
    guess = P_mra(f)
    guess.normalize()
    print(guess)
    Guess_orbital.append(guess)

Guess_orbital = np.array(Guess_orbital)
print(np.linalg.norm(calculate_overlap(Guess_orbital, Guess_orbital) - np.eye(N_orbitals)))
