from vampyr import vampyr3d as vp3
import numpy as np
import scipy
import os as os_functions
import itertools


n_max = 2


precision = 1.0e-4                 
ZERO = 1.0e-10

Position = ["_left", "_right"]

polynomial_order = 9

# Define molecule
def create_n_l_m_Z(n_max, Z):
    n_l_m_Z = []
    for n in range(1, n_max + 1):
        for l in range(n):
            for m in range(-l, l + 1):
                n_l_m_Z.append([ n, l, m, Z ])
    return n_l_m_Z



n_l_m_Z_list = create_n_l_m_Z(n_max, 1)
n_l_m_Z_list[0][-1] = 1
print(n_l_m_Z_list)


# Define molecule
N_orbitals = 2 * len(n_l_m_Z_list)
print(" ")
print("N_orbitals = ", N_orbitals)
print(" ")


molecule_name = 'h2'


def name_solution_file(
    directory_name = 'experiments/' + molecule_name,
    file_name = 'test'
):
    if not os_functions.path.exists(directory_name):
        os_functions.makedirs(directory_name)
    file_name = os_functions.path.join(directory_name, file_name)
    return file_name

computational_domain_radius = 20
mra = vp3.MultiResolutionAnalysis(order = polynomial_order, box = [-computational_domain_radius, computational_domain_radius]) # Computational domain in a.u.
print(mra)


Guess_orbital = []

for n_l_m_Z in n_l_m_Z_list:
    for position in Position:
        n = n_l_m_Z[0]
        l = n_l_m_Z[1]
        m = n_l_m_Z[2]
        Z = n_l_m_Z[3]
        formatted_str = f"n={n}_l={l}_m={m}_Z={Z}"
        file_name = 'guess_' + formatted_str + position
        print(file_name)
        name = name_solution_file(
            file_name = file_name
        )
        guess = vp3.ZeroTree(mra)
        guess.loadTree( name )
        guess.setName( file_name ) 
        print(guess)
        Guess_orbital.append(guess)

Guess_orbital_left = [ Guess_orbital[i] for i in range(0, N_orbitals, 2) ]
Guess_orbital_right = [ Guess_orbital[i] for i in range(1, N_orbitals, 2) ]

Guess_orbital_left = np.array(Guess_orbital_left)
Guess_orbital_right = np.array(Guess_orbital_right)

Guess_orbital_sum = Guess_orbital_right + Guess_orbital_left
Guess_orbital_dif = Guess_orbital_right - Guess_orbital_left

Guess_orbital = []
for s, d in zip(Guess_orbital_sum, Guess_orbital_dif):
    Guess_orbital.append(s)
    Guess_orbital.append(d)


file_name = f"guess_orbital_"
name = name_solution_file(
    file_name = file_name
)

for index, phi in enumerate( Guess_orbital ):
    print(index)
    #print(phi)
    phi.saveTree( name + str(index) )
