import sys
sys.path.append("src/")
from core import *
from hydrogen_type_orbitals import hydrogen_orbital
from hydrogen_type_orbitals import create_n_l_m_Z



precision = 1.0e-4                 

molecule_name = 'h2'
equilibrium_internuclear_distance = 1.4010784

x1 = (-0.5 * equilibrium_internuclear_distance, 0., 0.)
x2 = ( 0.5 * equilibrium_internuclear_distance, 0., 0.)


n_max = 1
Z = 2

n_l_m_Z_list = create_n_l_m_Z(n_max, Z)
n_l_m_Z_list[0][-1] = 1
print(n_l_m_Z_list)

Position = [["_left", x1], ["_right", x2]]




print(mra)
P_mra = vp3.ScalingProjector(mra, precision)


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
            directory_name = experiments_directory + molecule_name,
            file_name = file_name
        )
        guess.saveTree( name )

