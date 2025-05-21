import sys
sys.path.append("src/")
from core import *
from hydrogen_type_orbitals import hydrogen_orbital
from hydrogen_type_orbitals import create_n_l_m_Z



precision = 1.0e-4                 


molecule_name = 'helium'





n_max = 2
#Z = 5
Z = 2

n_l_m_Z_list = create_n_l_m_Z(n_max, Z)
n_l_m_Z_list[0][-1] = 2
print(n_l_m_Z_list)






print(mra)
P_mra = vp3.ScalingProjector(mra, precision)


for n_l_m_Z in n_l_m_Z_list:
    n = n_l_m_Z[0]
    l = n_l_m_Z[1]
    m = n_l_m_Z[2]
    Z = n_l_m_Z[3]
    formatted_str = f"n={n}_l={l}_m={m}_Z={Z}"
    file_name = 'guess_' + formatted_str
    print(file_name)
    def f(x):
        return hydrogen_orbital(n, l, m, x[0], x[1], x[2], Z)
    
    print("projecting...")
    guess = P_mra(f)
    guess.normalize()
    name = name_solution_file(
        directory_name = experiments_directory + molecule_name,
        file_name = file_name
    )
    guess.saveTree( name )

