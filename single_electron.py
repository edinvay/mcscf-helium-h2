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


