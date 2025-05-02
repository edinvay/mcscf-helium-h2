import sys
sys.path.append("src/")
from core import *
from potential_smoothing import smoothed_singularity


precision = 1.0e-4                 
XI = 0.001


molecule_name = 'helium'

x0 = (0., 0., 0.)






def VLxi(x):
    return 2.0 * smoothed_singularity(x, x0, XI)


print(mra)

P_mra = vp3.ScalingProjector(mra, precision)

print("projecting...")
V = P_mra(VLxi)

file_name = 'potential'
name = name_solution_file(
    directory_name = experiments_directory + molecule_name,
    file_name = file_name
)
V.saveTree( name ) 
print(name)
print(V)
