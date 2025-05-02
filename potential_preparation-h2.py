import sys
sys.path.append("src/")
from core import *


precision = 1.0e-4                 
XI = 0.001


molecule_name = 'h2'

equilibrium_internuclear_distance = 1.4010784
x0 = ( -0.5 * equilibrium_internuclear_distance, 0., 0.)
x1 = (  0.5 * equilibrium_internuclear_distance, 0., 0.)






def radius(x, x0):
    return np.sqrt( (x[0] - x0[0])**2 + (x[1] - x0[1])**2 + (x[2] - x0[2])**2 )




def U(r):
    return (scipy.special.erf(r) / r) + (np.exp(-r**2) / np.sqrt(np.pi))

def V_xi(r, xi):
    return - (1.0 / xi) * U(r / xi)

def VLxi(x):
    xi = XI
    return V_xi(radius(x, x0), xi) + V_xi(radius(x, x1), xi)

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
