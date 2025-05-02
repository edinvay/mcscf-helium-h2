import sys
sys.path.append("src/")
from core import *


precision = 1.0e-4                 
XI = 0.001


molecule_name = 'helium'

Z = 2.0
x0 = (0., 0., 0.)
L = 0.5




def radius(x):
    return np.sqrt( (x[0] - x0[0])**2 + (x[1] - x0[1])**2 + (x[2] - x0[2])**2 )



def U(r):
    return (scipy.special.erf(r) / r) + (np.exp(-r**2) / np.sqrt(np.pi))

def V_xi(r, L, xi):
    Z = 2
    return - (2 * L * Z / xi) * U(r / xi)

def VLxi(x):
    r = radius(x)
    xi = XI
    return V_xi(r, L, xi)


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
