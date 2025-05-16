from core import calculate_overlap
import numpy as np
from input import molecule_state



def lowdin_orthonormalization(Phi, coeff):
    if molecule_state == 'ground':
        res = lowdin_orthonormalization_ground(Phi, coeff)
    elif molecule_state == 'excited':
        res = lowdin_orthonormalization_excited(Phi, coeff)
    return res


def lowdin_orthonormalization_ground(Phi, coeff):
    sigma, U = np.linalg.eigh(calculate_overlap(Phi, Phi))
    Sm5 = U @ np.diag(sigma**(-0.5)) @ U.T
    return Sm5 @ Phi, coeff / np.linalg.norm(coeff)


def lowdin_orthonormalization_excited(Phi, coeff):
    print(Ground_orbital, Ground_coeff)
    #sigma, U = np.linalg.eigh(calculate_overlap(Phi, Phi))
    #Sm5 = U @ np.diag(sigma**(-0.5)) @ U.T
    res = Ground_orbital
    return res, Ground_coeff


def load_ground():
    from input import Ground_directory
    from input import Ground_coefficient_file_name
    from input import Ground_orbital_file_name
    from core import mra
    from core import name_solution_file
    from core import vp3
    import pickle
    
    Ground_orbital = []

    for file_name in Ground_orbital_file_name:
        name = name_solution_file(
            directory_name = Ground_directory,
            file_name = file_name
        )
        orbital = vp3.ZeroTree(mra)
        orbital.loadTree( name )
        orbital.setName( file_name ) 
        Ground_orbital.append(orbital)

    Ground_orbital = np.array(Ground_orbital)

    name = name_solution_file(
            directory_name = Ground_directory,
            file_name = Ground_coefficient_file_name
    )
    with open(name + '.pkl', 'rb') as file:
        Ground_coeff = pickle.load(file)['coeff']

    return Ground_orbital, Ground_coeff



if molecule_state == 'excited':
    print("Loading the ground orbitals")
    Ground_orbital, Ground_coeff = load_ground()
