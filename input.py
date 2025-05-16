#N_orbitals = int(sys.argv[1])
#molecule_name = str(sys.argv[2])

precision = 1.0e-4                  #default: 1.0e-4
ZERO = 1.0e-10                      #default: 1.0e-10

MAX_HISTORY_SCF = 3                 #default: 3

outer_max = 100                     #default: 100
inner_max = 15                      #default: 15
trust_radius = 0.5                  #default: 0.5
epsilon_matrix_correction_max =  0  #default: 10



#molecule_state = 'ground'
molecule_state = 'excited'

# Fill in the location to the ground state
if molecule_state == 'excited':
    Ground_directory = 'experiments/ground/helium'

    # Ground CI coefficients should be in a pkl-file accesible by the key 'coeff' (ommit extension '.pkl'):
    Ground_coefficient_file_name = 'general_1_orbital'

    Ground_orbital_file_name = []

    # Repeat for each orbital (ommit extension '.tree'):
    Ground_orbital_file_name.append( 'general_1_orbital_phi_0' )


