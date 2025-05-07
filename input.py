#N_orbitals = int(sys.argv[1])
#molecule_name = str(sys.argv[2])

precision = 1.0e-5                  #default: 1.0e-4
ZERO = 1.0e-10                      #default: 1.0e-10

MAX_HISTORY_SCF = 3                 #default: 3

outer_max = 100                     #default: 100
inner_max = 15                      #default: 15
trust_radius = 0.5                  #default: 0.5
epsilon_matrix_correction_max =  0  #default: 10
