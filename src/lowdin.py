from core import calculate_overlap
import numpy as np
from input import molecule_state

from scipy.optimize import minimize
from scipy.linalg import fractional_matrix_power


def lowdin_orthonormalization(Phi, coeff):
    if molecule_state == 'ground':
        res = lowdin_orthonormalization_ground(Phi, coeff)
    elif molecule_state == 'excited':
        res = lowdin_orthonormalization_excited(Phi, coeff)
    return res


def lowdin_orthonormalization_ground(Phi, coeff):
    Sm5 = fractional_matrix_power( calculate_overlap(Phi, Phi), -0.5 )
    return Sm5 @ Phi, coeff / np.linalg.norm(coeff)


def lowdin_orthonormalization_excited(Phi, coeff):
    G = calculate_overlap(Ground_orbital, Phi)
    PHI = calculate_overlap(Phi, Phi)
    A_opt, B_opt, b_opt, max_value = optimize_problem(len(Phi), len(Ground_orbital), PHI, G, coeff, Ground_coeff)
    return A_opt @ Phi+ B_opt @ Ground_orbital, b_opt


def objective(x, N, M, Phi, G, a):
    # Unpack flattened vector x into matrices A, B, and vector b
    A = x[:N*N].reshape(N, N)
    B = x[N*N:N*N+N*M].reshape(N, M)
    b = x[-N:]

    # First term: sum_k sum_n A_{kn} * Phi_{nk}
    term1 = np.sum(A * Phi.T)  # because Phi[n,k] = Phi_T[k,n]

    # Second term: sum_k sum_m B_{km} * G_{mk}
    term2 = np.sum(B * G.T)  # because G[m,k] = G_T[k,m]

    # Third term: a^T b
    term3 = np.dot(a, b)

    # Return negative for minimization
    return -(term1 + term2 + term3)

def orthonormality_constraint(x, N, M, Phi, G):
    A = x[:N*N].reshape(N, N)
    B = x[N*N:N*N+N*M].reshape(N, M)

    # Term 1: A @ Phi @ A.T
    term1 = A @ Phi @ A.T  # shape (N, N)

    # Term 2: B @ B.T
    term2 = B @ B.T  # shape (N, N)

    # Term 3: cross terms
    # G is M x N, so G.T is N x M
    # Compute G.T once
    GT = G.T  # shape (N, M)

    # Compute cross term C[k, l]
    # C_kl = sum_{m,n} (A_kn * B_lm + A_ln * B_km) * G_{mn}
    cross = np.zeros((N, N))
    for k in range(N):
        for l in range(N):
            # A[k, :] is A_kn, A[l, :] is A_ln
            # B[k, :] is B_km, B[l, :] is B_lm
            # G is M x N, so G[m, n]
            sum1 = A[k, None, :] * B[l, :, None]  # shape (M, N)
            sum2 = A[l, None, :] * B[k, :, None]  # shape (M, N)
            cross[k, l] = np.sum((sum1 + sum2) * G)

    # Full Gram matrix
    gram = term1 + term2 + cross

    # Return upper triangle minus identity
    i_upper = np.triu_indices(N)
    return (gram - np.eye(N))[i_upper]


def b_norm_constraint(x, N, M):
    b = x[-N:]
    return np.sum(b**2) - 1

def orthogonality_constraint(x, N, M, Phi, G, c):
    A = x[:N*N].reshape(N, N)   # shape (N, N)
    B = x[N*N:N*N+N*M].reshape(N, M)  # shape (N, M)
    b = x[-N:]  # shape (N,)

    total = 0.0
    for n in range(N):
        for m in range(M):
            overlap = np.dot(A[n], G[m]) + B[n, m]  # scalar: <ψ_n, g_m>
            total += b[n] * c[m] * overlap**2

    return total

def optimize_problem(N, M, Phi, G, a, c):
    # Initial guess
    A0 = fractional_matrix_power(Phi, -0.5)
    B0 = np.zeros((N, M))
    b0 = a / np.linalg.norm(a)

    x0 = np.concatenate([A0.flatten(), B0.flatten(), b0])

    constraints = [
        {'type': 'eq', 'fun': orthonormality_constraint, 'args': (N, M, Phi, G)},
        {'type': 'eq', 'fun': b_norm_constraint, 'args': (N, M)},
        {'type': 'eq', 'fun': orthogonality_constraint, 'args': (N, M, Phi, G, c)}
    ]

    result = minimize(
        objective, x0, args=(N, M, Phi, G, a),
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-12, 'maxiter': 10000, 'disp': False}
    )


    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    x_opt = result.x
    A_opt = x_opt[:N*N].reshape(N, N)
    B_opt = x_opt[N*N:N*N+N*M].reshape(N, M)
    b_opt = x_opt[-N:]

    return A_opt, B_opt, b_opt, -result.fun  # Return maximized value



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
