import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from vampyr import vampyr3d as vp

x_min = -20
x_max =  20
box = [x_min,x_max]
poly_order = 5
prec = 1e-4
# Convergence thresholds
energy_prec= 1e-8
orb_prec = 10*prec

MRA = vp.MultiResolutionAnalysis(box=box, order=poly_order)
D = vp.ABGVDerivative(MRA,0.5,0.5)
P = vp.PoissonOperator(MRA,prec)
P_mra = vp.ScalingProjector(mra=MRA, prec=prec)

def V_nuc_function(r):
    Z = 2
    # Regularize Coulomb potential with Boys function
    regularizer = 0.001
    R = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    return -Z*sp.special.erf(R/regularizer)/R

V_nuc = P_mra(V_nuc_function)

def main():
    L = 3
    # # Initial guess
    # def psi0_function(r):
    #     R = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    #     return np.exp(-R**2)
    # psi0 = P_mra(psi0_function)
    # Psi = []
    # for l in range(L):
    #     Psi += [psi0]
    #     for l2 in range(l):
    #         Psi[l] -= vp.dot(Psi[l2],Psi[l])*Psi[l2]
    #     Psi[l].normalize()
    #     x = np.linspace(x_min,x_max,500)
    #     for l2 in range(l+1):
    #         y = [Psi[l2]([xi,0,0]) for xi in x]
    #         plt.plot(x,y)
    #     plt.show()
    #     Psi = MCSCF(Psi, l+1)
    
    init_guess = []
    for l in range(L):
        def psi0_function(r):
            R = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
            return np.exp(-R/(l+1))
        init_guess += [P_mra(psi0_function)]
    MCSCF(init_guess,L)

def MCSCF(init_guess, L):
    plot = False
    Psi = init_guess
    Psi = orthonormalize(Psi,L)
    E, c = update_energy_and_coeff(Psi, L)
    l_focus = np.argmax(np.max(c**2))
    print("Initial guess")
    print(f"E = {E}, coefficients = {c}\n")
    if plot:
        x = np.linspace(x_min,x_max,500)
        for l in range(L):
            y = [Psi[l]([xi,0,0]) for xi in x]
            plt.plot(x,y)
        plt.show()

    # Iterations
    orbitals_converged = False
    energy_converged = False
    num_max_iter = 1000
    for k in range(num_max_iter):
        print(f"Iteration {k+1}")
        prev_Psi = Psi
        prev_E = E
        if k < 1000:
            print(f"Gradient descent with focus in orbital {l_focus}")
            eps_mat = compute_eps_mat(Psi, c, E, L)
            Psi = update_orbitals_graddesc(Psi, c, eps_mat, L, l_focus)
            delta_Psi = np.array([(Psi[l]-prev_Psi[l]).norm() for l in range(L)])
            # for l in range(L):
            #     Psi[l].normalize()
            Psi = canonical_representation(Psi,c,L)
            # Psi = orthonormalize(Psi,L)
        else:
            print("Quasi-Newton's method")
            newton_eps_mat, eps_mat = compute_newton_eps_mat(Psi, c, E, L)
            # eps_mat = update_newton_eps_mat(eps_mat, Psi, c, L)
            print(f"epsilon matrix =\n{eps_mat}")
            print(f"Newton's epsilon matrix =\n{newton_eps_mat}")
            if k % 2 == 0:
                levelshift = 0.01
            else:
                levelshift = 0
            # Psi = update_orbitals(Psi, c, eps_mat, newton_eps_mat, L, 0)
            Psi = update_orbitals_quasinewton(Psi, c, eps_mat, newton_eps_mat, L, levelshift)
            # Plot <deltapsi|psi> overlap matrix
            Y = np.empty([L,L])
            for l1 in range(L):
                for l2 in range(L):
                    Y[l1,l2] = vp.dot(Psi[l1],Psi[l2]-prev_Psi[l2])
            print(f"Overlap matrix <psi|Deltapsi> =\n{Y}")
            if plot:
                x = np.linspace(x_min,x_max,500)
                for l in range(L):
                    y = [Psi[l]([xi,0,0]) for xi in x]
                    plt.plot(x,y)
                plt.show()
            delta_Psi = np.array([(Psi[l]-prev_Psi[l]).norm() for l in range(L)])
            for l in range(L):
                Psi[l].normalize()
            # Psi = canonical_representation(Psi,c,L)
            Psi = orthonormalize(Psi,L)
        delta_norm_Psi = np.array([(Psi[l]-prev_Psi[l]).norm() for l in range(L)])
        if plot:
            x = np.linspace(x_min,x_max,500)
            for l in range(L):
                y = [Psi[l]([xi,0,0]) for xi in x]
                plt.plot(x,y)
            plt.show()
        l_focus = find_orbital_to_focus_on(delta_Psi,c)
        for l in range(L):
            Psi[l].crop(prec)
        E, c = update_energy_and_coeff(Psi, L)
        energy_converged = np.abs(E-prev_E) <= energy_prec
        if l_focus == None:
            orbitals_converged = True
            if not energy_converged:
                l_focus = np.argsort(c**2)[0]
        print(f"|psi - prev_psi|      = {delta_Psi}")
        print(f"|norm_psi - prev_psi| = {delta_norm_Psi}")
        print(f"coefficients          = {c}")
        print(f"E                     = {E}\n")
        # Test convergence
        if energy_converged and orbitals_converged:
            x = np.linspace(x_min,x_max,500)
            for l in range(L):
                y = [Psi[l]([xi,0,0]) for xi in x]
                plt.plot(x,y,label=f"Occupation number {2*L*c[l]**2}")
            plt.show()
            return Psi

def orthonormalize(Psi, L):
    S = compute_overlap_mat(Psi,L)
    s, Q = np.linalg.eigh(S)
    s = 1/np.sqrt(s)
    X = Q@np.diag(s)@Q.transpose()
    orthonorm_Psi = []
    for l1 in range(L):
        psi = vp.ZeroTree(MRA)
        for l2 in range(L):
            psi += X[l2,l1]*Psi[l2]
        orthonorm_Psi += [psi]
    return orthonorm_Psi

def canonical_representation(Psi, c, L):
    S = compute_overlap_mat(Psi,L)
    s, Q = np.linalg.eigh(S)
    sqrts = np.sqrt(s)
    sqrtS = Q@np.diag(s)@Q.transpose()
    # Constructing natural orbitals by diagonalizing density matrix
    D_mat = sqrtS@np.diag(c)@S@np.diag(c)@sqrtS
    rho, C = np.linalg.eigh(D_mat)
    rho = np.flip(rho,0)
    C = np.flip(C,1)
    invsqrts = 1/np.sqrt(s)
    X = Q@np.diag(invsqrts)@Q.transpose()
    X = X@C         # FIXME: Kan være oe galt med X
    canonical_Psi = []
    for l1 in range(L):
        sign = np.sign(X[l1,l1])
        psi = vp.ZeroTree(MRA)
        for l2 in range(L):
            psi += (sign*X[l2,l1])*Psi[l2]
        canonical_Psi += [psi]
    return canonical_Psi


def update_orbitals(Psi, coeff, eps_mat, newton_eps_mat, L, levelshift):
    new_Psi = []
    # Iterative step
    for l1 in range(L):
        psi = V_nuc*Psi[l1] - (0.5*levelshift/coeff[l1]**2 + newton_eps_mat[l1,l1] - eps_mat[l1,l1])*Psi[l1]
        for l2 in range(L):
            psi += (coeff[l2]/coeff[l1]*4*np.pi)*P(Psi[l1]*Psi[l2])*Psi[l2]
            if l1 != l2:
                psi -= (newton_eps_mat[l2,l1]/coeff[l1]**2)*Psi[l2]
        if eps_mat[l1,l1] > 0:
            print(f"Positive orbital energy = {eps_mat[l1,l1]} on orbital {l1}!\n")
            psi -= (eps_mat[l1,l1]/coeff[l1]**2)*Psi[l1]
            new_Psi += [-2*P(psi)]
        else:
            mu = np.sqrt(-2*eps_mat[l1,l1]+levelshift)/np.abs(coeff[l1])
            G = vp.HelmholtzOperator(mra=MRA, exp=mu, prec=prec)
            new_Psi += [-2*G(psi)]
    return new_Psi

def update_orbitals_graddesc(Psi, coeff, eps_mat, L, l_focus):
    if eps_mat[l_focus,l_focus] >= 0:
        print(f"Non-negative orbital energy = {eps_mat[l_focus,l_focus]/coeff[l_focus]**2} on orbital {l_focus}!\n")
        exit()
    stepsize = -0.5/eps_mat[l_focus,l_focus]
    # stepsize = -0.5/sum([eps_mat[l,l] for l in range(L)])*L
    # print(f"epsilon shift = {sum([eps_mat[l,l] for l in range(L)])/L}")
    # print(eps_mat)

    # Iterative step
    new_Psi = []
    for l1 in range(L):
        psi = V_nuc*Psi[l1]
        for l2 in range(L):
            psi += (coeff[l2]/coeff[l1]*4*np.pi)*P(Psi[l1]*Psi[l2])*Psi[l2]
            if l1 != l2:
                psi -= (eps_mat[l2,l1]/coeff[l1]**2)*Psi[l2]
            else:
                psi -= ((0.5/stepsize + eps_mat[l1,l1])/coeff[l1]**2)*Psi[l2]
        mu = np.sqrt(1/(stepsize*coeff[l1]**2))
        G = vp.HelmholtzOperator(mra=MRA, exp=mu, prec=prec)
        new_Psi += [-2*G(psi)]
    return new_Psi

def update_orbitals_quasinewton(Psi, coeff, eps_mat, newton_eps_mat, L, levelshift):
    Phi = []
    for l1 in range(L):
        Phi += [coeff[l1]*V_nuc*Psi[l1] - (0.5*levelshift/coeff[l1])*Psi[l1]]
        for l2 in range(L):
            Phi[l1] += (coeff[l2]*4*np.pi)*P(Psi[l1]*Psi[l2])*Psi[l2]
            Phi[l1] -= ((newton_eps_mat[l2,l1]-eps_mat[l2,l1])/coeff[l1])*Psi[l2]
    # Diagonalize epsilon matrix
    normalized_eps_mat = np.zeros([L,L])
    for l1 in range(L):
        normalized_eps_mat[l1,l1] -= 0.5*levelshift/coeff[l1]**2
        for l2 in range(L):
            normalized_eps_mat[l1,l2] += eps_mat[l1,l2]/(coeff[l1]*coeff[l2])
    e, Q = np.linalg.eigh(normalized_eps_mat)
    # Transform orbitals
    Phi2 = []
    for l1 in range(L):
        Phi2 += [vp.ZeroTree(MRA)]
        for l2 in range(L):
            Phi2[l1] += Q[l2,l1]*Phi[l2]
    for l in range(L):
        if e[l] > 0:
            print(f"Positive eigenvalue of normalized epsilon matrix = {e[l]}!")
            Phi2[l] = -2*P(Phi2[l])
        else:
            mu = np.sqrt(-2*e[l])
            G = vp.HelmholtzOperator(mra=MRA, exp=mu, prec=prec)
            Phi2[l] = -2*G(Phi2[l])
    # Transform back
    for l1 in range(L):
        Phi[l1].setZero()
        for l2 in range(L):
            Phi[l1] += Q[l1,l2]*Phi2[l2]
        Phi[l1] = (1/coeff[l1])*Phi[l1]
    return Phi

# Returns index of orbital with biggest occupation number that has not converged
def find_orbital_to_focus_on(delta_Psi, coeff):
    l_sorted = np.flip(np.argsort(coeff**2))
    for l in l_sorted:
        if delta_Psi[l] > orb_prec:
            return l
    return None

def update_energy_and_coeff(Psi, L):
    H = np.zeros([L,L])
    for l1 in range(L):
        # Potential energy
        H[l1,l1] += 2*vp.dot(Psi[l1],V_nuc*Psi[l1])
        # Kinetic energy
        Dpsi = D(Psi[l1],0)
        H[l1,l1] += Dpsi.squaredNorm()
        Dpsi = D(Psi[l1],1)
        H[l1,l1] += Dpsi.squaredNorm()
        Dpsi = D(Psi[l1],2)
        H[l1,l1] += Dpsi.squaredNorm()
        for l2 in range(L):
            # Two-electron repulsion energy
            psipsi = Psi[l1]*Psi[l2]
            H[l1,l2] += 4*np.pi*vp.dot(psipsi,P(psipsi))
    E, c = np.linalg.eigh(H)

    # c0 = 0.99
    # c = np.array([c0, np.sqrt(1-c0**2)])
    # E = c.transpose()@H@c
    # return E, c

    return E[0], c[:,0]

def compute_eps_mat(Psi, coeff, E, L):
    eps_mat = np.zeros([L,L])
    for l1 in range(L):
        eps_mat[l1,l1] += coeff[l1]**2*E
        for l2 in range(L):
            kinetic_energy = 1/2*(vp.dot(D(Psi[l1],0),D(Psi[l2],0))
                                 +vp.dot(D(Psi[l1],1),D(Psi[l2],1))
                                 +vp.dot(D(Psi[l1],2),D(Psi[l2],2)))
            potential_energy = vp.dot(Psi[l1],V_nuc*Psi[l2])
            eps_mat[l1,l2] -= coeff[l1]*coeff[l2]*(kinetic_energy + potential_energy)
    print(f"epsilon matrix =\n{eps_mat}")
    return eps_mat

def compute_newton_eps_mat(Psi, coeff, E, L):
    eps_mat = np.zeros([L,L])
    E_mat = np.zeros([L,L])
    for l1 in range(L):
        eps_mat[l1,l1] += coeff[l1]**2*E
        for l2 in range(L):
            kinetic_energy = 1/2*(vp.dot(D(Psi[l1],0),D(Psi[l2],0))
                                 +vp.dot(D(Psi[l1],1),D(Psi[l2],1))
                                 +vp.dot(D(Psi[l1],2),D(Psi[l2],2)))
            potential_energy = vp.dot(Psi[l1],V_nuc*Psi[l2])
            energy = kinetic_energy+potential_energy
            eps_mat[l1,l2] -= coeff[l1]*coeff[l2]*energy
            E_mat[l1,l2] += coeff[l2]**2*energy
            psi = (coeff[l2]*4*np.pi)*P(Psi[l1]*Psi[l2])
            for l3 in range(L):
                E_mat[l3,l1] += coeff[l1]*vp.dot(Psi[l3]*Psi[l2],psi)
    e, Q = np.linalg.eigh(eps_mat)
    print(f"E =\n{E_mat}")
    E_mat_copy = E_mat
    E_mat = Q.transpose()@E_mat@Q
    Omega = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            Omega[l1,l2] = (e[l1]*E_mat[l1,l2]+e[l2]*E_mat[l2,l1])/(e[l1]+e[l2])
    newton_eps_mat = Q@Omega@Q.transpose()
    Y = (E_mat_copy - newton_eps_mat)@np.linalg.inv(eps_mat)
    print(f"Y =\n{Y}")
    return newton_eps_mat, eps_mat

def update_newton_eps_mat(prev_eps_mat, Psi, coeff, L):
    E_mat = np.zeros([L,L])
    for l1 in range(L):
        for l2 in range(L):
            kinetic_energy = 1/2*(vp.dot(D(Psi[l1],0),D(Psi[l2],0))
                                 +vp.dot(D(Psi[l1],1),D(Psi[l2],1))
                                 +vp.dot(D(Psi[l1],2),D(Psi[l2],2)))
            potential_energy = vp.dot(Psi[l1],V_nuc*Psi[l2])
            energy = kinetic_energy+potential_energy
            E_mat[l1,l2] += coeff[l2]**2*energy
            psi = (coeff[l2]*4*np.pi)*P(Psi[l1]*Psi[l2])
            for l3 in range(L):
                E_mat[l3,l1] += coeff[l1]*vp.dot(Psi[l3]*Psi[l2],psi)
    e, Q = np.linalg.eigh(prev_eps_mat)
    print(f"E =\n{E_mat}")
    E_mat = Q.transpose()@E_mat@Q
    Omega = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            Omega[l1,l2] = (e[l1]*E_mat[l1,l2]+e[l2]*E_mat[l2,l1])/(e[l1]+e[l2])
    eps_mat = Q@Omega@Q.transpose()
    return eps_mat


def compute_overlap_mat(Psi, L):
    S = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            S[l1,l2] = vp.dot(Psi[l1],Psi[l2])
    return S


main()