import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from vampyr import vampyr3d as vp

x_min = -20
x_max =  20
box = [x_min,x_max]
poly_order = 5
prec = 1e-4
convergence_prec = 10*prec
convergence_prec_initguess = 10*prec
coulombpotential_regularizer = 0.001 # Regularize Coulomb potential with Boys function 1/r ≈ 2/sqrt(pi)*F_0(r^2/reg^2)

MRA = vp.MultiResolutionAnalysis(box=box, order=poly_order)
D = vp.ABGVDerivative(MRA,0.5,0.5)
Poisson = vp.PoissonOperator(MRA,prec)
P_mra = vp.ScalingProjector(mra=MRA, prec=prec)

def He_potential(r):
    Z = 2
    R = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    return -Z*sp.special.erf(R/coulombpotential_regularizer)/R

def He_initguess(l, r):
    # R1 = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    # return np.exp(-R1/(l+1)) + np.exp(-R2/(l+1))
    return hydrogen_orbital(l,2,r)

H2_dist = 1.4010784

def H2_potential(r):
    Z = 1
    r0 = H2_dist/2
    R1 = np.sqrt((r[0]+r0)**2+r[1]**2+r[2]**2)
    R2 = np.sqrt((r[0]-r0)**2+r[1]**2+r[2]**2)
    V = -Z*sp.special.erf(R1/coulombpotential_regularizer)/R1 - Z*sp.special.erf(R2/coulombpotential_regularizer)/R2
    return V

def H2_initguess(l, r):
    r0 = H2_dist/2
    return hydrogen_orbital(l,1,[r[0]+r0,r[1],r[2]]) + hydrogen_orbital(l,1,[r[0]-r0,r[1],r[2]])

def H2_stupid_initguess(l, r):
    Z = 1
    r0 = H2_dist/2
    R1 = np.sqrt((r[0]+r0)**2+r[1]**2+r[2]**2)
    R2 = np.sqrt((r[0]-r0)**2+r[1]**2+r[2]**2)
    L1 = gen_laguerre(l,1,2*Z*R1/(l+1))
    L2 = gen_laguerre(l,1,2*Z*R2/(l+1))
    return L1*np.exp(-Z*R1/(l+1)) + R2**l*np.exp(-Z*R2/(l+1))


def main():
    L = 6
    num_inner_iter = 2
    # ~ init_guess = np.empty(L,dtype=object)
    init_guess = np.empty(L+1,dtype=object)
    # ~ for l in range(L):
    for l in range(L+1):
        def initguess_function(r):
            return H2_stupid_initguess(l,r)
        init_guess[l] = P_mra(initguess_function)
    V_nuc = P_mra(H2_potential)
    print("Solving one-electron problem to use as initial guess.")
    # ~ init_guess = one_el_eigenstates(L,V_nuc,init_guess)
    init_guess = one_el_eigenstates(L+1,V_nuc,init_guess)
    x = np.linspace(x_min,x_max,2000)
    for psi in init_guess:
        y = [psi([xi,0,0]) for xi in x]
        plt.plot(x,y)
    plt.show()
    for psi in init_guess:
        plot_orbital(psi,-8,8)
    electrostatic_energy = 1/H2_dist
    Psi, coeff, E = MCSCF(L,V_nuc,init_guess,num_inner_iter,electrostatic_energy)
    print("Converged!")
    print(f"E = {E+electrostatic_energy}")
    print(f"c = {coeff}")
    print(f"Occupation numbers of natural orbitals =\n{2*coeff**2}")
    # Save orbitals
    for l in range(L):
        Psi[l].saveTree(f"orbital_{l}")
    # Plot orbitals
    x = np.linspace(x_min,x_max,2000)
    for psi in Psi:
        y = [psi([xi,0,0]) for xi in x]
        plt.plot(x,y)
    plt.show()
    for psi in Psi:
        plot_orbital(psi,-8,8)
    return

def MCSCF(L, V_nuc, init_guess, num_inner_iter, electrostatic_energy):
    Psi = orthonormalize(init_guess,L)
    x = np.linspace(x_min,x_max,2000)
    max_num_iter = 1000
    for k in range(max_num_iter):
        num_inner_iter = 1 if k < 4 else 5
        if k == 0:
            print("Initial guess")
        else:
            print(f"Iteration {k}")
        print("====================")
        V_nucPsi = np.empty(L,dtype=object)
        for l in range(L):
            V_nucPsi[l] = V_nuc*Psi[l]
        h = compute_h(Psi,V_nucPsi,L)
        PsiPsi = compute_PsiPsi(Psi,L)
        J = compute_J(PsiPsi,L)
        # Perform CI
        E, c = configuration_interaction(h,PsiPsi,J,L)
        print(f"E = {E+electrostatic_energy}")
        print(f"c = {c}")
        # Compute K*Psi
        KPsi = np.empty(L,dtype=object)
        for l1 in range(L):
            KPsi[l1] = vp.ZeroTree(MRA)
            for l2 in range(L):
                KPsi[l1] += c[l2]*(J[l2,l1]*Psi[l2])
        # Compute epsilon matrix
        eps_mat = E*np.identity(L) - h
        # Test convergence
        converged = True
        for l1 in range(L):
            phi = V_nucPsi[l1] + (1/c[l1])*KPsi[l1]
            for l2 in range(L):
                if l1 != l2:
                    phi -= (c[l2]/c[l1]*eps_mat[l2,l1])*Psi[l2]
            kappa = np.sqrt(-2*eps_mat[l1,l1])
            G = vp.HelmholtzOperator(mra=MRA, exp=kappa, prec=prec)
            psi_test = -2*G(phi)
            Delta_psi_test = (psi_test-Psi[l1]).norm()
            print(f"Delta_psi_test = {Delta_psi_test}")
            if Delta_psi_test > convergence_prec:
                converged = False
                break
        if converged:
            h = compute_h(Psi,V_nucPsi,L)
            PsiPsi = compute_PsiPsi(Psi,L)
            J = compute_J(PsiPsi,L)
            E, c = configuration_interaction(h,PsiPsi,J,L)
            return Psi, c, E
        # Perform Newton step
        Psi, DeltaPsi_norm = newton_step(Psi,c,eps_mat,h,V_nuc,V_nucPsi,KPsi,J,L,num_inner_iter)
        Psi = orthonormalize(Psi,L)
        for psi in Psi:
            psi.crop(prec)
        print()


def compute_h(Psi, V_nucPsi, L):
    h = np.empty([L,L])
    D0Psi = np.empty(L,dtype=object)
    D1Psi = np.empty(L,dtype=object)
    D2Psi = np.empty(L,dtype=object)
    for l in range(L):
        D0Psi[l] = D(Psi[l],0)
        D1Psi[l] = D(Psi[l],1)
        D2Psi[l] = D(Psi[l],2)
    for l1 in range(L):
        h[l1,l1] = vp.dot(Psi[l1],V_nucPsi[l1])
        h[l1,l1] += 0.5*(D0Psi[l1].squaredNorm())
        h[l1,l1] += 0.5*(D1Psi[l1].squaredNorm())
        h[l1,l1] += 0.5*(D2Psi[l1].squaredNorm())
        for l2 in range(l1+1,L):
            h[l1,l2] = vp.dot(Psi[l1],V_nucPsi[l2])
            h[l1,l2] += 0.5*vp.dot(D0Psi[l1],D0Psi[l2])
            h[l1,l2] += 0.5*vp.dot(D1Psi[l1],D1Psi[l2])
            h[l1,l2] += 0.5*vp.dot(D2Psi[l1],D2Psi[l2])
            h[l2,l1] = h[l1,l2]
    return h


def compute_PsiPsi(Psi, L):
    PsiPsi = np.empty([L,L],dtype=object)
    for l1 in range(L):
        PsiPsi[l1,l1] = Psi[l1]*Psi[l1]
        for l2 in range(l1+1,L):
            PsiPsi[l1,l2] = Psi[l1]*Psi[l2]
            PsiPsi[l2,l1] = PsiPsi[l1,l2]
    return PsiPsi


def compute_J(PsiPsi, L):
    J = np.empty([L,L],dtype=object)
    for l1 in range(L):
        J[l1,l1] = (4*np.pi)*Poisson(PsiPsi[l1,l1])
        for l2 in range(l1+1,L):
            J[l1,l2] = (4*np.pi)*Poisson(PsiPsi[l1,l2])
            J[l2,l1] = J[l1,l2]
    return J


def configuration_interaction(h, PsiPsi, J, L):
    H = np.empty([L,L])
    for l1 in range(L):
        H[l1,l1] = 2*h[l1,l1] + vp.dot(PsiPsi[l1,l1],J[l1,l1])
        for l2 in range(l1+1,L):
            H[l1,l2] = vp.dot(PsiPsi[l1,l2],J[l1,l2])
            H[l2,l1] = H[l1,l2]
    E, C = np.linalg.eigh(H)
    return E[0], C[:,0]


def newton_step(Psi, coeff, eps_mat, h, V_nuc, V_nucPsi, KPsi, J, L, num_iter):
    print(f"epsilon =\n{eps_mat}")
    # print(f"|epsilon|       = {np.linalg.norm(eps_mat,ord=2)}")
    # print(f"|diag(epsilon)| = {np.linalg.norm(np.diag(np.diag(eps_mat)),ord=2)}")
    norm_eps_mat = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            norm_eps_mat[l1,l2] = (coeff[l1]*coeff[l2])*eps_mat[l1,l2]
    # Compute matrix E0
    E0 = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            E0[l1,l2] = coeff[l2]*(coeff[l2]*h[l1,l2] + vp.dot(Psi[l1],KPsi[l2])) - norm_eps_mat[l1,l2]
    # Compute Delta_eps_mat
    norm_eps, Q = np.linalg.eigh(norm_eps_mat)
    Delta_eps_mat = compute_Delta_epsilon(E0,norm_eps,Q,coeff,L)
    Phi = np.empty(L,dtype=object)
    for l1 in range(L):
        Phi[l1] = coeff[l1]*V_nucPsi[l1] + KPsi[l1]
        for l2 in range(L):
            Phi[l1] -= (coeff[l2]*Delta_eps_mat[l2,l1])*Psi[l2]
    eps, P = np.linalg.eigh(eps_mat)
    # print(f"E0 =\n{E0}")
    print("Inner iteration 1")
    print("--------------------")
    print(f"|Delta epsilon|/|epsilon| = {np.linalg.norm(Delta_eps_mat,ord=2)/np.linalg.norm(eps_mat,ord=2)}")
    # print(f"Delta epsilon =\n{Delta_eps_mat}")
    Psi_next = apply_greensfunction(Phi,eps,P,coeff,L)
    DeltaPsi = np.empty(L,dtype=object)
    for l in range(L):
        DeltaPsi[l] = Psi_next[l] - Psi[l]
    DeltaPsi0_norm = np.array([psi.norm() for psi in DeltaPsi])
    print(f"|DeltaPsi| =\n{DeltaPsi0_norm}")
    Y = compute_Y(Psi,DeltaPsi,L)
    # print(f"Y =\n{Y}")
    # Remaining inner iterations
    for k in range(num_iter-1):
        print(f"Inner iteration {k+2}")
        print("--------------------")
        for l in range(L):
            Phi[l] = vp.ZeroTree(MRA)
        for l1 in range(L):
            Delta_J = (8*np.pi)*Poisson(DeltaPsi[l1]*Psi[l1])
            Phi[l1] += coeff[l1]*(Delta_J*Psi[l1] + J[l1,l1]*DeltaPsi[l1])
            for l2 in range(l1+1,L):
                Delta_J = (4*np.pi)*Poisson(DeltaPsi[l2]*Psi[l1] + Psi[l2]*DeltaPsi[l1])
                Phi[l1] += coeff[l2]*(Delta_J*Psi[l2] + J[l2,l1]*DeltaPsi[l2])
                Phi[l2] += coeff[l1]*(Delta_J*Psi[l1] + J[l2,l1]*DeltaPsi[l1])
        D0Psi = np.empty(L,dtype=object)
        D1Psi = np.empty(L,dtype=object)
        D2Psi = np.empty(L,dtype=object)
        D0DeltaPsi = np.empty(L,dtype=object)
        D1DeltaPsi = np.empty(L,dtype=object)
        D2DeltaPsi = np.empty(L,dtype=object)
        for l in range(L):
            D0Psi[l] = D(Psi[l],0)
            D1Psi[l] = D(Psi[l],1)
            D2Psi[l] = D(Psi[l],2)
            D0DeltaPsi[l] = D(DeltaPsi[l],0)
            D1DeltaPsi[l] = D(DeltaPsi[l],1)
            D2DeltaPsi[l] = D(DeltaPsi[l],2)
        E = np.empty([L,L])
        for l1 in range(L):
            for l2 in range(L):
                Delta_h = vp.dot(V_nucPsi[l1],DeltaPsi[l2])
                Delta_h += 0.5*vp.dot(D0Psi[l1],D0DeltaPsi[l2])
                Delta_h += 0.5*vp.dot(D1Psi[l1],D1DeltaPsi[l2])
                Delta_h += 0.5*vp.dot(D2Psi[l1],D2DeltaPsi[l2])
                E[l1,l2] = E0[l1,l2] + coeff[l2]*(coeff[l2]*Delta_h + vp.dot(Psi[l1],Phi[l2]))
        Delta_eps_mat = compute_Delta_epsilon(E,norm_eps,Q,coeff,L)
        print(f"|Delta epsilon|/|epsilon| = {np.linalg.norm(Delta_eps_mat,ord=2)/np.linalg.norm(eps_mat,ord=2)}")
        # print(f"Delta epsilon =\n{Delta_eps_mat}")
        for l1 in range(L):
            Phi[l1] += coeff[l1]*(V_nuc*Psi_next[l1]) + KPsi[l1]
            for l2 in range(L):
                Phi[l1] -= (coeff[l2]*Delta_eps_mat[l2,l1])*Psi[l2]
        Psi_next = apply_greensfunction(Phi,eps,P,coeff,L)
        for l in range(L):
            DeltaPsi[l] = Psi_next[l] - Psi[l]
        DeltaPsi_norm = np.array([psi.norm() for psi in DeltaPsi])
        print(f"|DeltaPsi| =\n{DeltaPsi_norm}")
        Y = compute_Y(Psi,DeltaPsi,L)
        # print(f"Y =\n{Y}")
    return Psi_next, DeltaPsi0_norm


def apply_greensfunction(Psi, eps, P, coeff, L):
    Phi1 = np.empty(L,dtype=object)
    for l1 in range(L):
        Phi1[l1] = vp.ZeroTree(MRA)
        for l2 in range(L):
            Phi1[l1] += P[l2,l1]*Psi[l2]
    Phi2 = np.empty(L,dtype=object)
    for l in range(L):
        if eps[l] > 0:
            print(f"Positive orbital energy = {eps[l]} for orbital {l}")
            exit()
        kappa = np.sqrt(-2*eps[l])
        G = vp.HelmholtzOperator(mra=MRA, exp=kappa, prec=prec)
        Phi2[l] = -2*G(Phi1[l])
    for l1 in range(L):
        Phi1[l1] = vp.ZeroTree(MRA)
        for l2 in range(L):
            Phi1[l1] += P[l1,l2]*Phi2[l2]
        Phi1[l1] = (1/coeff[l1])*Phi1[l1]
    return Phi1


def compute_Delta_epsilon(E, norm_eps, Q, coeff, L):
    F = Q.transpose()@E@Q
    G = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            G[l1,l2] = (norm_eps[l1]*F[l1,l2]+norm_eps[l2]*F[l2,l1])/(norm_eps[l1]+norm_eps[l2])
    Delta_norm_eps_mat = Q@G@Q.transpose()
    Delta_eps_mat = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            Delta_eps_mat[l1,l2] = Delta_norm_eps_mat[l1,l2]/(coeff[l1]*coeff[l2])
    return Delta_eps_mat


def orthonormalize(Psi, L):
    S = compute_overlap_mat(Psi,L)
    s, Q = np.linalg.eigh(S)
    s = 1/np.sqrt(s)
    X = Q@np.diag(s)@Q.transpose()
    orthonorm_Psi = L*[vp.ZeroTree(MRA)]
    for l1 in range(L):
        for l2 in range(L):
            orthonorm_Psi[l1] += X[l2,l1]*Psi[l2]
    return orthonorm_Psi


def compute_overlap_mat(Psi, L):
    S = np.empty([L,L])
    for l1 in range(L):
        S[l1,l1] = Psi[l1].squaredNorm()
        for l2 in range(l1+1,L):
            S[l1,l2] = vp.dot(Psi[l1],Psi[l2])
            S[l2,l1] = S[l1,l2]
    return S


def compute_Y(Psi, DeltaPsi, L):
    Y = np.empty([L,L])
    for l1 in range(L):
        for l2 in range(L):
            Y[l1,l2] = vp.dot(Psi[l1],DeltaPsi[l2])
    return Y

def hydrogen_orbital_quantum_numbers(k):
    """
    Returns the k-th (starting at k=0) quantum numbers (n,l,m) of the hydrogen-like orbitals.
    """
    count = 0
    # Loop over quantum number n
    n = 1
    while True:
        for l in range(n):
            for m in range(-l,l+1):
                if count == k:
                    return (n,l,m)
                count += 1
        n += 1


def one_el_eigenstates(N, V_nuc, init_guess):
    Psi = orthonormalize(init_guess,N)
    for psi in Psi:
            psi.crop(prec)
    max_num_iter = 1000
    for k in range(max_num_iter):
        print(f"Iteration {k+1}")
        # Compute h matrix
        V_nucPsi = np.empty(N,dtype=object)
        for n in range(N):
            V_nucPsi[n] = V_nuc*Psi[n]
        h = compute_h(Psi,V_nucPsi,N)
        # Diagonalize h
        eps, Q = np.linalg.eigh(h)
        print(f"Orbital energies =\n{eps}")
        Phi = np.empty(N,dtype=object)
        QPsi = np.empty(N,dtype=object)
        for n1 in range(N):
            Phi[n1] = vp.ZeroTree(MRA)
            QPsi[n1] = vp.ZeroTree(MRA)
            for n2 in range(N):
                Phi[n1] += Q[n2,n1]*V_nucPsi[n2]
                QPsi[n1] += Q[n2,n1]*Psi[n2]
        # Apply Green's function and test convergence
        converged = True
        for n in range(N):
            if eps[n] > 0:
                print(f"Positive orbital energy {eps[n]} for orbital {n}!")
                # eps_0 = -0.1
                # Phi[n] += (eps[n]-eps_0)*QPsi[n]
                kappa = np.sqrt(2*eps[n])
                G = vp.HelmholtzOperator(mra=MRA, exp=kappa, prec=prec)
                psi_new = -2*G(Phi[n])
            else:
                kappa = np.sqrt(-2*eps[n])
                G = vp.HelmholtzOperator(mra=MRA, exp=kappa, prec=prec)
                psi_new = -2*G(Phi[n])
            Delta_psi = (psi_new-QPsi[n]).norm()
            print(f"Delta psi_{n} = {Delta_psi}")
            if Delta_psi > convergence_prec_initguess:
                converged = False
            Psi[n] = psi_new
        Psi = orthonormalize(Psi,N)
        for psi in Psi:
            psi.crop(prec)
        print()
        if converged:
            return Psi


def hydrogen_orbital(k, Z, r):
    """
    Evaluates k-th (starting at k=0) non-relativistic hydrogen-like orbital with atom number Z
    """
    (n,l,m) = hydrogen_orbital_quantum_numbers(k)
    R = np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    theta = np.arctan2(r[1],r[0])
    phi = np.arccos(r[2]/R)
    Y = sp.special.sph_harm(m,l,theta,phi)
    if m < 0:
        Y = np.imag(Y)
    else:
        Y = np.real(Y)
    L = gen_laguerre(n-l-1,2*l+1,2*Z*R/n)
    psi = Y*R**l*L*np.exp(-Z*R/n)
    return psi


def gen_laguerre(deg, alpha, x):
    """
    Evaluates eneralized laguerre polynomials
    """
    L0 = 1
    L = L0
    if deg >= 1:
        L1 = 1+alpha-x
        L = L1
    for k in range(2,deg+1):
        L = ((2*k-1+alpha-x)*L1 - (k-1+alpha)*L0)/k
        L0 = L1
        L1 = L
    return L
    


def plot_orbital(psi, x_min, x_max):
    # Grid definition
    n = 100
    x = y = z = np.linspace(x_min, x_max, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Evaluate signed density of orbital
    rho = np.empty(len(X))
    for i in range(len(X)):
        psi_val = psi([X[i],Y[i],Z[i]])
        s = np.sign(psi_val)
        rho_val = s*psi_val**2
        rho[i] = rho_val

    # Plot isosurface of orbital
    fig = go.Figure(data=go.Isosurface(
        x=X, y=Y, z=Z,
        value=rho,
        isomin=-0.001,
        isomax=0.001,
        surface_count=2,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale=[  # Red = negative, Blue = positive
            [0.0, 'red'],
            [0.5, 'white'],
            [1.0, 'blue']
        ],
        showscale=True,
        colorbar=dict(title='ψ')
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode='cube'
        )
    )
    fig.show()


def plot_saved_orbitals(L):
    for l in range(L):
        psi = vp.ZeroTree(MRA)
        psi.loadTree(f"orbital_{l}")
        plot_orbital(psi,-8,8)
    


main()
