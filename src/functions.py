import scipy
import numpy as np




def build_coefficient_matrix(c, H, epsilon, v=None):
    """
    Constructs the matrix:
        If v is None:
            (  0     c^T  )
            (  c  ε - H )
        If v is provided:
            (  0     0     c^T )
            (  0     0     v^T )
            (  c     v   ε - H )

    Parameters:
    c : (M,) array_like
    H : (M, M) array_like
        Square matrix H.
    epsilon : float
    v : (M,) array_like or None
        Optional second vector.

    Returns:
    numpy.ndarray
        Constructed (M+1, M+1) or (M+2, M+2) matrix.
    """
    c = np.asarray(c).reshape(-1, 1)  # Column vector
    H = np.asarray(H)
    M = H.shape[0]

    if c.shape[0] != M or H.shape != (M, M):
        raise ValueError("Dimension mismatch between c and H")

    if v is None:
        # Original 1-row/column augmented matrix
        mat = np.zeros((M+1, M+1))
        mat[0, 1:] = c.T
        mat[1:, 0] = c[:, 0]
        mat[1:, 1:] = epsilon * np.eye(M) - H
    else:
        v = np.asarray(v).reshape(-1, 1)
        if v.shape[0] != M:
            raise ValueError("Dimension mismatch: v must be the same length as c")
        # New 2-row/column augmented matrix
        mat = np.zeros((M+2, M+2))
        mat[0, 2:] = c.T
        mat[1, 2:] = v.T
        mat[2:, 0] = c[:, 0]
        mat[2:, 1] = v[:, 0]
        mat[2:, 2:] = epsilon * np.eye(M) - H

    return mat

# The equations for the orbital energy updates $\delta \varepsilon_{kj}$
# have the following matrix form
# $$
#     X + \mathcal E Y = F
#     ,
# $$
# where $\mathcal E = ( \varepsilon_{kj} )$,
# $X = ( \delta \varepsilon_{kj} )$ is symmetric
# and $Y = \left( \int \delta \varphi_j \varphi_k \right)$ is antisymmetric.
# 
# 
# ### Solving the Equation $X + E Y = F$
# 
# We want to solve the equation:
# 
# $$ X + E Y = F $$
# 
# where:
# - $X$ is symmetric: $X^T = X$,
# - $Y$ is antisymmetric: $Y^T = -Y$.
# 
# #### Step 1: Split into Symmetric and Antisymmetric Parts
# 
# Taking the transpose of both sides:
# 
# $$ X^T + Y^T E^T = F^T. $$
# 
# Using the properties of $X$ and $Y$, we rewrite:
# 
# $$ X + (-Y) E^T = F^T. $$
# 
# Now, we add and subtract the original equation:
# 
# $$ (X + X) + E Y - Y E^T = F + F^T, $$
# 
# $$ (X - X) + E Y + Y E^T = F - F^T. $$
# 
# These simplify to two separate equations:
# 
# $$ 2X = (F + F^T) - E Y + Y E^T, $$
# 
# $$ E Y + Y E^T = (F - F^T). $$
# 
# #### Step 2: Solve for $Y$
# 
# The equation for $Y$:
# 
# $$ E Y + Y E^T = (F - F^T) $$
# 
# is a **Sylvester equation**, which can be solved using `scipy.linalg.solve_sylvester`.
# 
# #### Step 3: Solve for $X$
# 
# Once $Y$ is found, we solve for $X$:
# 
# $$ X = \frac{1}{2} (F + F^T - E Y + Y E^T). $$
# 
# This ensures that $X$ remains symmetric and $Y$ remains antisymmetric.
# 



def solve_symmetric_antisymmetric(E, F):
    # Ensure F is a square matrix
    assert E.shape == F.shape, "E and F must be square matrices of the same size"
    
    # Compute antisymmetric matrix Y
    Y = scipy.linalg.solve_sylvester(E, E.T, F - F.T)
    
    # Compute symmetric matrix X
    X = 0.5 * (F + F.T - E @ Y + Y @ E.T)
    
    return X, Y


def break_outer_loop(delta_Phi, delta_coeff, coeff, precision):
    norm_delta_Phi = np.array([ phi.norm() for phi in delta_Phi ])
    norm_delta_coeff = np.abs(delta_coeff)
    norm_coeff = np.abs(coeff)
    
    print("norm_delta_Phi   = ", norm_delta_Phi)
    print("norm_delta_coeff = ", norm_delta_coeff)
    print("norm_coeff       = ", norm_coeff)
    
    res = np.max( norm_coeff * norm_delta_Phi )
    print("res = ", res)
    res = max( res, np.max( norm_delta_coeff ) )
    print("res = ", res)
    
    return res < precision