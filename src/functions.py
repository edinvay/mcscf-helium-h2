import scipy

#from core import calculate_overlap
#import numpy as np
#from input import molecule_state

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

# In[11]:


def solve_symmetric_antisymmetric(E, F):
    # Ensure F is a square matrix
    assert E.shape == F.shape, "E and F must be square matrices of the same size"
    
    # Compute antisymmetric matrix Y
    Y = scipy.linalg.solve_sylvester(E, E.T, F - F.T)
    
    # Compute symmetric matrix X
    X = 0.5 * (F + F.T - E @ Y + Y @ E.T)
    
    return X, Y
