from input import ZERO
from input import precision
from core import vp3
from core import mra
import numpy as np



Derivative = vp3.ABGVDerivative(mra, 0.5, 0.5)
print(mra)

Poisson = vp3.PoissonOperator(mra, precision)

def Laplace(f_tree):
    return Derivative(Derivative(f_tree, 0), 0) + Derivative(Derivative(f_tree, 1), 1) + Derivative(Derivative(f_tree, 2), 2)



# Consider equation of the form
# \begin{equation}
#     -
#     \frac 12 \Delta \varphi(x)
#     -
#     \lambda \varphi(x)
#     =
#     \text{RHS}
# \end{equation}
# 
# 
# For example, the helium Hartree-Fock equation
# \begin{equation}
#     -
#     \frac 12 \Delta \varphi(x)
#     -
#     \varepsilon \varphi(x)
#     =
#     -
#     \left(
#         V_{\text{nuc}}(x) \varphi(x)
#         +
#         \int_{\mathbb R^3}
#         \frac{\varphi^2(y)}{|x - y|} dy \varphi(x)
#     \right)
# \end{equation}
# 
# We define operator $H_{\lambda}(\text{RHS}, \varphi)$ as
# \begin{equation}
#     H_{\lambda}(\text{RHS}, \varphi)
#     =
#     \left \{
#     \begin{aligned}
#         &
#         ( - \Delta - 2 \varepsilon )^{-1}
#         (\text{RHS})
#         , \quad
#         &
#         \varepsilon \leqslant 0
#         \\
#         &
#         ( - \Delta + 1 )^{-1}
#         (
#             \text{RHS} + (\varepsilon + 1/2) \varphi
#         )
#         , \quad
#         &
#         \varepsilon > 0
#     \end{aligned}
#     \right.
# \end{equation}
# 
# Therefore, the main equation takes the form
# \begin{equation}
#     \varphi
#     =
#     2 H_{\lambda}(\text{RHS}, \varphi)
# \end{equation}


class HelmholtzOperator(object):
    """
    lamb : mu = sqrt(-2*lamb)
    """
    def __init__(self, mra, lamb, prec):
        self.mra = mra
        self.lamb = lamb
        self.prec = prec
        self.operator = None
        self.setup()

    def setup(self):
        if self.lamb < - ZERO:
            self.operator = vp3.HelmholtzOperator(mra=self.mra, exp=np.sqrt(-2.0*self.lamb), prec=self.prec)
        elif self.lamb < ZERO:
            self.operator = Poisson
        else:
            self.operator = vp3.HelmholtzOperator(mra=self.mra, exp=1.0, prec=self.prec)

    def __call__(self, RHS, psi):
        res = None
        if self.lamb < ZERO:
            res = self.operator(RHS)
        else:
            res = RHS + (0.5 + self.lamb) * psi
            res = self.operator(res)
        return res
