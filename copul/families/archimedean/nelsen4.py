import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula


class GumbelHougaard(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {1: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return (-sympy.log(self.t)) ** self.theta

    @property
    def _raw_inv_generator(self):
        return sympy.exp(-(self.y ** (1 / self.theta)))

    @property
    def _cdf_expr(self):
        return sympy.exp(
            -(
                (
                    (-sympy.log(self.u)) ** self.theta
                    + (-sympy.log(self.v)) ** self.theta
                )
                ** (1 / self.theta)
            )
        )

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)

    def footrule(self):
        return 6*(1/(2**(1/self.theta) + 1))-2

Nelsen4 = GumbelHougaard

# B6 = GumbelHougaard

if __name__ == "__main__":
    # Example usage
    copula = GumbelHougaard(theta=2)
    footrule = copula.footrule()
    ccop = copula.to_checkerboard()
    ccop_footrule = ccop.footrule()
    ccop_xi = ccop.xi()
    ccop_rho = ccop.rho()
    print(f"Footrule distance: {footrule}, Checkerboard footrule: {ccop_footrule}",
          f"Checkerboard xi: {ccop_xi}", f"Checkerboard rho: {ccop_rho}")