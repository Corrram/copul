import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi


class Nelsen19(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, np.inf, left_open=False, right_open=True)
    special_cases = {0: PiOverSigmaMinusPi}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return sympy.exp(self.theta / self.t) - sympy.exp(self.theta)

    @property
    def _raw_inv_generator(self):
        return self.theta / sympy.log(self.y + sympy.exp(self.theta))

    @property
    def _cdf_expr(self):
        return self.theta / sympy.log(
            -sympy.exp(self.theta)
            + sympy.exp(self.theta / self.u)
            + sympy.exp(self.theta / self.v)
        )
