import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi


class Nelsen12(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {1: PiOverSigmaMinusPi}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return (1 / self.t - 1) ** self.theta

    @property
    def _raw_inv_generator(self):
        return 1 / (self.y ** (1 / self.theta) + 1)

    @property
    def _cdf_expr(self):
        return (
            1
            + ((self.u ** (-1) - 1) ** self.theta + (self.v ** (-1) - 1) ** self.theta)
            ** (1 / self.theta)
        ) ** (-1)

    def lambda_L(self):
        return 2 ** (-1 / self.theta)

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)
