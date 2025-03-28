import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen22(BivArchimedeanCopula):
    ac = BivArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)
    special_cases = {0: BivIndependenceCopula}
    _generator_at_0 = sympy.pi / 2

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return sympy.asin(1 - self.t**self.theta)

    @property
    def _raw_inv_generator(self) -> SymPyFuncWrapper:
        indicator = sympy.Piecewise((1, self.y <= sympy.pi / 2), (0, True))
        return (1 - sympy.sin(self.y)) ** (1 / self.theta) * indicator

    @property
    def _cdf_expr(self):
        u = self.u
        theta = self.theta
        v = self.v
        return sympy.Piecewise(
            (
                (sympy.sin(sympy.asin(u**theta - 1) + sympy.asin(v**theta - 1)) + 1)
                ** (1 / theta),
                sympy.asin(u**theta - 1) + sympy.asin(v**theta - 1) >= -sympy.pi / 2,
            ),
            (0, True),
        )

    def compute_gen_max(self):
        return np.pi / 2

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
