import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Nelsen22(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", nonnegative=True)
    theta_interval = sympy.Interval(0, 1, left_open=False, right_open=False)
    special_cases = {0: IndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return sympy.asin(1 - self.t**self.theta)

    @property
    def inv_generator(self) -> SymPyFuncWrapper:
        indicator = sympy.Piecewise((1, self.y <= sympy.pi / 2), (0, True))
        gen = (1 - sympy.sin(self.y)) ** (1 / self.theta) * indicator
        return SymPyFuncWrapper(gen)

    @property
    def cdf(self) -> SymPyFuncWrapper:
        u = self.u
        theta = self.theta
        v = self.v
        cdf = sympy.Piecewise(
            (
                (sympy.sin(sympy.asin(u**theta - 1) + sympy.asin(v**theta - 1)) + 1)
                ** (1 / theta),
                sympy.asin(u**theta - 1) + sympy.asin(v**theta - 1) >= -sympy.pi / 2,
            ),
            (0, True),
        )
        return SymPyFuncWrapper(cdf)

    def compute_gen_max(self):
        return np.pi / 2

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
