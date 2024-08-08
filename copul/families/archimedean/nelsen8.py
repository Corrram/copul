import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class Nelsen8(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _generator(self):
        return (1 - self.t) / (1 + (self.theta - 1) * self.t)

    @property
    def inv_generator(self):
        ind = sympy.Piecewise((1, self.y <= 1), (0, True))
        gen = (1 - self.y) / (self.theta * self.y - self.y + 1) * ind
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        num = self.theta**2 * self.u * self.v - (1 - self.u) * (1 - self.v)
        den = self.theta**2 - (self.theta - 1) ** 2 * (1 - self.u) * (1 - self.v)
        return SymPyFunctionWrapper(sympy.Max(num / den, 0))

    def cond_distr_1(self):
        theta = self.theta
        u = self.u
        v = self.v
        cond_distr_1 = (
            -(1 - v)
            * (theta - 1) ** 2
            * (theta**2 * u * v - (1 - u) * (1 - v))
            / (theta**2 - (1 - u) * (1 - v) * (theta - 1) ** 2) ** 2
            + (theta**2 * v - v + 1) / (theta**2 - (1 - u) * (1 - v) * (theta - 1) ** 2)
        ) * sympy.Heaviside(
            (theta**2 * u * v - (1 - u) * (1 - v))
            / (theta**2 - (1 - u) * (1 - v) * (theta - 1) ** 2)
        )
        return SymPyFunctionWrapper(cond_distr_1)

    def _squared_cond_distr_1(self, v, u):
        theta = self.theta
        sub_expr = theta**2 - (theta - 1) ** 2 * (u - 1) * (v - 1)
        return (
            (theta - 1) ** 2 * (v - 1) * (theta**2 * u * v - (u - 1) * (v - 1)) / sub_expr**2
            + (theta**2 * v - v + 1) / sub_expr
        ) ** 2 * sympy.Heaviside((theta**2 * u * v - (u - 1) * (v - 1)) / sub_expr)

    def _xi_int_1(self, v):
        theta = self.theta
        u = self.u
        integrand = (
            (theta - 1) ** 2
            * (u - 1)
            * (theta**2 * u * v - (u - 1) * (v - 1))
            / (theta**2 - (theta - 1) ** 2 * (u - 1) * (v - 1)) ** 2
            + (theta**2 * u - u + 1) / (theta**2 - (theta - 1) ** 2 * (u - 1) * (v - 1))
        ) ** 2
        simpler_integrand = sympy.simplify(integrand)
        int_1 = sympy.integrate(simpler_integrand, (u, 0, (1 - v) / (1 - v + theta**2)))
        return sympy.simplify(int_1)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
