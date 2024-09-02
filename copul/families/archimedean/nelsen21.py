import numpy as np
import sympy
from scipy.integrate import nquad, quad

from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class Nelsen21(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _generator(self):
        return 1 - (1 - (1 - self.t) ** self.theta) ** (1 / self.theta)

    @property
    def inv_generator(self) -> SymPyFunctionWrapper:
        indicator = sympy.Piecewise((1, self.y <= sympy.pi / 2), (0, True))
        gen = (1 - (1 - (1 - self.y) ** self.theta) ** (1 / self.theta)) * indicator
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self) -> SymPyFunctionWrapper:
        t = self.theta
        u = self.u
        v = self.v
        expr = (1 - (1 - u) ** t) ** (1 / t) + (1 - (1 - v) ** t) ** (1 / t) - 1
        sympy_max = sympy.Max(expr, 0)
        cdf = 1 - (1 - sympy_max**t) ** (1 / t)
        return SymPyFunctionWrapper(cdf)

    def _cdf(self, u, v, t):
        return 1 - (
            1
            - np.max(
                (1 - (1 - u) ** t) ** (1 / t) + (1 - (1 - v) ** t) ** (1 / t) - 1, 0
            )
            ** t
        ) ** (1 / t)

    def cond_distr_2(self, u=None, v=None):
        th = self.theta
        expr = (
            (1 - (1 - self.u) ** th) ** (1 / th)
            + (1 - (1 - self.v) ** th) ** (1 / th)
            - 1
        )
        cond_distr = (
            ((1 - self.v) * sympy.Max(0, expr)) ** (th - 1)
            * (1 - (1 - self.v) ** th) ** ((1 - th) / th)
            * (1 - sympy.Max(0, expr) ** th) ** ((1 - th) / th)
            * sympy.Heaviside(expr)
        )
        return CD2Wrapper(cond_distr)(u, v)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)

    def _rho_int_1_sympy(self):
        u = self.u
        v = self.v
        t = self.theta
        lower_bound = 1 - (1 - (1 - (1 - (1 - v) ** t) ** (1 / t)) ** t) ** (1 / t)
        positive_cdf = 1 - (
            1 - ((1 - (1 - u) ** t) ** (1 / t) + (1 - (1 - v) ** t) ** (1 / t) - 1) ** t
        ) ** (1 / t)
        print(sympy.latex(positive_cdf))
        print(positive_cdf)
        return sympy.Integral(positive_cdf, (self.u, lower_bound, 1))

    def _rho_int_2(self, t):
        # integral_value, _ = dblquad(lambda u, v: self._cdf(u, v, t), 0, 1, 0, 1, limit=200)
        options = {"limit": 200}
        integral_value, _ = nquad(
            lambda u, v: self._cdf(u, v, t), [[0, 1], [0, 1]], opts=[options, options]
        )
        return integral_value

    def _rho(self, theta):
        return 12 * self._rho_int_2(theta) - 3

    # Define your function in a way that can be evaluated numerically
    def _positive_cdf(self, u, v, t):
        return 1 - (
            1 - ((1 - (1 - u) ** t) ** (1 / t) + (1 - (1 - v) ** t) ** (1 / t) - 1) ** t
        ) ** (1 / t)

    def _lower_bound(self, v, t):
        return 1 - (1 - (1 - (1 - (1 - v) ** t) ** (1 / t)) ** t) ** (1 / t)

    # Integration function
    def _rho_int_1(self, v, t):
        integral_value, _ = quad(self._cdf, 0, 1, args=(v, t))
        return integral_value
