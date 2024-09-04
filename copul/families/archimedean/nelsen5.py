import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Frank(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta_interval = sympy.Interval(-np.inf, np.inf, left_open=True, right_open=True)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _generator(self):
        return -sympy.log(
            (sympy.exp(-self.theta * self.t) - 1) / (sympy.exp(-self.theta) - 1)
        )

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def inv_generator(self):
        theta = self.theta
        y = self.y
        gen = (
            theta + y - sympy.log(-sympy.exp(theta) + sympy.exp(theta + y) + 1)
        ) / theta
        return SymPyFuncWrapper(gen)

    @property
    def cdf(self):
        theta = self.theta
        u = self.u
        v = self.v
        cdf = (
            -1
            / theta
            * sympy.log(
                1
                + (sympy.exp(-theta * u) - 1)
                * (sympy.exp(-theta * v) - 1)
                / (sympy.exp(-theta) - 1)
            )
        )
        return SymPyFuncWrapper(cdf)

    def cond_distr_1(self, u=None, v=None):
        expr_u = sympy.exp(-self.theta * self.u)
        expr_v = sympy.exp(-self.theta * self.v) - 1
        expr = sympy.exp(-self.theta) - 1
        cond_distr_1 = expr_v * expr_u / (expr + (-1 + expr_u) * expr_v)
        return SymPyFuncWrapper(cond_distr_1)(u, v)

    def _squared_cond_distr_1(self, v, u):
        theta = self.theta
        return (
            (-1 + sympy.exp(-theta * v)) ** 2
            * sympy.exp(-2 * theta * u)
            / (
                (-1 + sympy.exp(-theta * u)) * (-1 + sympy.exp(-theta * v))
                - 1
                + sympy.exp(-theta)
            )
            ** 2
        )

    def _xi_int_1(self, v):
        theta = self.theta
        return (
            theta * v * sympy.exp(2 * theta * v)
            - theta * v * sympy.exp(2 * theta * (v + 1))
            - 2 * theta * v * sympy.exp(theta * (v + 1))
            + 2 * theta * v * sympy.exp(theta * (v + 2))
            + theta * sympy.exp(2 * theta)
            + theta * sympy.exp(2 * theta * (v + 1))
            - 2 * theta * sympy.exp(theta * (v + 2))
            - sympy.exp(3 * theta * v)
            + sympy.exp(2 * theta * v)
            - sympy.exp(2 * theta * (v + 1))
            - sympy.exp(theta * (v + 1))
            + sympy.exp(theta * (v + 2))
            + sympy.exp(theta * (3 * v + 1))
        ) / (
            theta
            * (
                sympy.exp(2 * theta)
                + sympy.exp(4 * theta * v)
                - 2 * sympy.exp(3 * theta * v)
                + sympy.exp(2 * theta * v)
                + sympy.exp(2 * theta * (v + 1))
                - 2 * sympy.exp(theta * (v + 1))
                - 2 * sympy.exp(theta * (v + 2))
                + 4 * sympy.exp(theta * (2 * v + 1))
                - 2 * sympy.exp(theta * (3 * v + 1))
            )
        )

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        theta = self.theta
        func = 1 - 12 / theta * (self._d_1() - self._d_2())
        return sympy.Piecewise((func, theta != 0), (0, True))

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        theta = self.theta
        func = 1 - 4 / theta * (1 - self._d_1())
        return sympy.Piecewise((func, theta != 0), (0, True))

    def _d_1(self):
        t = sympy.Symbol("t")
        polylog = t * sympy.log(1 - sympy.exp(-t)) - sympy.polylog(2, sympy.exp(-t))
        return 1 / self.theta * polylog.subs(t, self.theta)  # todo fix integral

    def _d_2(self):
        t = sympy.Symbol("t")
        polylog = (
            -2 * t * sympy.polylog(2, sympy.exp(-t))
            - 2 * sympy.polylog(3, sympy.exp(-t))
            + t**2 * sympy.log(1 - sympy.exp(-t))
        )
        return 2 / self.theta**2 * polylog.subs(t, self.theta)  # todo fix integral

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0


Nelsen5 = Frank

# B3 = Frank
