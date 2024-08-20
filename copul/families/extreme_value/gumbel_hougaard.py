import numpy as np
import sympy

from copul.families.other import IndependenceCopula
from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class GumbelHougaard(ExtremeValueCopula):

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        if "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    theta = sympy.symbols("theta", positive=True)
    params = [theta]
    intervals = {"theta": sympy.Interval(1, np.inf, left_open=False, right_open=True)}

    @property
    def pickands(self):
        func = (self.t**self.theta + (1 - self.t) ** self.theta) ** (1 / self.theta)
        return SymPyFunctionWrapper(func)

    @property
    def cdf(self):
        cdf = sympy.exp(
            -(
                (
                    sympy.log(1 / self.v) ** self.theta
                    + sympy.log(1 / self.u) ** self.theta
                )
                ** (1 / self.theta)
            )
        )
        return SymPyFunctionWrapper(cdf)

    def _rho(self):
        t = self.t
        theta = self.theta
        integrand = ((t**theta + (1 - t) ** theta) ** (1 / theta) + 1) ** (-2)
        sympy.plot(integrand.subs(theta, 2), (t, 0, 1))
        # integrand = (
        #     (t * (1 - t)) ** (1 / theta - 1)
        #     / (1 + t ** (1 / theta) + (1 - t) ** (1 / theta)) ** 2
        #     * 1
        #     / theta
        # )
        # sympy.plot(integrand.subs(theta, 2), (t, 0, 1))
        return 12 * sympy.Integral(integrand, (t, 0, 1)) - 3

    def kendalls_tau(self):
        return (self.theta - 1) / self.theta
