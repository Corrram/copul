import numpy as np
import sympy

from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.families.other import IndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class GumbelHougaard(ExtremeValueCopula):
    delta = sympy.symbols("theta", positive=True)
    params = [delta]
    intervals = {"theta": sympy.Interval(1, np.inf, left_open=False, right_open=True)}

    def __new__(cls, *args, **kwargs):
        # Check if theta=1 is passed either as a positional arg or keyword arg
        theta_is_one = False

        if "theta" in kwargs and kwargs["theta"] == 1:
            theta_is_one = True

        if theta_is_one:
            # Return an IndependenceCopula instance
            new_kwargs = kwargs.copy()
            if "theta" in new_kwargs:
                del new_kwargs["theta"]
            return IndependenceCopula(**new_kwargs)

        # If theta is not 1, proceed with normal initialization
        return super().__new__(cls)

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
    def _pickands(self):
        return (self.t**self.theta + (1 - self.t) ** self.theta) ** (1 / self.theta)

    @property
    def cdf(self):
        base_expr = sympy.exp(
            -(
                (sympy.log(1/self.v)**self.theta + sympy.log(1/self.u)**self.theta)
                ** (1/self.theta)
            )
        )
        # When u==0 or v==0, return 0; otherwise, use the base expression.
        cdf_expr = sympy.Piecewise(
            (0, sympy.Or(sympy.Eq(self.u, 0), sympy.Eq(self.v, 0))),
            (base_expr, True)
        )
        return SymPyFuncWrapper(cdf_expr)

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

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return (self.theta - 1) / self.theta
