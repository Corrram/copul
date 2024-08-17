import copy

import numpy as np
import sympy

from copul.families.extreme_value import GumbelHougaard
from copul.families.extreme_value.marshall_olkin import MarshallOlkin

from copul.families.other.independence_copula import IndependenceCopula
from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class Tawn(ExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return self.alpha_1 == self.alpha_2

    alpha_1, alpha_2 = sympy.symbols("alpha_1 alpha_2", nonnegative=True)
    theta = sympy.symbols("theta", positive=True)
    params = [alpha_1, alpha_2, theta]
    intervals = {
        "alpha_1": sympy.Interval(0, 1, left_open=False, right_open=False),
        "alpha_2": sympy.Interval(0, 1, left_open=False, right_open=False),
        "theta": sympy.Interval(1, np.inf, left_open=False, right_open=True),
    }

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 3:
            self.alpha_1 = args[0]
            self.alpha_2 = args[1]
            self.theta = args[2]
        elif args is not None:
            raise ValueError("Tawn copula requires three parameters")
        if (
            "alpha_1" in kwargs
            and kwargs["alpha_1"] == 1
            and "alpha_2" in kwargs
            and kwargs["alpha_2"] == 1
        ):
            del kwargs["alpha_1"]
            del kwargs["alpha_2"]
            return GumbelHougaard(**kwargs)
        elif "alpha_1" in kwargs and kwargs["alpha_1"] == 1:
            del kwargs["alpha_1"]
            if self.alpha_2 == 1:
                if "alpha_2" in kwargs:
                    del kwargs["alpha_2"]
                return GumbelHougaard(**kwargs)
            new_copula = copy.deepcopy(self)
            new_copula.alpha_1 = 1
            return new_copula(**kwargs)
        elif "alpha_2" in kwargs and kwargs["alpha_2"] == 1:
            del kwargs["alpha_2"]
            if self.alpha_1 == 1:
                if "alpha_1" in kwargs:
                    del kwargs["alpha_1"]
                return GumbelHougaard(**kwargs)
            new_copula = copy.deepcopy(self)
            new_copula.alpha_2 = 1
            return new_copula(**kwargs)
        elif "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            if "alpha_1" in kwargs:
                del kwargs["alpha_1"]
            if "alpha_2" in kwargs:
                del kwargs["alpha_2"]
            return IndependenceCopula()(**kwargs)
        elif "theta" in kwargs and kwargs["theta"] == sympy.oo:
            del kwargs["theta"]
            if "alpha_1" in kwargs:
                alpha1 = kwargs["alpha_1"]
                del kwargs["alpha_1"]
            else:
                alpha1 = self.alpha_1
            if "alpha_2" in kwargs:
                alpha2 = kwargs["alpha_2"]
                del kwargs["alpha_2"]
            else:
                alpha2 = self.alpha_2
            return MarshallOlkin(**kwargs)(alpha_1=alpha1, alpha_2=alpha2)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def pickands(self):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        t = self.t
        theta = self.theta
        func = (
            (1 - alpha_1) * (1 - t)
            + (1 - alpha_2) * t
            + ((alpha_1 * (1 - t)) ** theta + (alpha_2 * t) ** theta) ** (1 / theta)
        )
        return SymPyFunctionWrapper(func)

    @property
    def cdf(self):
        cdf = (
            self.u ** (1 - self.alpha_1)
            * self.v ** (1 - self.alpha_2)
            * sympy.exp(
                -(
                    (
                        (self.alpha_1 * sympy.log(1 / self.u)) ** self.theta
                        + (self.alpha_2 * sympy.log(1 / self.v)) ** self.theta
                    )
                    ** (1 / self.theta)
                )
            )
        )
        return SymPyFunctionWrapper(cdf)
