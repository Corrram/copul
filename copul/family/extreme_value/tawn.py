import copy

import numpy as np
import sympy

from copul.family.extreme_value import GumbelHougaardEV as GumbelHougaard
from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.family.extreme_value.marshall_olkin import MarshallOlkin
from copul.family.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


class Tawn(BivExtremeValueCopula):
    def __new__(cls, *args, **kwargs):
        # Handle special cases during initialization
        if len(args) == 3:
            # Check for GumbelHougaard special case
            if args[0] == 1 and args[1] == 1:
                return GumbelHougaard(args[2])
            # Check for Independence special case
            if args[2] == 1:
                return BivIndependenceCopula()
            # Check for MarshallOlkin special case
            if args[2] == sympy.oo:
                return MarshallOlkin(args[0], args[1])

        # Handle keyword arguments
        if len(args) == 0:
            if "alpha_1" in kwargs and "alpha_2" in kwargs:
                if (
                    kwargs["alpha_1"] == 1
                    and kwargs["alpha_2"] == 1
                    and "theta" in kwargs
                ):
                    # GumbelHougaard special case
                    theta = kwargs.pop("theta")
                    # Remove alpha params
                    kwargs.pop("alpha_1")
                    kwargs.pop("alpha_2")
                    return GumbelHougaard(theta)(**kwargs)

            if "theta" in kwargs:
                if kwargs["theta"] == 1:
                    # Independence special case
                    return BivIndependenceCopula()(**kwargs)
                elif kwargs["theta"] == sympy.oo:
                    # MarshallOlkin special case
                    alpha_1 = kwargs.pop("alpha_1", cls.alpha_1)
                    alpha_2 = kwargs.pop("alpha_2", cls.alpha_2)
                    kwargs.pop("theta")
                    return MarshallOlkin(alpha_1, alpha_2)(**kwargs)

        # Default case - proceed with normal initialization
        return super().__new__(cls)

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
        # Handle positional arguments
        if len(args) == 3:
            self.alpha_1 = args[0]
            self.alpha_2 = args[1]
            self.theta = args[2]
        # Raise error only if there are some args but not exactly 3
        elif len(args) > 0:
            raise ValueError("Tawn copula requires three parameters")

        # Handle special cases with keyword arguments
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
            return BivIndependenceCopula()(**kwargs)
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
    def _pickands(self):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        t = self.t
        theta = self.theta
        return (
            (1 - alpha_1) * (1 - t)
            + (1 - alpha_2) * t
            + ((alpha_1 * (1 - t)) ** theta + (alpha_2 * t) ** theta) ** (1 / theta)
        )

    @property
    def cdf(self):
        theta = self.theta
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        u = self.u
        v = self.v
        cdf = (
            u ** (1 - alpha_1)
            * v ** (1 - alpha_2)
            * sympy.exp(
                -(
                    (
                        (alpha_1 * sympy.log(1 / u)) ** theta
                        + (alpha_2 * sympy.log(1 / v)) ** theta
                    )
                    ** (1 / theta)
                )
            )
        )
        return CDFWrapper(cdf)

    # @property
    # def pdf(self):
    #     u = self.u
    #     v = self.v
    #     result = None
    #     return SymPyFunctionWrapper(result)
