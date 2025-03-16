import copy

import numpy as np
import sympy

from copul.families.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.families.extreme_value.galambos import Galambos
from copul.families.extreme_value.marshall_olkin import MarshallOlkin
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


class JoeEV(BivExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return self.alpha_1 == self.alpha_2

    alpha_1, alpha_2 = sympy.symbols("alpha_1 alpha_2", nonnegative=True)
    delta = sympy.symbols("delta", positive=True)
    params = [alpha_1, alpha_2, delta]

    intervals = {
        "alpha_1": sympy.Interval(0, 1, left_open=False, right_open=False),
        "alpha_2": sympy.Interval(0, 1, left_open=False, right_open=False),
        "delta": sympy.Interval(0, np.inf, left_open=True, right_open=True),
    }

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 3:
            self.alpha_1 = args[0]
            self.alpha_2 = args[1]
            self.delta = args[2]
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
            return Galambos(**kwargs)
        elif "alpha_1" in kwargs and kwargs["alpha_1"] == 1:
            del kwargs["alpha_1"]
            if self.alpha_2 == 1:
                if "alpha_2" in kwargs:
                    del kwargs["alpha_2"]
                return Galambos()(**kwargs)
            new_copula = copy.deepcopy(self)
            new_copula.alpha_1 = 1
            return new_copula(**kwargs)
        elif "alpha_2" in kwargs and kwargs["alpha_2"] == 1:
            del kwargs["alpha_2"]
            if self.alpha_1 == 1:
                if "alpha_1" in kwargs:
                    del kwargs["alpha_1"]
                return Galambos()(**kwargs)
            new_copula = copy.deepcopy(self)
            new_copula.alpha_2 = 1
            return new_copula(**kwargs)
        elif "alpha_1" in kwargs and kwargs["alpha_1"] == 0:
            del kwargs["alpha_1"]
            if "alpha_2" in kwargs:
                del kwargs["alpha_2"]
            if "delta" in kwargs:
                del kwargs["delta"]
            return BivIndependenceCopula()(**kwargs)
        elif "alpha_2" in kwargs and kwargs["alpha_2"] == 0:
            del kwargs["alpha_2"]
            if "alpha_1" in kwargs:
                del kwargs["alpha_1"]
            if "delta" in kwargs:
                del kwargs["delta"]
            return BivIndependenceCopula()(**kwargs)
        elif "delta" in kwargs and kwargs["delta"] == sympy.oo:
            del kwargs["delta"]
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
        elif "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return BivIndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        delta = self.delta
        t = self.t
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        return 1 - ((alpha_1 * (1 - t)) ** (-delta) + (alpha_2 * t) ** (-delta)) ** (
            -1 / delta
        )

    @property
    def cdf(self):
        cdf = (
            self.u
            * self.v
            * sympy.exp(
                (
                    (self.alpha_1 * sympy.log(1 / self.u)) ** (-self.delta)
                    + (self.alpha_2 * sympy.log(1 / self.v)) ** (-self.delta)
                )
                ** (-1 / self.delta)
            )
        )
        return CDFWrapper(cdf)

    # @property
    # def pdf(self):
    #     u = self.u
    #     v = self.v
    #     result = None
    #     return SymPyFunctionWrapper(result)
