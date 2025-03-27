import numpy as np
import sympy

from copul.families.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.families.extreme_value.galambos import Galambos
from copul.families.extreme_value.gumbel_hougaard import (
    GumbelHougaardEV as GumbelHougaard,
)
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.cdf_wrapper import CDFWrapper


class BB5(BivExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    theta, delta = sympy.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sympy.Interval(1, np.inf, left_open=False, right_open=True),
        "delta": sympy.Interval(0, np.inf, left_open=True, right_open=True),
    }

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 2:
            self.theta = args[0]
            self.delta = args[1]
        elif args is not None:
            raise ValueError("BB5 copula requires two parameters")
        if "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            return Galambos(delta=self.delta)(**kwargs)
        elif "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return GumbelHougaard(self.theta)(**kwargs)
        elif "delta" in kwargs and kwargs["delta"] == sympy.oo:
            del kwargs["delta"]
            return UpperFrechet(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        theta = self.theta
        t = self.t
        return (
            t**theta
            + (1 - t) ** theta
            - ((1 - t) ** (-theta * self.delta) + t ** (-theta * self.delta))
            ** (-1 / self.delta)
        ) ** (1 / theta)

    @property
    def cdf(self):
        theta = self.theta
        u = self.u
        v = self.v
        delta = self.delta
        cdf = sympy.exp(
            -(
                (
                    sympy.log(1 / v) ** theta
                    + sympy.log(1 / u) ** theta
                    - (
                        sympy.log(1 / u) ** (-delta * theta)
                        + sympy.log(1 / v) ** (-delta * theta)
                    )
                    ** (-1 / delta)
                )
                ** (1 / theta)
            )
        )
        return CDFWrapper(cdf)

    # @property
    # def pdf(self):
    #     u = self.u
    #     v = self.v
    #     result = None
    #     return SymPyFunctionWrapper(result)
