import copy

import numpy as np
import sympy
from sympy import stats

from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.families.other.independence_copula import IndependenceCopula


class HueslerReiss(ExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, np.inf, left_open=False, right_open=True)}

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            self.delta = args[0]
        if "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return IndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def pickands(self):
        std_norm = stats.cdf(stats.Normal("x", 0, 1))
        return (1 - self.t) * std_norm(self._z(1 - self.t)) + self.t * std_norm(
            self._z(self.t)
        )

    def _z(self, t):
        if t == 0:
            return 0
        elif t == 1:
            return 1
        return 1 / self.delta + self.delta / 2 * sympy.ln(t / (1 - t))


B8 = HueslerReiss
