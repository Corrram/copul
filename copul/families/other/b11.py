import sympy

from copul.families.abstract_copula import AbstractCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.upper_frechet import UpperFrechet
from copul.sympy_wrapper import SymPyFunctionWrapper


class B11(AbstractCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def is_absolutely_continuous(self) -> bool:
        return self.delta < 1

    # special case of the Frechet copula family
    delta = sympy.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, 1, left_open=False, right_open=False)}

    def __call__(self, **kwargs):
        if "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return IndependenceCopula()(**kwargs)
        if "delta" in kwargs and kwargs["delta"] == 1:
            del kwargs["delta"]
            return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def cdf(self):
        cdf = (
            self.delta * sympy.Min(self.u, self.v) + (1 - self.delta) * self.u * self.v
        )
        return SymPyFunctionWrapper(cdf)
