import sympy

from copul.families.other.frechet import Frechet


class UpperFrechet(Frechet):
    _alpha = 1
    _beta = 0
    t = sympy.symbols("t", positive=True)

    @property
    def alpha(self):
        return 1

    @property
    def beta(self):
        return 0

    @property
    def pickands(self):
        return sympy.Max(self.t, 1 - self.t)
