import sympy
from copul.families.other.frechet import Frechet


class IndependenceCopula(Frechet):
    _alpha = 0
    _beta = 0

    @property
    def alpha(self):
        return 0

    @property
    def beta(self):
        return 0

    @property
    def pickand(self):
        return sympy.Max(1)
