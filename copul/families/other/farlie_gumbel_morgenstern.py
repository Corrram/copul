import sympy

from copul.families.abstract_copula import AbstractCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class FarlieGumbelMorgenstern(AbstractCopula):
    theta = sympy.symbols("theta")
    params = [theta]
    intervals = {"theta": sympy.Interval(-1, 1, left_open=False, right_open=False)}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def cdf(self):
        u = self.u
        v = self.v
        cdf = u * v + self.theta * u * v * (1 - u) * (1 - v)
        return SymPyFunctionWrapper(cdf)

    def cond_distr_2(self, u=None, v=None):
        cd2 = self.u + self.theta * self.u * (1 - self.u) * (1 - 2 * self.v)
        return SymPyFunctionWrapper(cd2)(u, v)

    @property
    def pdf(self):
        return 1 + self.theta * (1 - 2 * self.u) * (1 - 2 * self.v)

    @property
    def spearmans_rho(self):
        return self.theta / 3

    @property
    def kendalls_tau(self):
        return 2 * self.theta / 9


B10 = FarlieGumbelMorgenstern
