import sympy

from copul.families.copula import Copula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class FarlieGumbelMorgenstern(Copula):
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
        result = 1 + self.theta * (1 - 2 * self.u) * (1 - 2 * self.v)
        return SymPyFunctionWrapper(result)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return self.theta / 3

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return 2 * self.theta / 9


# B10 = FarlieGumbelMorgenstern
