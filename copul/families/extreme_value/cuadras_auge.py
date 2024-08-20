import sympy as sp
import logging

from copul.cd1_wrapper import CD1Wrapper
from copul.exceptions import PropertyUnavailableException
from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.upper_frechet import UpperFrechet
from copul.sympy_wrapper import SymPyFunctionWrapper
from sympy import Min, Heaviside, DiracDelta

log = logging.getLogger(__name__)


class CuadrasAuge(ExtremeValueCopula):
    """
    Cuadras-Auge copula, special case of the Marshall-Olkin copula.
    """

    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sp.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sp.Interval(0, 1, left_open=False, right_open=False)}

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            self.delta = args[0]
        if "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return IndependenceCopula()(**kwargs)
        if "delta" in kwargs and kwargs["delta"] == 1:
            del kwargs["delta"]
            return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self):
        return self.delta == 0

    @property
    def pickands(self):
        func = 1 - self.delta * sp.Min(1 - self.t, self.t)
        return SymPyFunctionWrapper(func)

    @property
    def cdf(self):
        cdf = sp.Min(self.u, self.v) ** self.delta * (self.u * self.v) ** (
            1 - self.delta
        )
        return SymPyFunctionWrapper(cdf)

    @property
    def pdf(self):
        raise PropertyUnavailableException("Cuadras-Auge copula does not have a pdf")

    def cond_distr_1(self, u=None, v=None):
        delta = self.delta
        cond_distr_1 = (
            self.v ** (1 - delta)
            * (
                delta * self.u * sp.Heaviside(-self.u + self.v)
                - delta * sp.Min(self.u, self.v)
                + sp.Min(self.u, self.v)
            )
            * sp.Min(self.u, self.v) ** (delta - 1)
            / self.u**delta
        )
        return CD1Wrapper(cond_distr_1)(u, v)

    def _squared_cond_distr_1(self, v, u):
        delta = self.delta
        func = (
            (u * v) ** (2 - 2 * delta)
            * (delta * u * sp.Heaviside(-u + v) - (delta - 1) * sp.Min(u, v)) ** 2
            * sp.Min(u, v) ** (2 * delta - 2)
            / u**2
        )
        return sp.simplify(func)

    def _xi_int_1(self, v):
        delta = self.delta
        u = self.u
        func_u_lower_v = (
            (u * v) ** (2 - 2 * delta)
            * (delta * u - (delta - 1) * u) ** 2
            * u ** (2 * delta - 2)
            / u**2
        )
        func_u_greater_v = (delta - 1) ** 2 * v**2 / u ** (2 * delta)
        int1 = sp.simplify(sp.integrate(func_u_lower_v, (u, 0, v)))
        # int2 = sp.simplify(sp.integrate(func_u_greater_v, (u, v, 1)))
        int2 = sp.integrate(func_u_greater_v, (u, v, 1))
        # int2 = -v**2*v**(1 - 2*delta)*(delta - 1)**2/(1 - 2*delta) + v**2*(delta - 1)**2/(1 - 2*delta)
        log.debug("sub int1 sp: ", int1)
        log.debug("sub int1: ", sp.latex(int1))
        log.debug("sub int2 sp: ", int2)
        log.debug("sub int2: ", sp.latex(int2))
        return sp.simplify(int1 + int2)

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return self.delta**2 / (2 - self.delta)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return 3 * self.delta / (4 - self.delta)

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return self.delta / (2 - self.delta)


# B12 = CuadrasAuge
