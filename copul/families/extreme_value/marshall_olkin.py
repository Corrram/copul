import logging

import sympy

from copul.exceptions import PropertyUnavailableException
from copul.families.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

log = logging.getLogger(__name__)


class MarshallOlkin(BivExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return self.alpha_1 == self.alpha_2

    _alpha_1, _alpha_2 = sympy.symbols("alpha_1 alpha_2", nonnegative=True)
    params = [_alpha_1, _alpha_2]

    intervals = {
        "alpha_1": sympy.Interval(0, 1, left_open=False, right_open=False),
        "alpha_2": sympy.Interval(0, 1, left_open=False, right_open=False),
    }

    @property
    def is_absolutely_continuous(self):
        return (self._alpha_1 == 0) | (self._alpha_2 == 0)

    @property
    def alpha_1(self):
        if isinstance(self._alpha_1, property):
            return self._alpha_1.fget(self)
        return self._alpha_1

    @alpha_1.setter
    def alpha_1(self, value):
        self._alpha_1 = value

    @property
    def alpha_2(self):
        if isinstance(self._alpha_2, property):
            return self._alpha_2.fget(self)
        return self._alpha_2

    @alpha_2.setter
    def alpha_2(self, value):
        self._alpha_2 = value

    @property
    def _pickands(self):
        return sympy.Max(1 - self.alpha_1 * (1 - self.t), 1 - self.alpha_2 * self.t)

    @property
    def cdf(self):
        if self.alpha_1 == self.alpha_2 == 0:
            return SymPyFuncWrapper(self.u * self.v)
        arg1 = self.v * self.u ** (1 - self.alpha_1)
        arg2 = self.u * self.v ** (1 - self.alpha_2)
        cdf = sympy.Min(arg1, arg2)
        return SymPyFuncWrapper(cdf)

    def cond_distr_1(self, u=None, v=None):
        alpha_1 = self.alpha_1
        alpha2 = self.alpha_2
        heavy_expr = self.u * self.v ** (1 - alpha2) - self.u ** (1 - alpha_1) * self.v
        cd1 = (
            self.u * self.v ** (1 - alpha2) * sympy.Heaviside(-heavy_expr)
            - self.u ** (1 - alpha_1)
            * self.v
            * (alpha_1 - 1)
            * sympy.Heaviside(heavy_expr)
        ) / self.u
        return CD1Wrapper(cd1)(u, v)

    def cond_distr_2(self, u=None, v=None):
        alpha1 = self.alpha_1
        alpha2 = self.alpha_2
        heavy_expr = -self.u * self.v ** (1 - alpha2) + self.u ** (1 - alpha1) * self.v
        cond_distr = (
            self.u * self.v ** (1 - alpha2) * (1 - alpha2) * sympy.Heaviside(heavy_expr)
            + self.u ** (1 - alpha1) * self.v * sympy.Heaviside(-heavy_expr)
        ) / self.v
        return CD2Wrapper(cond_distr)(u, v)

    def _squared_cond_distr_1(self, u, v):
        alpha1 = self.alpha_1
        alpha2 = self.alpha_2
        return (
            u
            * v ** (1 - alpha2)
            * sympy.Heaviside(-u * v ** (1 - alpha2) + u ** (1 - alpha1) * v)
            - u ** (1 - alpha1)
            * v
            * (alpha1 - 1)
            * sympy.Heaviside(u * v ** (1 - alpha2) - u ** (1 - alpha1) * v)
        ) ** 2 / u**2

    def _xi_int_1(self, v):
        u = self.u
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        integrand_1 = (u * v ** (1 - alpha_2)) ** 2 / u**2
        integrand_2 = (u ** (1 - alpha_1) * v * (alpha_1 - 1)) ** 2 / u**2
        log.debug(sympy.latex(sympy.simplify(integrand_1)))
        log.debug(sympy.latex(sympy.simplify(integrand_2)))
        int_1 = sympy.simplify(
            sympy.integrate(integrand_1, (u, 0, v ** (alpha_2 / alpha_1)))
        )
        int_2 = sympy.simplify(
            sympy.integrate(integrand_2, (u, v ** (alpha_2 / alpha_1), 1))
        )
        int_2 = sympy.simplify(int_2)
        log.debug(sympy.latex(int_1))
        log.debug(sympy.latex(int_2))
        return sympy.simplify(int_1 + int_2)

    @property
    def pdf(self):
        raise PropertyUnavailableException("Marshall-Olkin copula does not have a pdf")

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # Handle edge case for alpha_1 = alpha_2 = 0
        if self.alpha_1 == 0 and self.alpha_2 == 0:
            return 0

        # Original formula
        result = (
            3
            * self.alpha_1
            * self.alpha_2
            / (2 * self.alpha_1 - self.alpha_1 * self.alpha_2 + 2 * self.alpha_2)
        )
        return result

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # Handle edge cases
        if self.alpha_1 == 0 and self.alpha_2 == 0:
            return 0
        if self.alpha_1 == 1 and self.alpha_2 == 1:
            return 1

        # Original formula
        result = (
            self.alpha_1
            * self.alpha_2
            / (self.alpha_1 - self.alpha_1 * self.alpha_2 + self.alpha_2)
        )
        return result

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # Handle edge case for alpha_1 = alpha_2 = 0
        if self.alpha_1 == 0 and self.alpha_2 == 0:
            return 0

        # Original formula
        result = (
            2
            * self.alpha_1**2
            * self.alpha_2
            / (3 * self.alpha_1 + self.alpha_2 - 2 * self.alpha_1 * self.alpha_2)
        )
        return result


def MarshallOlkinDiag():
    """Creates a Marshall-Olkin copula with alpha_1 = alpha_2"""
    copula = MarshallOlkin()
    # Using the correct parameter name alpha_2 instead of alpha2
    copula.alpha_2 = copula.alpha_1
    return copula
