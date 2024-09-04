import copy

import sympy

from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.exceptions import PropertyUnavailableException
from copul.families.bivcopula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class Frechet(BivCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    _alpha, _beta = sympy.symbols("alpha beta", nonnegative=True)
    params = [_alpha, _beta]
    intervals = {
        "alpha": sympy.Interval(0, 1, left_open=False, right_open=False),
        "beta": sympy.Interval(0, 1, left_open=False, right_open=False),
    }

    @property
    def is_absolutely_continuous(self) -> bool:
        return (self.alpha == 0) & (self.beta == 0)

    @property
    def alpha(self):
        if isinstance(self._alpha, property):
            return self._alpha.fget(self)
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        if isinstance(self._beta, property):
            return self._beta.fget(self)
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    def __init__(self, *args, **kwargs):
        if args and len(args) == 2:
            kwargs["alpha"] = args[0]
            kwargs["beta"] = args[1]
        if "alpha" in kwargs:
            self._alpha = kwargs["alpha"]
            self.intervals["beta"] = sympy.Interval(
                0, 1 - self.alpha, left_open=False, right_open=False
            )
            self.params = [param for param in self.params if str(param) != "alpha"]
            del kwargs["alpha"]
        if "beta" in kwargs:
            self._beta = kwargs["beta"]
            self.intervals["alpha"] = sympy.Interval(
                0, 1 - self.beta, left_open=False, right_open=False
            )
            self.params = [param for param in self.params if str(param) != "beta"]
            del kwargs["beta"]
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 2:
            kwargs["alpha"] = args[0]
            kwargs["beta"] = args[1]
        if "alpha" in kwargs:
            new_copula = copy.deepcopy(self)
            new_copula._alpha = kwargs["alpha"]
            new_copula.intervals["beta"] = sympy.Interval(
                0, 1 - new_copula.alpha, left_open=False, right_open=False
            )
            new_copula.params = [
                param for param in new_copula.params if param != self._alpha
            ]
            del kwargs["alpha"]
            return new_copula.__call__(**kwargs)
        if "beta" in kwargs:
            new_copula = copy.deepcopy(self)
            new_copula._beta = kwargs["beta"]
            new_copula.intervals["alpha"] = sympy.Interval(
                0, 1 - new_copula.beta, left_open=False, right_open=False
            )
            new_copula.params = [
                param for param in new_copula.params if param != self._beta
            ]
            del kwargs["beta"]
            return new_copula.__call__(**kwargs)
        return super().__call__(**kwargs)

    @property
    def cdf(self):
        frechet_upper = sympy.Min(self.u, self.v)
        frechet_lower = sympy.Max(self.u + self.v - 1, 0)
        cdf = (
            self._alpha * frechet_upper
            + (1 - self._alpha - self._beta) * self.u * self.v
            + self._beta * frechet_lower
        )
        return CDFWrapper(cdf)

    def cond_distr_2(self, u=None, v=None):
        cond_distr = (
            self._alpha * sympy.Heaviside(self.u - self.v)
            + self._beta * sympy.Heaviside(self.u + self.v - 1)
            + self.u * (-self._alpha - self._beta + 1)
        )
        return SymPyFunctionWrapper(cond_distr)(u, v)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return self._alpha - self._beta

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return (self._alpha - self._beta) * (2 + self._alpha + self._beta) / 3

    @property
    def lambda_L(self):
        return self._alpha

    @property
    def lambda_U(self):
        return self._alpha

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return (self.alpha - self.beta) ** 2 + self.alpha * self.beta

    @property
    def pdf(self):
        raise PropertyUnavailableException("Frechet copula does not have a pdf")


# B11 = lambda: Frechet(beta=0)
