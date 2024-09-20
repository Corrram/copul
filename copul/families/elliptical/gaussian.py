import numpy as np
import sympy
from scipy.stats import norm
from statsmodels.distributions.copula.elliptical import GaussianCopula

from copul.families.elliptical.elliptical_copula import EllipticalCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Gaussian(EllipticalCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    rho = sympy.symbols("rho")

    generator = sympy.exp(-EllipticalCopula.t / 2)

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 1:
            kwargs["rho"] = args[0]
        if "rho" in kwargs:
            if kwargs["rho"] == -1:
                del kwargs["rho"]
                return LowerFrechet()(**kwargs)
            elif kwargs["rho"] == 0:
                del kwargs["rho"]
                return IndependenceCopula()(**kwargs)
            elif kwargs["rho"] == 1:
                del kwargs["rho"]
                return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def rvs(self, n=1):
        return GaussianCopula(self.rho).rvs(n)

    @property
    def cdf(self):
        cop = GaussianCopula(self.rho)

        def gauss_cdf(u, v):
            if u == 0 or v == 0:
                return sympy.S.Zero
            else:
                return sympy.S(cop.cdf([u, v]))

        return lambda u, v: SymPyFuncWrapper(gauss_cdf(u, v))

    def _conditional_distribution(self, u=None, v=None):
        scale = sympy.sqrt(1 - self.rho**2)

        def conditional_func(u_, v_):
            return norm.cdf(norm.ppf(v_), loc=self.rho * norm.ppf(u_), scale=scale)

        if u is None and v is None:
            return conditional_func
        elif u is not None and v is not None:
            return conditional_func(u, v)
        elif u is not None:
            return lambda v_: conditional_func(u, v_)
        else:
            return lambda u_: conditional_func(u_, v)

    def cond_distr_1(self, u=None, v=None):
        if v in [0, 1]:
            return SymPyFuncWrapper(sympy.Number(v))
        return SymPyFuncWrapper(sympy.Number(self._conditional_distribution(u, v)))

    def cond_distr_2(self, u=None, v=None):
        if u in [0, 1]:
            return SymPyFuncWrapper(sympy.Number(u))
        return SymPyFuncWrapper(sympy.Number(self._conditional_distribution(v, u)))

    @property
    def pdf(self):
        return lambda u, v: SymPyFuncWrapper(
            sympy.Number(GaussianCopula(self.rho).pdf([u, v]))
        )

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return 3 / np.pi * np.arcsin(1 / 2 + self.rho**2 / 2) - 0.5

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return 6 / np.pi * np.arcsin(self.rho / 2)

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return 2 / np.pi * np.arcsin(self.rho)
