import sympy
from scipy.stats import norm
from statsmodels.distributions.copula.elliptical import GaussianCopula

from copul.families.elliptical.elliptical_copula import EllipticalCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet


class Gaussian(EllipticalCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    rho = sympy.symbols("rho")

    generator = sympy.exp(-EllipticalCopula.t / 2)

    def __call__(self, **kwargs):
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

    def cdf(self):
        return lambda u, v: GaussianCopula(self.rho).cdf([u, v])

    def cond_distr_1(self):
        scale = sympy.sqrt(1 - self.rho**2)
        return lambda u, v: norm.cdf(
            norm.ppf(v), loc=self.rho * norm.ppf(u), scale=scale
        )

    def cond_distr_2(self):
        scale = sympy.sqrt(1 - self.rho**2)
        return lambda u, v: norm.cdf(
            norm.ppf(u), loc=self.rho * norm.ppf(v), scale=scale
        )

    def pdf(self):
        return lambda u, v: GaussianCopula(self.rho).pdf([u, v])


B1 = Gaussian
