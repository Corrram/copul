import scipy
import sympy
from statsmodels.distributions.copula.elliptical import StudentTCopula

from copul.families.elliptical.elliptical_copula import EllipticalCopula


class StudentT(EllipticalCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    rho = sympy.symbols("rho")
    nu = sympy.symbols("nu", positive=True)
    modified_bessel_function = sympy.Function("K")(nu)
    gamma_function = sympy.Function("gamma")(nu / 2)
    params = [nu, rho]
    intervals = {
        "nu": sympy.Interval(0, sympy.oo, left_open=True, right_open=True),
        "rho": sympy.Interval(-1, 1, left_open=False, right_open=False),
    }

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def rvs(self, n=1):
        return StudentTCopula(self.rho, df=self.nu).rvs(n)

    def cdf(self):
        pass

    def cond_distr_1(self):
        return lambda u, v: scipy.stats.t.cdf(
            scipy.stats.t.ppf(v, self.nu),
            self.nu,
            loc=self.rho * scipy.stats.t.ppf(u, self.nu),
            scale=(
                (1 - self.rho**2)
                * (self.nu + 1)
                / (self.nu + scipy.stats.t.ppf(u, self.nu) ** 2)
            )
            ** 0.5,
        )

    def cond_distr_2(self):
        return lambda u, v: scipy.stats.t.cdf(
            scipy.stats.t.ppf(u, self.nu),
            self.nu,
            loc=self.rho * scipy.stats.t.ppf(v, self.nu),
            scale=(
                (1 - self.rho**2)
                * (self.nu + 1)
                / (self.nu + scipy.stats.t.ppf(v, self.nu) ** 2)
            )
            ** 0.5,
        )

    def pdf(self):
        return lambda u, v: StudentTCopula(self.rho, df=self.nu).pdf([u, v])
