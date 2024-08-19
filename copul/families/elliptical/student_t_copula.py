import sympy
from statsmodels.distributions.copula.elliptical import StudentTCopula
from scipy.stats import t as student_t
from scipy.stats import multivariate_t

from copul.families.other import LowerFrechet, UpperFrechet
from copul.families.elliptical.elliptical_copula import EllipticalCopula


class StudentT(EllipticalCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    rho = sympy.symbols("rho")
    nu = sympy.symbols("nu", positive=True)
    modified_bessel_function = sympy.Function("K")(nu)
    gamma_function = sympy.Function("gamma")(nu / 2)
    params = [rho, nu]
    intervals = {
        "rho": sympy.Interval(-1, 1, left_open=False, right_open=False),
        "nu": sympy.Interval(0, sympy.oo, left_open=True, right_open=True),
    }

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 1:
            kwargs["rho"] = args[0]
        if args is not None and len(args) == 2:
            kwargs["rho"] = args[0]
            kwargs["nu"] = args[1]
        if "rho" in kwargs:
            if kwargs["rho"] == -1:
                del kwargs["rho"]
                return LowerFrechet()(**kwargs)
            elif kwargs["rho"] == 1:
                del kwargs["rho"]
                return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def rvs(self, n=1):
        return StudentTCopula(self.rho, df=self.nu).rvs(n)

    @property
    def cdf(self):
        copula = multivariate_t(df=self.nu, shape=[[1, self.rho], [self.rho, 1]])

        def student_t_copula_cdf(u, v):
            # Transform the uniform marginals to the t-distribution's quantiles
            z_u = student_t.ppf(u, self.nu)
            z_v = student_t.ppf(v, self.nu)
            # Calculate the bivariate Student's t CDF
            return copula.cdf([z_u, z_v])

        return lambda u, v: student_t_copula_cdf(u, v)

    def _conditional_distribution(self, u, v):
        def conditional_func(primary, secondary):
            return student_t.cdf(
                student_t.ppf(secondary, self.nu),
                self.nu,
                loc=self.rho * student_t.ppf(primary, self.nu),
                scale=(
                    (1 - self.rho**2)
                    * (self.nu + 1)
                    / (self.nu + student_t.ppf(primary, self.nu) ** 2)
                )
                ** 0.5,
            )(u, v)

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
            return v
        return self._conditional_distribution(u, v)

    def cond_distr_2(self, u=None, v=None):
        if u in [0, 1]:
            return u
        return self._conditional_distribution(v, u)

    def pdf(self):
        return lambda u, v: StudentTCopula(self.rho, df=self.nu).pdf([u, v])
