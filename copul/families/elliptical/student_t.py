import sympy
from scipy.stats import multivariate_t
from scipy.stats import t as student_t
from statsmodels.distributions.copula.elliptical import StudentTCopula
from copul.families.elliptical.elliptical_copula import EllipticalCopula
from copul.families.other import LowerFrechet, UpperFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class StudentT(EllipticalCopula):
    """
    Student's t Copula implementation.

    The Student's t copula is an elliptical copula derived from the multivariate t-distribution.
    It is characterized by a correlation parameter rho in [-1, 1] and a degrees of freedom
    parameter nu > 0.

    Special cases:
    - rho = -1: Lower Fréchet bound (countermonotonicity)
    - rho = 1: Upper Fréchet bound (comonotonicity)
    - nu → ∞: Approaches the Gaussian copula
    """

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
            # Handle special cases
            if kwargs["rho"] == -1:
                # Don't pass 'nu' parameter to LowerFrechet
                new_kwargs = kwargs.copy()
                if "nu" in new_kwargs:
                    del new_kwargs["nu"]
                if "rho" in new_kwargs:
                    del new_kwargs["rho"]
                return LowerFrechet()(**new_kwargs)
            elif kwargs["rho"] == 1:
                # Don't pass 'nu' parameter to UpperFrechet
                new_kwargs = kwargs.copy()
                if "nu" in new_kwargs:
                    del new_kwargs["nu"]
                if "rho" in new_kwargs:
                    del new_kwargs["rho"]
                return UpperFrechet()(**new_kwargs)

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def rvs(self, n=1):
        """
        Generate random samples from the Student's t copula.

        Args:
            n (int): Number of samples to generate

        Returns:
            numpy.ndarray: Array of shape (n, 2) containing the samples
        """
        return StudentTCopula(self.rho, df=self.nu).rvs(n)

    def _calculate_student_t_cdf(self, u, v, rho_val, nu_val):
        """Calculate Student's t CDF at point (u, v)."""
        mvt = multivariate_t(df=nu_val, shape=[[1, rho_val], [rho_val, 1]])
        z_u = student_t.ppf(u, nu_val)
        z_v = student_t.ppf(v, nu_val)
        return mvt.cdf([z_u, z_v])

    @property
    def cdf(self):
        """
        Compute the cumulative distribution function of the Student's t copula.

        Returns:
            callable: Function that computes the CDF at given points
        """
        # Store the parameters to avoid capturing 'self' in the lambda
        rho_val = self.rho
        nu_val = self.nu

        # Use a reference to the method, not self
        cdf_calc = self._calculate_student_t_cdf

        def student_t_copula_cdf(u, v):
            return cdf_calc(u, v, rho_val, nu_val)

        return lambda u, v: CDFWrapper(sympy.S(student_t_copula_cdf(u, v)))

    def _conditional_distribution(self, u, v):
        """
        Compute the conditional distribution function of the Student's t copula.

        Args:
            u (float, optional): First marginal value
            v (float, optional): Second marginal value

        Returns:
            callable or sympy.Expr: Conditional distribution function or value
        """

        def conditional_func(primary, secondary):
            cdf = student_t.cdf(
                student_t.ppf(secondary, self.nu),
                self.nu,
                loc=self.rho * student_t.ppf(primary, self.nu),
                scale=(
                    (1 - self.rho**2)
                    * (self.nu + 1)
                    / (self.nu + student_t.ppf(primary, self.nu) ** 2)
                )
                ** 0.5,
            )
            if isinstance(cdf, float):
                return sympy.S(cdf)
            return sympy.S(cdf(u, v))

        if u is None and v is None:
            return conditional_func
        elif u is not None and v is not None:
            return conditional_func(u, v)
        elif u is not None:
            return lambda v_: conditional_func(u, v_)
        else:
            return lambda u_: conditional_func(u_, v)

    def cond_distr_1(self, u=None, v=None):
        """
        Compute the first conditional distribution C(v|u).

        Args:
            u (float, optional): Conditioning value
            v (float, optional): Value at which to evaluate

        Returns:
            CD1Wrapper: Wrapped conditional distribution function or value
        """
        if v in [0, 1]:
            return CD1Wrapper(sympy.S(v))
        cd1 = self._conditional_distribution(u, v)
        return CD1Wrapper(cd1)

    def cond_distr_2(self, u=None, v=None):
        """
        Compute the second conditional distribution C(u|v).

        Args:
            u (float, optional): Value at which to evaluate
            v (float, optional): Conditioning value

        Returns:
            CD2Wrapper: Wrapped conditional distribution function or value
        """
        if u in [0, 1]:
            return CD2Wrapper(sympy.S(u))
        cd2 = self._conditional_distribution(v, u)
        return CD2Wrapper(cd2)

    @property
    def pdf(self):
        """
        Compute the probability density function of the Student's t copula.

        Returns:
            callable: Function that computes the PDF at given points
        """
        return lambda u, v: SymPyFuncWrapper(
            sympy.S(StudentTCopula(self.rho, df=self.nu).pdf([u, v]))
        )
