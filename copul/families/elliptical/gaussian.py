import numpy as np
import sympy
from scipy.stats import norm
from statsmodels.distributions.copula.elliptical import (
    GaussianCopula as StatsGaussianCopula,
)
from copul.families.elliptical.elliptical_copula import EllipticalCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Gaussian(EllipticalCopula):
    """
    Gaussian copula implementation.

    The Gaussian copula is an elliptical copula based on the multivariate normal distribution.
    It is characterized by a correlation parameter rho in [-1, 1].

    Special cases:
    - rho = -1: Lower Fréchet bound (countermonotonicity)
    - rho = 0: Independence copula
    - rho = 1: Upper Fréchet bound (comonotonicity)
    """

    @property
    def is_symmetric(self) -> bool:
        return True

    rho = sympy.symbols("rho")
    # Define generator as a symbolic expression with 't' as the variable
    t = sympy.symbols("t", positive=True)
    generator = sympy.exp(-t / 2)

    def __new__(cls, *args, **kwargs):
        # Handle special cases during initialization with positional args
        if len(args) == 1:
            if args[0] == -1:
                return LowerFrechet()
            elif args[0] == 0:
                return IndependenceCopula()
            elif args[0] == 1:
                return UpperFrechet()

        # Default case - proceed with normal initialization
        return super().__new__(cls)

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
        """
        Generate random samples from the Gaussian copula.

        Args:
            n (int): Number of samples to generate

        Returns:
            numpy.ndarray: Array of shape (n, 2) containing the samples
        """
        return StatsGaussianCopula(self.rho).rvs(n)

    @property
    def cdf(self):
        """
        Compute the cumulative distribution function of the Gaussian copula.

        Returns:
            callable: Function that computes the CDF at given points
        """
        cop = StatsGaussianCopula(self.rho)

        def gauss_cdf(u, v):
            if u == 0 or v == 0:
                return sympy.S.Zero
            else:
                return sympy.S(cop.cdf([u, v]))

        return lambda u, v: SymPyFuncWrapper(gauss_cdf(u, v))

    def _conditional_distribution(self, u=None, v=None):
        """
        Compute the conditional distribution function of the Gaussian copula.

        Args:
            u (float, optional): First marginal value
            v (float, optional): Second marginal value

        Returns:
            callable or float: Conditional distribution function or value
        """
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
        """
        Compute the first conditional distribution C(v|u).

        Args:
            u (float, optional): Conditioning value
            v (float, optional): Value at which to evaluate

        Returns:
            SymPyFuncWrapper: Wrapped conditional distribution function or value
        """
        if v in [0, 1]:
            return SymPyFuncWrapper(sympy.Number(v))
        return SymPyFuncWrapper(sympy.Number(self._conditional_distribution(u, v)))

    def cond_distr_2(self, u=None, v=None):
        """
        Compute the second conditional distribution C(u|v).

        Args:
            u (float, optional): Value at which to evaluate
            v (float, optional): Conditioning value

        Returns:
            SymPyFuncWrapper: Wrapped conditional distribution function or value
        """
        if u in [0, 1]:
            return SymPyFuncWrapper(sympy.Number(u))
        return SymPyFuncWrapper(sympy.Number(self._conditional_distribution(v, u)))

    @property
    def pdf(self):
        """
        Compute the probability density function of the Gaussian copula.

        Returns:
            callable: Function that computes the PDF at given points
        """
        return lambda u, v: SymPyFuncWrapper(
            sympy.Number(StatsGaussianCopula(self.rho).pdf([u, v]))
        )

    def characteristic_function(self, t1, t2):
        """
        Compute the characteristic function of the Gaussian copula.

        For the Gaussian copula, this evaluates the generator expression
        at the appropriate argument value.

        Args:
            t1 (float or sympy.Symbol): First argument
            t2 (float or sympy.Symbol): Second argument

        Returns:
            sympy.Expr: Value of the characteristic function
        """
        # Calculate the argument for the generator
        arg = (
            t1**2 * self.corr_matrix[0, 0]
            + t2**2 * self.corr_matrix[1, 1]
            + 2 * t1 * t2 * self.corr_matrix[0, 1]
        )

        # Substitute the argument into the generator expression
        return self.generator.subs(self.t, arg)

    def chatterjees_xi(self, *args, **kwargs):
        """
        Compute Chatterjee's xi measure of dependence.

        Returns:
            float: Chatterjee's xi value
        """
        self._set_params(args, kwargs)
        return 3 / np.pi * np.arcsin(1 / 2 + self.rho**2 / 2) - 0.5

    def spearmans_rho(self, *args, **kwargs):
        """
        Compute Spearman's rho rank correlation.

        Returns:
            float: Spearman's rho value
        """
        self._set_params(args, kwargs)
        return 6 / np.pi * np.arcsin(self.rho / 2)

    def kendalls_tau(self, *args, **kwargs):
        """
        Compute Kendall's tau rank correlation.

        Returns:
            float: Kendall's tau value
        """
        self._set_params(args, kwargs)
        return 2 / np.pi * np.arcsin(self.rho)
