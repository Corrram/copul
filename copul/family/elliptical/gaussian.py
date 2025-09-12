import numpy as np
import sympy as sp
from scipy.stats import norm

from copul.copula_sampler import CopulaSampler
from copul.family.elliptical.multivar_gaussian import MultivariateGaussian
from copul.family.elliptical.elliptical_copula import EllipticalCopula
from copul.family.other.biv_independence_copula import BivIndependenceCopula
from copul.family.other.lower_frechet import LowerFrechet
from copul.family.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Gaussian(MultivariateGaussian, EllipticalCopula):
    """
    Bivariate Gaussian copula implementation.

    This class extends MultivariateGaussian for the bivariate (2-dimensional) case.
    The Gaussian copula is an elliptical copula based on the multivariate normal distribution.
    It is characterized by a correlation parameter rho in [-1, 1].

    Special cases:
    - rho = -1: Lower Fréchet bound (countermonotonicity)
    - rho = 0: Independence copula
    - rho = 1: Upper Fréchet bound (comonotonicity)
    """

    # Define generator as a symbolic expression with 't' as the variable
    t = sp.symbols("t", positive=True)
    generator = sp.exp(-t / 2)

    def __new__(cls, *args, **kwargs):
        """
        Factory method to handle special cases during initialization.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to the constructor.

        Returns
        -------
        Copula
            An instance of the appropriate copula class.
        """
        # Handle special cases during initialization with positional args
        if len(args) == 1:
            if args[0] == -1:
                return LowerFrechet()
            elif args[0] == 0:
                return BivIndependenceCopula()
            elif args[0] == 1:
                return UpperFrechet()

        # Default case - proceed with normal initialization
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        """
        Initialize a bivariate Gaussian copula.

        Parameters
        ----------
        *args : tuple
            Positional arguments corresponding to copula parameters.
        **kwargs : dict
            Keyword arguments to override default symbolic parameters.
        """
        # Handle special cases from __new__
        if len(args) == 1 and isinstance(args[0], (int, float)):
            kwargs["rho"] = args[0]
            args = ()

        # Call parent initializer
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Create a new instance with updated parameters.

        Special case handling for boundary rho values.

        Parameters
        ----------
        *args, **kwargs
            Updated parameter values.

        Returns
        -------
        Copula
            A new instance with the updated parameters.
        """
        if args is not None and len(args) == 1:
            kwargs["rho"] = args[0]

        if "rho" in kwargs:
            if kwargs["rho"] == -1:
                del kwargs["rho"]
                return LowerFrechet()(**kwargs)
            elif kwargs["rho"] == 0:
                del kwargs["rho"]
                return BivIndependenceCopula()(**kwargs)
            elif kwargs["rho"] == 1:
                del kwargs["rho"]
                return UpperFrechet()(**kwargs)

        return super().__call__(**kwargs)

    def rvs(self, n=1, approximate=False, random_state=None, **kwargs):
        """
        Generate random samples from the Gaussian copula.

        For the bivariate case, this can use the statsmodels implementation
        for efficiency.

        Parameters
        ----------
        n : int
            Number of samples to generate
        **kwargs
            Additional keyword arguments

        Returns
        -------
        numpy.ndarray
            Array of shape (n, 2) containing the samples
        """
        if approximate:
            sampler = CopulaSampler(self, random_state=random_state)
            return sampler.rvs(n, approximate)
        from statsmodels.distributions.copula.elliptical import (
            GaussianCopula as StatsGaussianCopula,
        )

        # For bivariate case, we can use the statsmodels implementation
        if self.dim == 2:
            return StatsGaussianCopula(float(self.rho)).rvs(n)
        else:
            # Otherwise use the multivariate implementation
            return super().rvs(n, **kwargs)

    @property
    def cdf(self):
        """
        Compute the cumulative distribution function of the Gaussian copula.

        For the bivariate case, this can use the statsmodels implementation
        for efficiency.

        Returns
        -------
        callable
            Function that computes the CDF at given points
        """
        from statsmodels.distributions.copula.elliptical import (
            GaussianCopula as StatsGaussianCopula,
        )

        # For bivariate case, we can use the statsmodels implementation
        if self.dim == 2:
            cop = StatsGaussianCopula(float(self.rho))

            def gauss_cdf(u, v):
                if u == 0 or v == 0:
                    return sp.S.Zero
                else:
                    return float(cop.cdf([u, v]))

            return lambda u, v: SymPyFuncWrapper(gauss_cdf(u, v))
        else:
            # Otherwise use the multivariate implementation
            return super().cdf

    def cdf_vectorized(self, u, v):
        """
        Vectorized implementation of the cumulative distribution function for the bivariate Gaussian copula.

        This highly optimized method evaluates the CDF at multiple points simultaneously.

        Parameters
        ----------
        u : array_like
            First uniform marginal, must be in [0, 1].
        v : array_like
            Second uniform marginal, must be in [0, 1].

        Returns
        -------
        numpy.ndarray
            The CDF values at the specified points.

        Notes
        -----
        This implementation leverages the highly optimized, compiled backend of statsmodels
        for the core computation, providing a significant performance increase over manual
        iteration or less specialized scipy functions. The formula is:
            C(u,v) = Φ_ρ(Φ^(-1)(u), Φ^(-1)(v))
        where Φ is the standard normal CDF and Φ_ρ is the bivariate normal CDF with correlation ρ.
        """
        # Ensure inputs are numpy arrays for vectorized operations
        u = np.asarray(u)
        v = np.asarray(v)

        # Validate that inputs are within the valid [0, 1] range
        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Marginals must be in [0, 1]")

        # Use numpy's broadcasting to handle scalar and array inputs seamlessly
        if u.ndim == 0 and v.ndim > 0:
            u = np.broadcast_to(u, v.shape)
        elif v.ndim == 0 and u.ndim > 0:
            v = np.broadcast_to(v, u.shape)

        # Get the correlation parameter as a standard float
        rho_val = float(self.rho)

        # Handle special cases for correlation extremes with fast, vectorized numpy functions
        if rho_val == -1:
            return np.maximum(u + v - 1, 0)  # Lower Fréchet bound
        if rho_val == 0:
            return u * v  # Independence copula
        if rho_val == 1:
            return np.minimum(u, v)  # Upper Fréchet bound

        # Initialize the result array. The default of 0 correctly handles C(0,v) and C(u,0).
        result = np.zeros_like(u, dtype=float)

        # Identify points that are not on the boundaries of the unit square [0,1]^2
        # These are the only points that require the complex bivariate normal CDF calculation.
        interior_mask = (u > 0) & (u < 1) & (v > 0) & (v < 1)

        # Process interior points if any exist
        if np.any(interior_mask):
            from statsmodels.distributions.copula.elliptical import (
                GaussianCopula as StatsGaussianCopula,
            )

            # Instantiate the statsmodels Gaussian Copula
            # Note: statsmodels expects the correlation matrix parameter named 'corr'
            cop = StatsGaussianCopula(corr=rho_val)

            # Create pairs of (u, v) for the interior points
            uv_pairs = np.column_stack((u[interior_mask], v[interior_mask]))

            # Calculate the CDF for all interior points in a single, fast, vectorized call
            result[interior_mask] = cop.cdf(uv_pairs)

        # Handle the u=1 and v=1 boundary cases.
        # np.where is used for vectorized conditional assignment.
        result[u == 1] = v[u == 1]
        result[v == 1] = u[v == 1]

        return result

    def _conditional_distribution(self, u=None, v=None):
        """
        Compute the conditional distribution function of the bivariate Gaussian copula.

        Parameters
        ----------
        u : float, optional
            First marginal value
        v : float, optional
            Second marginal value

        Returns
        -------
        callable or float
            Conditional distribution function or value
        """
        scale = float(np.sqrt(1 - float(self.rho) ** 2))

        def conditional_func(u_, v_):
            return norm.cdf(
                norm.ppf(v_), loc=float(self.rho) * norm.ppf(u_), scale=scale
            )

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

        Parameters
        ----------
        u : float, optional
            Conditioning value
        v : float, optional
            Value at which to evaluate

        Returns
        -------
        SymPyFuncWrapper
            Wrapped conditional distribution function or value
        """
        if v in [0, 1]:
            return SymPyFuncWrapper(sp.Number(v))
        return SymPyFuncWrapper(sp.Number(self._conditional_distribution(u, v)))

    def cond_distr_2(self, u=None, v=None):
        """
        Compute the second conditional distribution C(u|v).

        Parameters
        ----------
        u : float, optional
            Value at which to evaluate
        v : float, optional
            Conditioning value

        Returns
        -------
        SymPyFuncWrapper
            Wrapped conditional distribution function or value
        """
        if u in [0, 1]:
            return SymPyFuncWrapper(sp.Number(u))
        return SymPyFuncWrapper(sp.Number(self._conditional_distribution(v, u)))

    @property
    def pdf(self):
        """
        Compute the probability density function of the Gaussian copula.

        For the bivariate case, this can use the statsmodels implementation
        for efficiency.

        Returns
        -------
        callable
            Function that computes the PDF at given points
        """
        from statsmodels.distributions.copula.elliptical import (
            GaussianCopula as StatsGaussianCopula,
        )

        # For bivariate case, we can use the statsmodels implementation
        if self.dim == 2:
            return lambda u, v: SymPyFuncWrapper(
                sp.Number(StatsGaussianCopula(float(self.rho)).pdf([u, v]))
            )
        else:
            # Otherwise use the multivariate implementation
            return super().pdf

    def chatterjees_xi(self, *args, **kwargs):
        """
        Compute Chatterjee's xi measure of dependence.

        Returns
        -------
        float
            Chatterjee's xi value
        """
        self._set_params(args, kwargs)
        return 3 / np.pi * np.arcsin(1 / 2 + float(self.rho) ** 2 / 2) - 0.5

    def spearmans_rho(self, *args, **kwargs):
        """
        Compute Spearman's rho rank correlation.

        Returns
        -------
        float
            Spearman's rho value
        """
        self._set_params(args, kwargs)
        return 6 / np.pi * np.arcsin(float(self.rho) / 2)

    def kendalls_tau(self, *args, **kwargs):
        """
        Compute Kendall's tau rank correlation.

        Returns
        -------
        float
            Kendall's tau value
        """
        self._set_params(args, kwargs)
        return 2 / np.pi * np.arcsin(float(self.rho))

    def spearmans_footrule(self, *args, **kwargs) -> float:
        """
        Spearman's footrule F = E[|U - V|] for the Gaussian copula.

        Closed form:
            F(rho) = 1/2 - (3/pi) * arcsin( (1 + rho) / 2 )

        Returns
        -------
        float
            Footrule distance in [0, 0.5].
        """
        self._set_params(args, kwargs)
        rho = float(self.rho)

        # numeric safety: clamp to [-1, 1]
        rho = max(-1.0, min(1.0, rho))

        footrule = (3.0 / np.pi) * np.arcsin((1.0 + rho) / 2.0) - 0.5
        return footrule
