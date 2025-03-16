import numpy as np
import sympy as sp
from scipy.stats import norm, multivariate_normal

from copul.copula_sampler import CopulaSampler
from copul.families.elliptical.multivar_gaussian import MultivariateGaussian
from copul.families.elliptical.elliptical_copula import EllipticalCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
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
            kwargs['rho'] = args[0]
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
        from statsmodels.distributions.copula.elliptical import GaussianCopula as StatsGaussianCopula
        
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
        from statsmodels.distributions.copula.elliptical import GaussianCopula as StatsGaussianCopula
        
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
        Vectorized implementation of the cumulative distribution function for bivariate Gaussian copula.
        
        This method evaluates the CDF at multiple points simultaneously, which is more efficient
        than calling the scalar CDF function repeatedly.
        
        Parameters
        ----------
        u : array_like
            First uniform marginal, should be in [0, 1].
        v : array_like
            Second uniform marginal, should be in [0, 1].
            
        Returns
        -------
        numpy.ndarray
            The CDF values at the specified points.
            
        Notes
        -----
        This implementation uses scipy's norm functions for vectorized operations, providing
        significant performance improvements for large inputs. The formula used is:
            C(u,v) = Φ_ρ(Φ^(-1)(u), Φ^(-1)(v))
        where Φ is the standard normal CDF and Φ_ρ is the bivariate normal CDF with correlation ρ.
        """
        # Convert inputs to numpy arrays if they aren't already
        u = np.asarray(u)
        v = np.asarray(v)

        # Ensure inputs are within [0, 1]
        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Marginals must be in [0, 1]")

        # Handle scalar inputs by broadcasting to the same shape
        if u.ndim == 0 and v.ndim > 0:
            u = np.full_like(v, u.item())
        elif v.ndim == 0 and u.ndim > 0:
            v = np.full_like(u, v.item())

        # Get correlation parameter as a float
        rho_val = float(self.rho)

        # Special cases for correlation extremes
        if rho_val == -1:
            # Lower Fréchet bound: max(u + v - 1, 0)
            return np.maximum(u + v - 1, 0)
        elif rho_val == 0:
            # Independence: u * v
            return u * v
        elif rho_val == 1:
            # Upper Fréchet bound: min(u, v)
            return np.minimum(u, v)

        # Initialize result array with zeros
        result = np.zeros_like(u, dtype=float)

        # Handle boundary cases efficiently
        # Where u=0 or v=0, C(u,v)=0 (already initialized to zero)
        # Where u=1, C(u,v)=v
        # Where v=1, C(u,v)=u
        result = np.where(u == 1, v, result)
        result = np.where(v == 1, u, result)

        # Find indices where neither u nor v are at the boundaries
        interior_idx = (u > 0) & (u < 1) & (v > 0) & (v < 1)

        if np.any(interior_idx):
            u_interior = u[interior_idx]
            v_interior = v[interior_idx]

            try:
                # Convert u and v to standard normal quantiles
                x = norm.ppf(u_interior)
                y = norm.ppf(v_interior)

                # Create arrays for the points and the correlation matrix
                points = np.column_stack((x, y))
                corr_matrix = np.array([[1.0, rho_val], [rho_val, 1.0]])

                # Use a batch evaluation approach for the bivariate normal CDF
                # to avoid memory issues with large inputs
                batch_size = 10000  # Adjust based on available memory
                num_points = len(points)
                result_interior = np.zeros(num_points, dtype=float)

                for i in range(0, num_points, batch_size):
                    batch_end = min(i + batch_size, num_points)
                    batch_points = points[i:batch_end]

                    # Evaluate the multivariate normal CDF for this batch
                    mvn = multivariate_normal(mean=[0, 0], cov=corr_matrix)
                    result_interior[i:batch_end] = mvn.cdf(batch_points)

                # Assign the results back to the original array
                result[interior_idx] = result_interior

            except Exception as e:
                # Fallback to using the statsmodels implementation for any failures
                import warnings
                from statsmodels.distributions.copula.elliptical import GaussianCopula as StatsGaussianCopula

                warnings.warn(
                    f"Error in vectorized CDF calculation: {e}. Using statsmodels fallback."
                )

                # Use the statsmodels implementation
                cop = StatsGaussianCopula(rho_val)

                # Process points in batches to avoid memory issues
                batch_size = 5000  # Adjust based on available memory
                num_points = np.sum(interior_idx)
                u_flat = u_interior.flatten()
                v_flat = v_interior.flatten()
                result_interior = np.zeros(num_points, dtype=float)

                for i in range(0, num_points, batch_size):
                    batch_end = min(i + batch_size, num_points)
                    uv_pairs = np.column_stack(
                        (u_flat[i:batch_end], v_flat[i:batch_end])
                    )
                    result_interior[i:batch_end] = cop.cdf(uv_pairs)

                # Assign the results back to the original array
                result[interior_idx] = result_interior.reshape(u_interior.shape)

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
        scale = float(np.sqrt(1 - float(self.rho)**2))
        
        def conditional_func(u_, v_):
            return norm.cdf(norm.ppf(v_), loc=float(self.rho) * norm.ppf(u_), scale=scale)
            
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
        from statsmodels.distributions.copula.elliptical import GaussianCopula as StatsGaussianCopula
        
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
        return 3 / np.pi * np.arcsin(1 / 2 + float(self.rho)**2 / 2) - 0.5
    
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