from typing import TypeAlias
import numpy as np
import sympy as sp
from sympy import exp, log

from copul.families.extreme_value.multivariate_extreme_value_copula import MultivariateExtremeValueCopula
from copul.families.other import BivIndependenceCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class MultivariateGumbelHougaard(MultivariateExtremeValueCopula):
    """
    Multivariate Gumbel-Hougaard Extreme Value Copula.
    
    A specialized extreme value copula with the form:
    C(u₁, u₂, ..., uₙ) = exp(-((-ln u₁)^θ + (-ln u₂)^θ + ... + (-ln uₙ)^θ)^(1/θ))
    
    Special cases:
    - θ = 1: Independence copula
    - θ → ∞: Comonotonicity copula (perfect positive dependence)
    
    Parameters
    ----------
    dimension : int
        Dimension of the copula (number of variables).
    theta : float, optional
        Dependence parameter (default is None). 
        Must be greater than or equal to 1.
    """
    
    # Define parameters
    theta = sp.symbols("theta", positive=True)
    params = [theta]
    intervals = {str(theta): sp.Interval(1, float('inf'), left_open=False, right_open=True)}
    
    def __new__(cls, dimension=2, *args, **kwargs):
        """
        Custom instance creation to handle the special case of θ=1.
        
        When θ=1, the Gumbel-Hougaard copula reduces to the independence copula.
        
        Parameters
        ----------
        dimension : int, optional
            Dimension of the copula (default is 2).
        *args, **kwargs
            Additional arguments, potentially including 'theta'.
            
        Returns
        -------
        Copula
            Either a MultivariateGumbelHougaard or IndependenceCopula instance.
        """
        # Check if theta=1 is passed either as a positional arg or keyword arg
        theta_is_one = False
        if args and len(args) > 0 and args[0] == 1:
            theta_is_one = True
        if "theta" in kwargs and kwargs["theta"] == 1:
            theta_is_one = True
            
        if theta_is_one:
            # Return an IndependenceCopula instance
            new_kwargs = kwargs.copy()
            if "theta" in new_kwargs:
                del new_kwargs["theta"]
            if dimension == 2:
                return BivIndependenceCopula(dimension, **new_kwargs)
            return IndependenceCopula(dimension, **new_kwargs)
            
        # If theta is not 1, proceed with normal initialization
        return super().__new__(cls)
    
    def __init__(self, dimension=2, theta=None, *args, **kwargs):
        """
        Initialize a Multivariate Gumbel-Hougaard copula.
        
        Parameters
        ----------
        dimension : int, optional
            Dimension of the copula (default is 2).
        theta : float, optional
            Dependence parameter. Must be greater than or equal to 1.
        *args, **kwargs
            Additional parameters.
        """
        super().__init__(dimension=dimension, *args, **kwargs)
        
        if theta is not None:
            self.theta = theta
            self._free_symbols = {"theta": self.theta}
    
    def __call__(self, *args, **kwargs):
        """
        Return a new instance with updated parameters.
        
        Handle the special case where θ=1, which should return 
        an independence copula.
        
        Parameters
        ----------
        *args
            Positional arguments, potentially including theta.
        **kwargs
            Keyword arguments, potentially including 'theta'.
            
        Returns
        -------
        Copula
            Either a MultivariateGumbelHougaard or IndependenceCopula instance.
        """
        if args and len(args) > 0:
            kwargs["theta"] = args[0]
            
        if "theta" in kwargs and kwargs["theta"] == 1:
            # If theta is 1, return an IndependenceCopula
            del kwargs["theta"]
            if self.dim == 2:
                return BivIndependenceCopula(self.dim, **kwargs)
            return IndependenceCopula(self.dim, **kwargs)
            
        # Otherwise, proceed with normal call
        return super().__call__(*args, **kwargs)
    
    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.
        
        The Gumbel-Hougaard copula is absolutely continuous.
        
        Returns
        -------
        bool
            True
        """
        return True
    
    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric.
        
        The Gumbel-Hougaard copula is symmetric in all its arguments.
        
        Returns
        -------
        bool
            True
        """
        return True
    
    def _compute_extreme_value_function(self, u_values):
        """
        Compute the Gumbel-Hougaard extreme value function.
        
        Parameters
        ----------
        u_values : list
            List of u values (marginals) for evaluation.
            
        Returns
        -------
        float
            The computed extreme value function value.
        """
        # Check for boundary conditions
        if any(u <= 0 for u in u_values):
            return 0
        if all(u == 1 for u in u_values):
            return 1
        
        # Extract theta value (handle both numeric and symbolic cases)
        theta_val = self.theta
        if not isinstance(theta_val, (int, float)) and hasattr(theta_val, 'evalf'):
            theta_val = float(theta_val.evalf())
        
        # Compute the sum of negative log u values raised to theta
        neg_log_sum = sum((-np.log(u)) ** theta_val for u in u_values)
        
        # Return the extremal function value
        return np.exp(-(neg_log_sum) ** (1 / theta_val))
    
    @property
    def cdf(self):
        """
        Compute the CDF of the multivariate Gumbel-Hougaard copula.
        
        The explicit formula is:
        C(u₁, u₂, ..., uₙ) = exp(-((-ln u₁)^θ + (-ln u₂)^θ + ... + (-ln uₙ)^θ)^(1/θ))
        
        Returns
        -------
        CDFWrapper
            A wrapper around the CDF function.
        """
        try:
            # Get the u symbols
            u_symbols = self.u_symbols
            
            # Create the sum of negative logs raised to theta
            neg_log_sum = sum((-sp.log(u)) ** self.theta for u in u_symbols)
            
            # Create the CDF expression
            cdf_expr = sp.exp(-(neg_log_sum) ** (1 / self.theta))
            
            # Create piecewise expression to handle boundary cases
            boundary_conditions = sp.Or(*[sp.Eq(u, 0) for u in u_symbols])
            cdf_piecewise = sp.Piecewise(
                (0, boundary_conditions),
                (cdf_expr, True)
            )
            
            return CDFWrapper(cdf_piecewise)
            
        except Exception as e:
            # Fall back to the general implementation in case of errors
            return super().cdf
    
    def tau(self, *args, **kwargs):
        """
        Compute Kendall's tau for the multivariate Gumbel-Hougaard copula.
        
        For the bivariate case, Kendall's tau is (θ-1)/θ.
        For higher dimensions, the pairwise Kendall's tau is the same for any pair.
        
        Parameters
        ----------
        *args, **kwargs
            Parameters for the copula.
            
        Returns
        -------
        float or sympy.Expr
            Kendall's tau value or expression.
        """
        self._set_params(args, kwargs)
        return (self.theta - 1) / self.theta
    
    def cdf_vectorized(self, *args):
        """
        Vectorized implementation of the CDF for the Gumbel-Hougaard copula.
        
        Parameters
        ----------
        *args : array_like
            Arrays of dimension values to evaluate the CDF at.
            
        Returns
        -------
        numpy.ndarray
            Array of CDF values.
        """
        # Convert inputs to numpy arrays
        arrays = [np.asarray(arg) for arg in args]
        
        # Check dimensions
        if len(arrays) != self.dim:
            raise ValueError(f"Expected {self.dim} arrays, got {len(arrays)}")
        
        # Ensure all arrays have compatible shapes
        shape = np.broadcast(*arrays).shape
        result = np.zeros(shape)
        
        # Get theta value
        theta_val = self.theta
        if not isinstance(theta_val, (int, float)) and hasattr(theta_val, 'evalf'):
            theta_val = float(theta_val.evalf())
        
        # Handle boundary case: if any u is 0, the result is 0
        mask_zeros = np.zeros(shape, dtype=bool)
        for arr in arrays:
            mask_zeros = np.logical_or(mask_zeros, np.isclose(arr, 0))
        
        # Handle interior points
        interior_mask = ~mask_zeros
        if np.any(interior_mask):
            # Calculate the sum of negative logs raised to theta
            neg_log_sum = np.zeros(shape)
            for arr in arrays:
                neg_log_sum[interior_mask] += (-np.log(arr[interior_mask])) ** theta_val
            
            # Calculate CDF values
            result[interior_mask] = np.exp(-(neg_log_sum[interior_mask]) ** (1 / theta_val))
        
        return result


# Define alias
MultivariateGumbelHougaardEV: TypeAlias = MultivariateGumbelHougaard