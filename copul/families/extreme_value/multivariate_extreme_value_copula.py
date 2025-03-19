import logging
import warnings
from contextlib import contextmanager

import numpy as np
import sympy as sp
from sympy import exp, log

from copul.families.core.copula import Copula
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

logger = logging.getLogger(__name__)


class CallableCDFWrapper:
    """
    A wrapper for Python callable functions to be used as CDF functions.
    
    This is different from CDFWrapper and SymPyFuncWrapper which expect sympy expressions.
    This wrapper accepts Python callables and provides a compatible interface.
    """
    
    def __init__(self, callable_func):
        """
        Initialize with a callable function.
        
        Parameters
        ----------
        callable_func : callable
            A Python function that takes n arguments and returns a float.
        """
        self.func = callable_func
    
    def __call__(self, *args):
        """
        Call the wrapped function with the given arguments.
        
        Parameters
        ----------
        *args
            Arguments to pass to the function.
            
        Returns
        -------
        float
            The result of calling the function.
        """
        return self.func(*args)
    
    def subs(self, *args, **kwargs):
        """
        Placeholder for the subs method to maintain compatibility.
        
        Returns
        -------
        self
            This instance, unchanged.
        """
        # For Python functions, substitution doesn't make sense
        # Return self for method chaining
        return self
    
    def evalf(self):
        """
        Placeholder for the evalf method to maintain compatibility.
        
        Returns
        -------
        self
            This instance, unchanged.
        """
        # For Python functions, numerical evaluation doesn't make sense
        # Return self for method chaining
        return self


class MultivariateExtremeValueCopula(Copula):
    """
    Multivariate Extreme Value Copula.
    
    An extension of the Copula class designed for multivariate extreme value distributions.
    Extreme value copulas arise as the limiting distributions of component-wise maxima
    of random vectors.
    
    Attributes
    ----------
    params : list
        List of parameters for the extreme value copula.
    intervals : dict
        Parameter intervals for the extreme value copula.
    """
    
    params = []
    intervals = {}
    _free_symbols = {}
    
    def __init__(self, dimension, *args, **kwargs):
        """
        Initialize a multivariate extreme value copula.
        
        Parameters
        ----------
        dimension : int
            Dimension of the copula (number of variables).
        *args, **kwargs
            Additional parameters for the specific extreme value copula.
        """
        Copula.__init__(self, dimension, *args, **kwargs)
        
    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.
        
        Returns
        -------
        bool
            Must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented in a subclass")
    
    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric.
        
        Returns
        -------
        bool
            Must be implemented by subclasses.
        """
        raise NotImplementedError("This method must be implemented in a subclass")
    
    def _compute_extreme_value_function(self, u_values):
        """
        Compute the extreme value function based on the specific copula type.
        
        Parameters
        ----------
        u_values : list
            List of u values (marginals) for evaluation.
            
        Returns
        -------
        float
            The computed extreme value function value.
        """
        raise NotImplementedError("This method must be implemented in a subclass")
    
    @property
    def cdf(self):
        """
        Compute the cumulative distribution function (CDF) of the extreme value copula.
        
        Returns
        -------
        CallableCDFWrapper
            A wrapper around the CDF function.
        """
        try:
            # Create a function to compute the extreme value function for any u values
            def cdf_func(*args):
                if len(args) != self.dim:
                    raise ValueError(f"Expected {self.dim} arguments, got {len(args)}")
                
                # Handle boundary cases
                if any(u <= 0 for u in args):
                    return 0
                if all(u == 1 for u in args):
                    return 1
                
                # Compute the extreme value function
                ev_value = self._compute_extreme_value_function(args)
                
                # Return the computed CDF value
                return ev_value
            
            # Return a wrapper around the CDF function
            return CallableCDFWrapper(cdf_func)
            
        except Exception as e:
            # Fallback implementation if the approach fails
            warnings.warn(
                f"Error in CDF calculation: {e}. Using fallback implementation."
            )
            
            # Simple fallback that returns min of all arguments (Fréchet-Hoeffding upper bound)
            return CallableCDFWrapper(lambda *args: min(args))
    
    def cdf_vectorized(self, *args):
        """
        Vectorized implementation of the CDF function for improved performance with arrays.
        
        Parameters
        ----------
        *args : array_like
            Arrays of dimension values to evaluate the CDF at.
            
        Returns
        -------
        numpy.ndarray
            Array of CDF values.
        """
        # Convert all inputs to numpy arrays
        arrays = [np.asarray(arg) for arg in args]
        
        # Ensure all arrays have the same shape
        shapes = [arr.shape for arr in arrays]
        if len(set(shapes)) > 1:
            raise ValueError("All input arrays must have the same shape")
        
        # Create result array
        result = np.zeros(shapes[0])
        
        # Handle boundary cases
        mask_zeros = np.any(np.array([arr <= 0 for arr in arrays]), axis=0)
        mask_ones = np.all(np.array([arr == 1 for arr in arrays]), axis=0)
        
        # Set boundary values
        result[mask_zeros] = 0
        result[mask_ones] = 1
        
        # Process interior points
        interior_mask = ~(mask_zeros | mask_ones)
        if np.any(interior_mask):
            # Get flattened indices of interior points
            indices = np.where(interior_mask.flatten())[0]
            
            # Process each point individually
            flat_shape = result.flatten().shape
            flat_arrays = [arr.flatten() for arr in arrays]
            
            for idx in indices:
                point_values = [arr[idx] for arr in flat_arrays]
                result.flat[idx] = self.cdf(*point_values)
        
        return result
    
    def sample_parameters(self, n=1):
        """
        Sample random parameter values within the defined intervals.
        
        Parameters
        ----------
        n : int, optional
            Number of parameter sets to sample (default is 1).
            
        Returns
        -------
        dict
            Dictionary of parameter name-value pairs.
        """
        # Make sure self.intervals is properly initialized
        if not hasattr(self, "intervals") or not self.intervals:
            # Fall back to class-level intervals if instance-level is empty
            intervals_to_use = self.__class__.intervals
        else:
            intervals_to_use = self.intervals

        return {
            k: list(np.random.uniform(max(-10, v.inf), min(10, v.sup), n))
            for k, v in intervals_to_use.items()
        }
    
    @contextmanager
    def suppress_warnings(self):
        """
        Context manager to temporarily suppress warnings.
        
        Yields
        ------
        None
        """
        warnings.filterwarnings("ignore")
        yield
        warnings.filterwarnings("default")