import numpy as np
import sympy as sp

from copul.families.extreme_value.multivariate_extreme_value_copula import MultivariateExtremeValueCopula, CallableCDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class MultivariateExtremeIndependenceCopula(MultivariateExtremeValueCopula):
    """
    Multivariate Independence Copula as an Extreme Value Copula.
    
    This class represents the independence copula C(u₁, u₂, ..., uₙ) = u₁ × u₂ × ... × uₙ
    as a special case of an extreme value copula.
    
    The independence copula is the only copula that is both an extreme value copula
    and an Archimedean copula. In the extreme value context, it corresponds to
    having complete tail independence between variables.
    
    Parameters
    ----------
    dimension : int, optional
        Dimension of the copula (number of variables). Default is 2.
    """
    
    params = []  # No parameters for independence copula
    intervals = {}  # No parameter intervals
    
    def __init__(self, dimension=2, **kwargs):
        """
        Initialize a multivariate independence copula.
        
        Parameters
        ----------
        dimension : int, optional
            Dimension of the copula (default is 2).
        **kwargs
            Additional keyword arguments (ignored).
        """
        super().__init__(dimension=dimension, **kwargs)
        self._free_symbols = {}
    
    def __call__(self, **kwargs):
        """
        Return a new instance with the same dimension.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments (ignored).
            
        Returns
        -------
        MultivariateExtremeIndependenceCopula
            A new instance with the same dimension.
        """
        return MultivariateExtremeIndependenceCopula(dimension=self.dim)
    
    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.
        
        The independence copula is absolutely continuous.
        
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
        
        The independence copula is symmetric in all its arguments.
        
        Returns
        -------
        bool
            True
        """
        return True
    
    def _compute_extreme_value_function(self, u_values):
        """
        Compute the extreme value function for the independence copula.
        
        For the independence copula, the extreme value function is simply the product
        of all arguments.
        
        Parameters
        ----------
        u_values : list
            List of u values (marginals) for evaluation.
            
        Returns
        -------
        float
            The product of all u values.
        """
        # The independence copula CDF is the product of all arguments
        return np.prod(u_values)
    
    @property
    def cdf(self):
        """
        Compute the cumulative distribution function (CDF) of the independence copula.
        
        For the independence copula, the CDF is the product of all marginals:
        C(u₁, u₂, ..., uₙ) = u₁ × u₂ × ... × uₙ
        
        Returns
        -------
        CallableCDFWrapper
            A wrapper around the CDF function.
        """
        return CallableCDFWrapper(lambda *args: np.prod(args))
    
    def cdf_vectorized(self, *args):
        """
        Vectorized implementation of the CDF for the independence copula.
        
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
        
        # Check dimensions
        if len(arrays) != self.dim:
            raise ValueError(f"Expected {self.dim} arrays, got {len(arrays)}")
            
        # Compute the product of all arrays
        # Start with ones array of the right shape
        result = np.ones(np.broadcast(*arrays).shape)
        
        # Multiply by each array
        for arr in arrays:
            result = result * arr
            
        return result
    
    @property
    def pdf(self):
        """
        Compute the probability density function (PDF) of the independence copula.
        
        For the independence copula, the PDF is constant 1 on the unit cube.
        
        Returns
        -------
        SymPyFuncWrapper
            A wrapper around the PDF function.
        """
        return SymPyFuncWrapper(sp.Integer(1))
    
    def pdf_vectorized(self, *args):
        """
        Vectorized implementation of the PDF for the independence copula.
        
        Parameters
        ----------
        *args : array_like
            Arrays of dimension values to evaluate the PDF at.
            
        Returns
        -------
        numpy.ndarray
            Array of PDF values (all 1s).
        """
        # Convert all inputs to numpy arrays
        arrays = [np.asarray(arg) for arg in args]
        
        # Check dimensions
        if len(arrays) != self.dim:
            raise ValueError(f"Expected {self.dim} arrays, got {len(arrays)}")
            
        # Return array of ones with appropriate shape
        return np.ones(np.broadcast(*arrays).shape)
    
    def tau(self):
        """
        Compute Kendall's tau for the independence copula.
        
        For the independence copula, Kendall's tau is 0 as there is no dependence.
        
        Returns
        -------
        float
            0
        """
        return 0
    
    def rho(self):
        """
        Compute Spearman's rho for the independence copula.
        
        For the independence copula, Spearman's rho is 0 as there is no dependence.
        
        Returns
        -------
        float
            0
        """
        return 0
    
    def lambda_L(self):
        """
        Compute the lower tail dependence coefficient.
        
        For the independence copula, there is no tail dependence.
        
        Returns
        -------
        float
            0
        """
        return 0
    
    def lambda_U(self):
        """
        Compute the upper tail dependence coefficient.
        
        For the independence copula, there is no tail dependence.
        
        Returns
        -------
        float
            0
        """
        return 0
    
    def rvs(self, n=1, random_state=None):
        """
        Generate random variates from the independence copula.
        
        For the independence copula, this simply generates independent uniform random variables.
        
        Parameters
        ----------
        n : int, optional
            Number of samples to generate (default is 1).
        random_state : int or None, optional
            Seed for the random number generator.
            
        Returns
        -------
        numpy.ndarray
            Array of shape (n, dim) containing independent uniform samples.
        """
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            
        # Generate independent uniform random variables
        return np.random.uniform(0, 1, size=(n, self.dim))