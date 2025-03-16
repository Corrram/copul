import sympy as sp
from abc import abstractmethod

from copul.families.elliptical.multivar_elliptical_copula import MultivariateEllipticalCopula
from copul.families.bivcopula import BivCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.families.other.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

class EllipticalCopula(MultivariateEllipticalCopula, BivCopula):
    """
    Abstract base class for bivariate elliptical copulas.
    
    This class extends MultivariateEllipticalCopula for the bivariate (2-dimensional) case.
    Elliptical copulas are derived from elliptical distributions and are
    characterized by a correlation parameter rho in [-1, 1].
    
    Special cases:
    - rho = -1: Lower Fréchet bound (countermonotonicity)
    - rho = 1: Upper Fréchet bound (comonotonicity)
    
    Attributes
    ----------
    t : sympy.Symbol
        Symbol representing the generator variable.
    generator : sympy.Expr or None
        Generator function for the elliptical distribution (to be defined in subclasses).
    rho : sympy.Symbol
        Symbol representing the correlation parameter.
    params : list
        List of parameters defining the copula.
    intervals : dict
        Dictionary mapping parameter names to their valid intervals.
    """
    
    t = sp.symbols("t", positive=True)
    generator = None
    rho = sp.symbols("rho", real=True)
    params = [rho]
    intervals = {"rho": sp.Interval(-1, 1, left_open=False, right_open=False)}
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a bivariate elliptical copula.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments corresponding to copula parameters.
        **kwargs : dict
            Keyword arguments to override default symbolic parameters.
        """
        # Set dimension to 2 since this is a bivariate copula
        if "dimension" in kwargs and kwargs["dimension"] != 2:
            raise ValueError("EllipticalCopula is a bivariate copula with dimension=2")
        
        kwargs["dimension"] = 2
        
        # Preserve symbolic parameters if numeric values aren't provided
        if 'rho' not in kwargs and len(args) == 0:
            self.rho = self.__class__.rho  # Use the class's symbolic rho
            self.params = list(self.__class__.params)  # Copy the class's params list
            self.intervals = dict(self.__class__.intervals)  # Copy the class's intervals dict
        
        # If rho is provided, construct a 2x2 correlation matrix
        if 'rho' in kwargs:
            rho_val = kwargs['rho']
            corr_matrix = sp.Matrix([[1, rho_val], [rho_val, 1]])
            kwargs['corr_matrix'] = corr_matrix
            self.rho = rho_val  # Set the instance's rho
            
            # Update params and intervals for numeric rho
            if hasattr(self, 'params'):
                self.params = [p for p in self.params if str(p) != 'rho']
            if hasattr(self, 'intervals') and 'rho' in self.intervals:
                self.intervals = {k: v for k, v in self.intervals.items() if k != 'rho'}
        
        # Initialize from MultivariateEllipticalCopula
        # Since we've already set dimension in kwargs, don't pass it again
        if 'dimension' in kwargs:
            dimension = kwargs['dimension']
            del kwargs['dimension']
        MultivariateEllipticalCopula.__init__(self, dimension, *args, **kwargs)
        
        # Initialize from BivCopula (with explicit dimension to avoid conflict)
        biv_kwargs = {k: v for k, v in kwargs.items() if k != "dimension"}
        BivCopula.__init__(self, **biv_kwargs)
    
    def __call__(self, **kwargs):
        """
        Create a new instance with updated parameters.
        
        Special case handling for boundary rho values.
        
        Parameters
        ----------
        **kwargs
            Updated parameter values.
            
        Returns
        -------
        EllipticalCopula
            A new instance with the updated parameters.
        """
        if "rho" in kwargs:
            if kwargs["rho"] == -1:
                del kwargs["rho"]
                return LowerFrechet()(**kwargs)
            elif kwargs["rho"] == 1:
                del kwargs["rho"]
                return UpperFrechet()(**kwargs)
        return super().__call__(**kwargs)
    
    @property
    def corr_matrix(self):
        """
        Get the 2x2 correlation matrix based on the rho parameter.
        
        Returns
        -------
        sympy.Matrix
            2x2 correlation matrix with ones on the diagonal
            and rho on the off-diagonal.
        """
        return sp.Matrix([[1, self.rho], [self.rho, 1]])
        
    @corr_matrix.setter
    def corr_matrix(self, matrix):
        """
        Set the correlation matrix and update rho.
        
        Parameters
        ----------
        matrix : sympy.Matrix
            2x2 correlation matrix.
        """
        # Only extract rho from the matrix if it's a 2x2 matrix
        if isinstance(matrix, sp.Matrix) and matrix.shape == (2, 2):
            self.rho = matrix[0, 1]
    
    def characteristic_function(self, t1, t2):
        """
        Compute the characteristic function of the elliptical copula.
        
        Parameters
        ----------
        t1 : float or sympy.Symbol
            First argument
        t2 : float or sympy.Symbol
            Second argument
            
        Returns
        -------
        sympy.Expr
            Value of the characteristic function
            
        Raises
        ------
        NotImplementedError
            If generator is not defined in the subclass
        """
        if self.generator is None:
            raise NotImplementedError("Generator function must be defined in subclass")
            
        arg = (
            t1**2 * self.corr_matrix[0, 0]
            + t2**2 * self.corr_matrix[1, 1]
            + 2 * t1 * t2 * self.corr_matrix[0, 1]
        )
        # Make a proper substitution with a dictionary
        t = self.t  # Get the symbol
        return self.generator.subs({t: arg})