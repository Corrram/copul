import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.independence_copula import IndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Clayton(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta_interval = sympy.Interval(-1, np.inf, left_open=False, right_open=True)

    @classmethod
    def create(cls, *args, **kwargs):
        """Factory method to create the appropriate copula instance."""
        # ToDo - this create, __new__, and __call__ methods should be implemented in the base class
        # this can be achieved by in the subclass having a class attribute mentioning the special cases
        # and then the base class can handle the special cases
        # Handle positional arguments
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
            
        # Check for special case: theta = -1 (Lower Fréchet)
        if "theta" in kwargs and kwargs["theta"] == -1:
            del kwargs["theta"]
            return LowerFrechet()
            
        # Check for special case: theta = 0 (Independence)
        if "theta" in kwargs and kwargs["theta"] == 0:
            del kwargs["theta"]
            return IndependenceCopula()
            
        # Otherwise create a Clayton copula
        return cls(**kwargs)

    def __new__(cls, *args, **kwargs):
        """Override __new__ to handle special cases."""
        # Handle positional arguments
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
            
        # Check for special cases
        if "theta" in kwargs:
            if kwargs["theta"] == -1:
                del kwargs["theta"]
                return LowerFrechet()
            if kwargs["theta"] == 0:
                del kwargs["theta"]
                return IndependenceCopula()
                
        # Standard creation for normal cases
        return super().__new__(cls)
    
    def __call__(self, *args, **kwargs):
        """Handle special cases when calling the instance."""
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
            
        # Check for special cases
        if "theta" in kwargs:
            if kwargs["theta"] == -1:
                del kwargs["theta"]
                return LowerFrechet()
            if kwargs["theta"] == 0:
                del kwargs["theta"]
                return IndependenceCopula()
                
        return super().__call__(**kwargs)

    @property
    def _generator(self):
        # Fix for theta = 0: use logarithmic generator
        if self.theta == 0:
            return -sympy.log(self.t)
        return ((1 / self.t) ** self.theta - 1) / self.theta

    @property
    def inv_generator(self):
        if self.theta == 0:
            return SymPyFuncWrapper(sympy.exp(-self.y))
        ind = sympy.Piecewise(
            (1, (self.y < -1 / self.theta) | (self.theta > 0)), (0, True)
        )
        cdf = ind * (self.theta * self.y + 1) ** (-1 / self.theta)
        return SymPyFuncWrapper(cdf)

    @property
    def cdf(self):
        u = self.u
        theta = self.theta
        v = self.v
        
        # Special case for theta = 0 (Independence Copula)
        if theta == 0:
            return CDFWrapper(u * v)
            
        cdf = sympy.Max((u ** (-theta) + v ** (-theta) - 1), 0) ** (-1 / theta)
        return CDFWrapper(cdf)

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        
        # Handle special case for theta = 0
        if theta == 0:
            return v  # For independence copula, conditional distribution is just v
            
        cond_distr = sympy.Heaviside(-1 + self.u ** (-theta) + self.v ** (-theta)) / (
            self.u
            * self.u**theta
            * (-1 + self.u ** (-theta) + self.v ** (-theta))
            * (-1 + self.u ** (-theta) + self.v ** (-theta)) ** (1 / theta)
        )
        wrapped_cd1 = CD1Wrapper(cond_distr)
        evaluated_cd1 = wrapped_cd1(u, v)
        return evaluated_cd1

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        
        # Handle special case for theta = 0
        if theta == 0:
            return u  # For independence copula, conditional distribution is just u
            
        cond_distr = sympy.Heaviside(
            (-1 + self.v ** (-theta) + self.u ** (-theta)) ** (-1 / theta)
        ) / (
            self.v
            * self.v**theta
            * (-1 + self.v ** (-theta) + self.u ** (-theta))
            * (-1 + self.v ** (-theta) + self.u ** (-theta)) ** (1 / theta)
        )
        return CD2Wrapper(cond_distr)(u, v)

    def _squared_cond_distr_1(self, u, v):
        theta = self.theta
        
        # Handle special case for theta = 0
        if theta == 0:
            return 0  # For independence copula, second derivative is 0
            
        return sympy.Heaviside((-1 + v ** (-theta) + u ** (-theta)) ** (-1 / theta)) / (
            u**2
            * u ** (2 * theta)
            * (-1 + v ** (-theta) + u ** (-theta)) ** 2
            * (-1 + v ** (-theta) + u ** (-theta)) ** (2 / theta)
        )

    @property
    def pdf(self):
        theta = self.theta
        
        # Handle special case for theta = 0
        if theta == 0:
            return SymPyFuncWrapper(1)  # Uniform density for independence copula
            
        result = (
            (self.u ** (-theta) + self.v ** (-theta) - 1) ** (-2 - 1 / theta)
            * self.u ** (-theta - 1)
            * self.v ** (-theta - 1)
            * (theta + 1)
        )
        return SymPyFuncWrapper(result)

    @property
    def is_absolutely_continuous(self) -> bool:
        return self.theta >= 0

    def lambda_L(self):
        # Avoid division by zero for theta = 0
        if self.theta == 0:
            return 0  # Independence has no tail dependence
        return 2 ** (-1 / self.theta)

    def lambda_U(self):
        return 0


Nelsen1 = Clayton

# B4 = Clayton

PiOverSigmaMinusPi = Clayton(1)