import numpy as np
import sympy

from typing import TypeAlias
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
    # Define special cases as a dictionary mapping parameter values to classes
    special_cases = {-1: LowerFrechet, 0: IndependenceCopula}

    @property
    def _generator_at_0(self):
        return sympy.Piecewise((sympy.oo, self.theta >= 0), (-1/self.theta, True))

    @property
    def generator(self):
        # Regular case expression for theta != 0
        regular_expr = ((1 / self.t) ** self.theta - 1) / self.theta
        # Logarithmic generator for theta = 0
        log_expr = -sympy.log(self.t)
        
        gen = sympy.Piecewise(
            (log_expr, self.theta == 0),              # Case: theta = 0
            (regular_expr, self.t > 0),               # Regular case
            (sympy.oo, self.theta >= 0),              # Default case for invalid values
            (-1/self.theta, True)                     # Default case for theta < 0
        )
        return SymPyFuncWrapper(gen)

    @property
    def _raw_inv_generator(self):
        # Optional optimization - provide direct formula instead of solving
        if self.theta == 0:
            return sympy.exp(-self.y)
        return (self.theta * self.y + 1) ** (-1 / self.theta)
    
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


Nelsen1: TypeAlias = Clayton

# B4 = Clayton
