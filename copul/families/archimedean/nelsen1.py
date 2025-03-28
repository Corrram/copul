import numpy as np
import sympy

from copul.families.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper
from typing import TypeAlias


class BivClayton(BivArchimedeanCopula):
    """
    Bivariate Clayton Copula.

    The Clayton copula is defined by its generator:
    φ(t) = (t^(-θ) - 1) / θ for θ > 0
    φ(t) = -log(t) for θ = 0
    φ(t) = (t^(-θ) - 1) / θ for θ ∈ [-1, 0) with restricted range

    It allows for asymptotic lower tail dependence but no upper tail dependence.

    Parameters
    ----------
    theta : float
        The parameter controlling the strength of dependence.
        θ > 0 indicates positive dependence, θ = 0 gives independence,
        and θ ∈ [-1, 0) gives negative dependence.
    """

    theta_interval = sympy.Interval(-1, np.inf, left_open=False, right_open=True)
    # Define special cases as a dictionary mapping parameter values to classes
    special_cases = {-1: LowerFrechet, 0: BivIndependenceCopula}

    @property
    def _generator_at_0(self):
        """
        Value of the generator at t=0, which depends on θ.

        Returns
        -------
        sympy expression
            ∞ for θ ≥ 0, -1/θ for θ < 0
        """
        return sympy.Piecewise(
            (sympy.oo, self.theta >= 0),  # For θ ≥ 0, limit is infinity
            (-1 / self.theta, True),  # For θ < 0, limit is -1/θ
        )

    @property
    def _raw_generator(self):
        """
        Raw generator function for Clayton copula.

        Returns
        -------
        sympy expression
            The Clayton generator
        """
        # Regular case expression for theta != 0
        regular_expr = ((1 / self.t) ** self.theta - 1) / self.theta
        # Logarithmic generator for theta = 0
        log_expr = -sympy.log(self.t)

        # Return appropriate expression based on theta
        return sympy.Piecewise(
            (log_expr, self.theta == 0),  # Independence case (θ = 0)
            (regular_expr, True),  # Regular case (θ ≠ 0)
        )

    @property
    def _raw_inv_generator(self):
        """
        Raw inverse generator function for Clayton copula.

        Returns
        -------
        sympy expression
            The inverse generator
        """
        # Handle independence case (θ = 0)
        if self.theta == 0:
            return sympy.exp(-self.y)
        # Regular case (θ ≠ 0)
        return (self.theta * self.y + 1) ** (-1 / self.theta)

    @property
    def _cdf_expr(self):
        """
        Cumulative distribution function of the bivariate Clayton copula.

        Returns
        -------
        CDFWrapper
            The CDF function C(u,v)
        """
        u = self.u
        v = self.v
        theta = self.theta

        # Special case for theta = 0 (Independence Copula)
        if theta == 0:
            return CDFWrapper(u * v)

        # Regular formula for Clayton copula
        cdf = sympy.Max((u ** (-theta) + v ** (-theta) - 1), 0) ** (-1 / theta)
        return cdf

    def cond_distr_1(self, u=None, v=None):
        """
        First conditional distribution function: ∂C(u,v)/∂u

        Parameters
        ----------
        u, v : float or None, optional
            Values to evaluate at. If None, returns the symbolic expression.

        Returns
        -------
        CD1Wrapper
            The conditional distribution
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return v  # For independence copula, conditional distribution is just v

        # Formula for Clayton copula
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
        """
        Second conditional distribution function: ∂C(u,v)/∂v

        Parameters
        ----------
        u, v : float or None, optional
            Values to evaluate at. If None, returns the symbolic expression.

        Returns
        -------
        CD2Wrapper
            The conditional distribution
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return u  # For independence copula, conditional distribution is just u

        # Formula for Clayton copula
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
        """
        Second partial derivative of CDF with respect to u.

        Parameters
        ----------
        u, v : float
            Values to evaluate at

        Returns
        -------
        sympy expression
            The second derivative
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return 0  # For independence copula, second derivative is 0

        # Formula for Clayton copula
        return sympy.Heaviside((-1 + v ** (-theta) + u ** (-theta)) ** (-1 / theta)) / (
            u**2
            * u ** (2 * theta)
            * (-1 + v ** (-theta) + u ** (-theta)) ** 2
            * (-1 + v ** (-theta) + u ** (-theta)) ** (2 / theta)
        )

    @property
    def pdf(self):
        """
        Probability density function of the bivariate Clayton copula.

        Returns
        -------
        SymPyFuncWrapper
            The PDF function c(u,v)
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return SymPyFuncWrapper(1)  # Uniform density for independence copula

        # Formula for Clayton copula
        result = (
            (self.u ** (-theta) + self.v ** (-theta) - 1) ** (-2 - 1 / theta)
            * self.u ** (-theta - 1)
            * self.v ** (-theta - 1)
            * (theta + 1)
        )
        return SymPyFuncWrapper(result)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        Returns
        -------
        bool
            True for θ ≥ 0, False otherwise
        """
        return self.theta >= 0

    def lambda_L(self):
        """
        Lower tail dependence coefficient.

        For Clayton copula with θ > 0, this is 2^(-1/θ).

        Returns
        -------
        float or sympy expression
            The lower tail dependence coefficient
        """
        # Clayton with θ = 0 has no tail dependence
        if self.theta == 0:
            return 0
        # Formula for Clayton with θ > 0
        return 2 ** (-1 / self.theta)

    def lambda_U(self):
        """
        Upper tail dependence coefficient.

        For Clayton copula, this is always 0.

        Returns
        -------
        float
            Always 0 for Clayton
        """
        return 0


# Type alias for backwards compatibility
Clayton: TypeAlias = BivClayton
Nelsen1: TypeAlias = BivClayton
