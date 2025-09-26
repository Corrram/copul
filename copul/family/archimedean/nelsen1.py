import numpy as np
import sympy
from typing import Optional, TypeAlias

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


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

    def rvs(
        self, n: int = 1, random_state: Optional[int] = None, approximate: bool = False
    ) -> np.ndarray:
        """
        Generate random samples from the Clayton copula using a fast, vectorized algorithm.

        This method overrides the slow, iterative solver from the parent class. It uses a
        numerically stable, closed-form inverse of the conditional distribution, allowing
        for thousands of samples to be generated almost instantly.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        random_state : int, optional
            Seed for the random number generator for reproducibility.
        approximate : bool
            This parameter is ignored as the exact vectorized method is always fast.

        Returns
        -------
        numpy.ndarray
            Array of shape (n, 2) containing the generated samples.
        """
        rng = np.random.default_rng(random_state)
        w = rng.random((n, 2))

        theta_val = float(self.theta)

        # Handle special cases for independence and countermonotonicity
        if np.isclose(theta_val, 0):
            return w
        if np.isclose(theta_val, -1):
            u = w[:, 0]
            v = 1 - u
            return np.column_stack((u, v))

        u = w[:, 0]
        w2 = w[:, 1]

        # Use the closed-form inverse of the conditional distribution C(v|u) = w2
        # Formula: v = [u**(-theta) * (w2**(-theta / (theta + 1)) - 1) + 1]**(-1 / theta)
        term1 = w2 ** (-theta_val / (theta_val + 1)) - 1
        term2 = u ** (-theta_val)
        v = (term2 * term1 + 1) ** (-1 / theta_val)

        return np.column_stack((u, v))

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
        return CD1Wrapper(cond_distr)(u, v)

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


if __name__ == "__main__":
    # Example usage
    copula = BivClayton(theta=1)
    result = copula.cdf(u=0.5)
    # Test the new rvs method
    print("Generating 5 samples with the fast rvs method:")
    samples = copula.rvs(5, random_state=42)
    print(samples)

    copula_neg = BivClayton(theta=-0.8)
    print("\nGenerating 5 samples with a negative theta:")
    samples_neg = copula_neg.rvs(5, random_state=42)
    print(samples_neg)

    print("\nDone!")
