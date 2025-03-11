"""
Bivariate Checkerboard Copula module.

This module provides a bivariate checkerboard copula implementation
that combines properties of both CheckPi and BivCopula classes.
"""

import numpy as np
from typing import Union, List, Optional, Any
import warnings

from copul import basictools
from copul.checkerboard.check_pi import CheckPi
from copul.families.bivcopula import BivCopula


class BivCheckPi(CheckPi, BivCopula):
    """
    Bivariate Checkerboard Copula class.

    This class implements a bivariate checkerboard copula, which is defined by
    a matrix of values that determine the copula's distribution.

    Attributes:
        params (List): Empty list as checkerboard copulas are non-parametric.
        intervals (dict): Empty dictionary as there are no parameters to bound.
        m (int): Number of rows in the checkerboard matrix.
        n (int): Number of columns in the checkerboard matrix.
    """

    params: List = []
    intervals: dict = {}

    def __init__(self, matr: Union[List[List[float]], np.ndarray], **kwargs):
        """
        Initialize a bivariate checkerboard copula.

        Args:
            matr: A matrix (2D array) defining the checkerboard distribution.
            **kwargs: Additional parameters passed to BivCopula.

        Raises:
            ValueError: If matrix dimensions are invalid or matrix contains negative values.
        """
        # Convert input to numpy array if it's a list
        if isinstance(matr, list):
            matr = np.array(matr, dtype=float)

        # Input validation
        if matr.ndim != 2:
            raise ValueError(
                f"Input matrix must be 2-dimensional, got {matr.ndim} dimensions"
            )
        if np.any(matr < 0):
            raise ValueError("All matrix values must be non-negative")

        CheckPi.__init__(self, matr)
        BivCopula.__init__(self, **kwargs)

        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]

        # Normalize matrix if not already normalized
        if not np.isclose(np.sum(self.matr), 1.0):
            warnings.warn(
                "Matrix not normalized. Normalizing to ensure proper density.",
                UserWarning,
            )
            self.matr = self.matr / np.sum(self.matr)

    def __str__(self) -> str:
        """
        Return a string representation of the copula.

        Returns:
            str: String representation showing dimensions of the checkerboard.
        """
        return f"BivCheckPi(m={self.m}, n={self.n})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation for debugging.

        Returns:
            str: Detailed representation including matrix information.
        """
        return f"BivCheckPi(matr={self.matr.tolist()}, m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric (C(u,v) = C(v,u)).

        Returns:
            bool: True if the copula is symmetric, False otherwise.
        """
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        For checkerboard copulas, this property is always True.

        Returns:
            bool: Always True for checkerboard copulas.
        """
        return True

    def cond_distr_1(
        self, u: Optional[float] = None, v: Optional[float] = None
    ) -> float:
        """
        Compute the conditional distribution F(U1 ≤ u | U2 = v).

        Args:
            u: Value of U1 (first variable).
            v: Value of U2 (second variable).

        Returns:
            float: Conditional probability.
        """
        return self.cond_distr(1, (u, v))

    def cond_distr_2(
        self, u: Optional[float] = None, v: Optional[float] = None
    ) -> float:
        """
        Compute the conditional distribution F(U2 ≤ v | U1 = u).

        Args:
            u: Value of U1 (first variable).
            v: Value of U2 (second variable).

        Returns:
            float: Conditional probability.
        """
        return self.cond_distr(2, (u, v))

    def tau(self) -> float:
        """
        Compute Kendall's tau for the checkerboard copula.

        Kendall's tau is a measure of ordinal association between two random variables.

        Returns:
            float: Kendall's tau value.
        """
        result = basictools.monte_carlo_integral(
            lambda x, y: self.cdf(x, y) * self.pdf(x, y),
            n_samples=10_000,  # Using a reasonable default
        )
        return 4 * result - 1

    def rho(self) -> float:
        """
        Compute Spearman's rho for the checkerboard copula.

        Spearman's rho is a measure of rank correlation between two random variables.

        Returns:
            float: Spearman's rho value.
        """
        result = basictools.monte_carlo_integral(
            lambda x, y: self.cdf(x, y),
            n_samples=10_000,  # Using a reasonable default
        )
        return 12 * result - 3

    def chatterjees_xi(
        self,
        n_samples: int = 100_000,
        condition_on_y: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """
        Compute Chatterjee's xi correlation measure.

        Args:
            n_samples: Number of samples for Monte Carlo integration.
            condition_on_y: If True, condition on Y (U2) instead of X (U1).
            *args: Additional positional arguments passed to _set_params.
            **kwargs: Additional keyword arguments passed to _set_params.

        Returns:
            float: Chatterjee's xi correlation value.
        """
        self._set_params(args, kwargs)
        i = 2 if condition_on_y else 1

        def f(x: float, y: float) -> float:
            return self.cond_distr(i, (x, y)) ** 2

        result = basictools.monte_carlo_integral(
            f, n_samples=n_samples, vectorized=False
        )
        return 6 * result - 2

    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the checkerboard copula.

        Args:
            n: Number of samples to generate.
            random_state: Seed for random number generator.

        Returns:
            np.ndarray: Array of shape (n, 2) containing the sampled (u, v) pairs.
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Generate uniformly distributed samples
        u_samples = np.random.uniform(0, 1, n)
        v_samples = np.zeros(n)

        # For each u, sample v from the conditional distribution
        for i, u in enumerate(u_samples):
            # Find the corresponding bin for u
            u_bin = int(np.floor(u * self.m))
            if u_bin == self.m:  # Handle edge case
                u_bin = self.m - 1

            # Calculate the conditional distribution for this u
            cond_probs = self.matr[u_bin, :] / np.sum(self.matr[u_bin, :])

            # Sample from the conditional distribution
            v_bin = np.random.choice(self.n, p=cond_probs)

            # Convert bin index to uniform value (add some jitter within the bin)
            v_samples[i] = (v_bin + np.random.uniform(0, 1)) / self.n

        # Combine u and v samples
        return np.column_stack((u_samples, v_samples))


if __name__ == "__main__":
    # Example usage
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    copul = BivCheckPi(matr)

    # Basic properties
    print(f"Copula: {copul}")
    print(f"Is symmetric: {copul.is_symmetric}")

    # Generate samples
    samples = copul.sample(1000, random_state=42)

    # Calculate dependence measures
    print(f"Kendall's tau: {copul.tau():.4f}")
    print(f"Spearman's rho: {copul.rho():.4f}")
    print(f"Chatterjee's xi: {copul.chatterjees_xi(n_samples=10_000):.4f}")

    # Visualize conditional distribution
    try:
        import matplotlib.pyplot as plt

        # Plot samples
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
        plt.title("Samples from Checkerboard Copula")
        plt.xlabel("U1")
        plt.ylabel("U2")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        copul.plot_cond_distr_1()

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization.")
