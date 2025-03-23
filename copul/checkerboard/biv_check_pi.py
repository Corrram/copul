"""
Bivariate Checkerboard Copula module.

This module provides a bivariate checkerboard copula implementation
that combines properties of both CheckPi and BivCopula classes.
"""

import numpy as np
from typing import Union, List, Optional, Any
import warnings

import sympy
from copul.checkerboard.check_pi import CheckPi
from copul.families.core.biv_core_copula import BivCoreCopula
from copul.families.core.copula_plotting_mixin import CopulaPlottingMixin


class BivCheckPi(CheckPi, BivCoreCopula, CopulaPlottingMixin):
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
        if isinstance(matr, sympy.Matrix):
            matr = np.array(matr).astype(float)

        # Input validation
        if not hasattr(matr, "ndim"):
            raise ValueError("Input matrix must be a 2D array or list")
        if matr.ndim != 2:
            raise ValueError(
                f"Input matrix must be 2-dimensional, got {matr.ndim} dimensions"
            )
        if np.any(matr < 0):
            raise ValueError("All matrix values must be non-negative")

        CheckPi.__init__(self, matr)
        BivCoreCopula.__init__(self, **kwargs)

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
        self, *args
    ):
        return self.cond_distr(1, *args)

    def cond_distr_2(self, *args):
        return self.cond_distr(2, *args)

    def rho(self) -> float:
        """
        Compute Spearman's rho for a bivariate checkerboard copula.
        
        Uses the formula:
        ρ = 12 * (Σ₍i,j₎ p[i,j]·((2i+1)(2j+1))/(4*m*n)) - 3.
        
        This version creates index arrays and computes the sum in one shot.
        """
        p = np.asarray(self.matr, dtype=float)
        m, n = p.shape
        # Compute the factors (2*(i+1)-1)=2i+1 for rows and columns:
        I = 2 * np.arange(m) + 1  # shape (m,)
        J = 2 * np.arange(n) + 1  # shape (n,)
        # Outer product produces a matrix of shape (m,n) with entry (i,j) = (2i+1)(2j+1)
        prod = np.outer(I, J)
        uv_sum = np.sum(p * prod) / (m * n)
        return 3 * uv_sum - 3

    def tau(self) -> float:
        """
        Calculate the tau coefficient more efficiently using numpy's vectorized operations.
        
        Returns:
            float: The calculated tau coefficient.
        """
        Xi_m = 2 *np.tri(self.m) - np.eye(self.m)
        Xi_n = 2 * np.tri(self.n) - np.eye(self.n)
        return 1 - np.trace(Xi_m @ self.matr @ Xi_n @ self.matr.T)

    def xi(self, condition_on_y: bool = False) -> float:
        if condition_on_y:
            delta = self.matr.T
            m = self.n
            n = self.m
        else:
            delta = self.matr
            m = self.m
            n = self.n
        T = np.ones(n) - np.tri(n)
        M = T @ T.T + T.T + 1/3*np.eye(n)
        trace = np.trace(delta.T @ delta @ M)
        xi = 6*m/n * trace - 2
        return xi

    def xi_(self, condition_on_y: bool = False) -> float:
        """
        Compute Chatterjee's xi via a closed-form formula.
        
        For each cell (i,j) define the "prior" as follows:
        - If condition_on_y is True, let prior = sum_{r < i} p[r,j] (cumulative in rows).
        - Otherwise, let prior = sum_{c < j} p[i,c] (cumulative in columns).
        
        Then the cell's contribution is:
            contribution = prior^2 + prior * p[i,j] + (1/3)*(p[i,j]**2).
        
        Finally, xi = 6 * (sum of contributions) - 2.
        
        This implementation uses vectorized cumulative sums.
        """
        p = np.asarray(self.matr, dtype=float)
        m, n = p.shape

        if condition_on_y:
            # Cumulative sum along rows (for each column)
            prior = np.vstack([np.zeros((1, n)), np.cumsum(p, axis=0)[:-1, :]])
            contrib = prior**2 + prior * p + (p**2) / 3.0
            result = 6 * np.sum(contrib)*n/m - 2
        else:
            # Cumulative sum along columns (for each row)
            prior = np.hstack([np.zeros((m, 1)), np.cumsum(p, axis=1)[:, :-1]])
            contrib = prior**2 + prior * p + (p**2) / 3.0
            result = 6 * np.sum(contrib)*m/n - 2
        return result


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
    print(f"Chatterjee's xi: {copul.xi(n_samples=10_000):.4f}")

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
