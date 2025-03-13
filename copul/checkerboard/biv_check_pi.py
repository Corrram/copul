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

    def tau(self, n_samples=2_000_000, grid_size=500) -> float:
        """
        Compute Kendall's tau using optimized Monte Carlo integration.

        Parameters
        ----------
        n_samples : int
            Number of samples for Monte Carlo integration.
        grid_size : int
            Size of grid for precomputing conditional distributions.

        Returns
        -------
        float
            Kendall's tau value.
        """
        # For bivariate case only (2D)
        if len(self.dim) != 2:
            raise ValueError(
                "This optimized tau2 implementation is for bivariate copulas only"
            )

        # Generate a grid of points for precomputation
        grid_points = (
            np.linspace(0, 1, grid_size + 1)[:-1] + 0.5 / grid_size
        )  # Midpoints

        # Precompute conditional distributions on the grid
        cd1_grid = np.zeros((grid_size, grid_size))
        cd2_grid = np.zeros((grid_size, grid_size))

        for i, x in enumerate(grid_points):
            for j, y in enumerate(grid_points):
                cd1_grid[i, j] = self.cond_distr(1, (x, y))
                cd2_grid[i, j] = self.cond_distr(2, (x, y))

        # Define a fast interpolation function for the grids
        def fast_interp2d(grid, x, y):
            """Fast 2D interpolation on a regular grid."""
            ix = np.clip(np.floor(x * grid_size).astype(int), 0, grid_size - 1)
            iy = np.clip(np.floor(y * grid_size).astype(int), 0, grid_size - 1)
            return grid[ix, iy]

        # Vectorized Monte Carlo integration
        rng = np.random.default_rng()
        xs = rng.random(n_samples)
        ys = rng.random(n_samples)

        # Use the interpolated conditional distributions
        cd1_values = fast_interp2d(cd1_grid, xs, ys)
        cd2_values = fast_interp2d(cd2_grid, xs, ys)

        # Compute the product and take the mean (equivalent to the double integral)
        result = np.mean(cd1_values * cd2_values)

        # Apply the formula for Kendall's tau
        return 1 - 4 * result

    def rho(self, n_samples=1_000_000, grid_size=50) -> float:
        """
        Compute Spearman's rho using optimized Monte Carlo integration.

        Parameters
        ----------
        n_samples : int
            Number of samples for Monte Carlo integration.
        grid_size : int
            Size of grid for precomputing CDF values.

        Returns
        -------
        float
            Spearman's rho value.
        """
        # For bivariate case only (2D)
        if len(self.dim) != 2:
            raise ValueError(
                "This optimized rho implementation is for bivariate copulas only"
            )

        # Generate a grid of points for precomputation
        grid_points = (
            np.linspace(0, 1, grid_size + 1)[:-1] + 0.5 / grid_size
        )  # Midpoints

        # Precompute CDF values on the grid
        cdf_grid = np.zeros((grid_size, grid_size))

        for i, x in enumerate(grid_points):
            for j, y in enumerate(grid_points):
                cdf_grid[i, j] = self.cdf(x, y)

        # Define a fast interpolation function for the grid
        def fast_interp2d(grid, x, y):
            """Fast 2D interpolation on a regular grid."""
            ix = np.clip(np.floor(x * grid_size).astype(int), 0, grid_size - 1)
            iy = np.clip(np.floor(y * grid_size).astype(int), 0, grid_size - 1)
            return grid[ix, iy]

        # Vectorized Monte Carlo integration
        rng = np.random.default_rng()
        xs = rng.random(n_samples)
        ys = rng.random(n_samples)

        # Use the interpolated CDF values
        cdf_values = fast_interp2d(cdf_grid, xs, ys)

        # Compute the mean (equivalent to the double integral)
        result = np.mean(cdf_values)

        # Apply the formula for Spearman's rho
        return 12 * result - 3

    def chatterjees_xi(
        self,
        n_samples: int = 1_000_000,
        condition_on_y: bool = False,
        grid_size: int = 50,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """
        Compute Chatterjee's xi correlation measure using optimized Monte Carlo integration.

        Parameters
        ----------
        n_samples : int
            Number of samples for Monte Carlo integration.
        condition_on_y : bool
            If True, condition on Y (U2) instead of X (U1).
        grid_size : int
            Size of grid for precomputing conditional distributions.
        *args, **kwargs:
            Additional arguments passed to _set_params.

        Returns
        -------
        float
            Chatterjee's xi correlation value in the range [0, 1].
        """
        self._set_params(args, kwargs)

        # For bivariate case only (2D)
        if len(self.dim) != 2:
            raise ValueError(
                "This optimized implementation is for bivariate copulas only"
            )

        # Determine which dimension to condition on
        i = 2 if condition_on_y else 1

        # Generate a grid of points for precomputation
        grid_points = (
            np.linspace(0, 1, grid_size + 1)[:-1] + 0.5 / grid_size
        )  # Midpoints

        # Precompute squared conditional distributions on the grid
        cond_distr_grid = np.zeros((grid_size, grid_size))

        for ii, x in enumerate(grid_points):
            for jj, y in enumerate(grid_points):
                # Compute the squared conditional distribution directly
                cond_val = self.cond_distr(i, (x, y))
                cond_distr_grid[ii, jj] = cond_val * cond_val

        # Define a fast interpolation function for the grid
        def fast_interp2d(grid, x, y):
            """Fast 2D interpolation on a regular grid."""
            ix = np.clip(np.floor(x * grid_size).astype(int), 0, grid_size - 1)
            iy = np.clip(np.floor(y * grid_size).astype(int), 0, grid_size - 1)
            return grid[ix, iy]

        # Vectorized Monte Carlo integration
        rng = np.random.default_rng()
        xs = rng.random(n_samples)
        ys = rng.random(n_samples)

        # Use the interpolated values
        values = fast_interp2d(cond_distr_grid, xs, ys)

        # Compute the mean (equivalent to the integral)
        result = np.mean(values)

        # Apply the formula for Chatterjee's xi
        xi_value = 6 * result - 2

        # Ensure the result is in [0, 1]
        return max(0, min(1, xi_value))


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
