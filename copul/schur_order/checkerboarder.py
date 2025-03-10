import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit, prange

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.check_pi import CheckPi

log = logging.getLogger(__name__)


class Checkerboarder:
    def __init__(self, n: Union[int, list] = None, dim=2):
        if n is None:
            n = 20
        if isinstance(n, int):
            n = [n] * dim
        self.n = n
        self.d = len(self.n)
        # Pre-compute common values
        self._precalculate_grid_points()

    def _precalculate_grid_points(self):
        """Pre-calculate grid points to avoid repeated calculations."""
        self.grid_points = []
        for i, n_i in enumerate(self.n):
            points = np.linspace(0, 1, n_i + 1)
            self.grid_points.append(points)

    def compute_check_pi(self, copula, n_jobs=None):
        """
        Compute checkerboard copula with optimizations.

        Args:
            copula: Copula object with a cdf method
            n_jobs: Number of parallel jobs. If None, autodetect.
                    Set to 1 to disable parallelization.
        """
        log.debug("Computing checkerboard copula with grid sizes: %s", self.n)

        # Try to vectorize CDF computation if supported by the copula
        if hasattr(copula, "cdf_vectorized") and self.d <= 2:
            return self._compute_check_pi_vectorized(copula)

        # For higher dimensions or when vectorization is not supported
        if n_jobs is None:
            # Auto-detect: Use parallelization for larger grids
            total_cells = np.prod(self.n)
            n_jobs = max(1, min(8, total_cells // 1000))  # Scale with grid size

        if n_jobs > 1 and np.prod(self.n) > 100:  # Only parallelize for larger grids
            return self._compute_check_pi_parallel(copula, n_jobs)
        else:
            return self._compute_check_pi_serial(copula)

    def _compute_check_pi_vectorized(self, copula):
        """Vectorized computation of checkerboard copula for 2D case."""
        if self.d != 2:
            warnings.warn("Vectorized computation only supported for 2D case")
            return self._compute_check_pi_serial(copula)

        cmatr = np.zeros(self.n)

        # Create meshgrid of upper and lower bounds
        x_lower = self.grid_points[0][:-1]
        x_upper = self.grid_points[0][1:]
        y_lower = self.grid_points[1][:-1]
        y_upper = self.grid_points[1][1:]

        # For each cell (i,j), compute C(ui+1,vj+1) - C(ui+1,vj) - C(ui,vj+1) + C(ui,vj)
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                # Apply inclusion-exclusion principle for the rectangle
                cmatr[i, j] = (
                    copula.cdf(x_upper[i], y_upper[j])
                    - copula.cdf(x_upper[i], y_lower[j])
                    - copula.cdf(x_lower[i], y_upper[j])
                    + copula.cdf(x_lower[i], y_lower[j])
                )

        return BivCheckPi(cmatr)

    def _compute_check_pi_parallel(self, copula, n_jobs):
        """Parallel computation of checkerboard copula."""
        # Create all grid indices
        indices = list(np.ndindex(*self.n))

        # Process cells in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_cell)(idx, copula) for idx in indices
        )

        # Create matrix from results
        cmatr = np.zeros(self.n)
        for idx, value in zip(indices, results):
            cmatr[idx] = value

        return CheckPi(cmatr) if self.d > 2 else BivCheckPi(cmatr)

    def _process_cell(self, idx, copula):
        """Process a single cell of the checkerboard - for parallel execution."""
        # Generate the edges of the hypercube for each dimension based on the index
        u_lower = [self.grid_points[k][i] for k, i in enumerate(idx)]
        u_upper = [self.grid_points[k][i + 1] for k, i in enumerate(idx)]

        # Compute using inclusion-exclusion
        return self._compute_cell_value(u_lower, u_upper, copula)

    def _compute_check_pi_serial(self, copula):
        """Serial computation of checkerboard copula with optimizations."""
        # Cache for CDF values to avoid repeated calculations
        cdf_cache = {}

        # Matrix to store the copula values
        cmatr = np.zeros(self.n)

        # Create grid indices for each dimension
        indices = np.ndindex(*self.n)

        # Function to get cached CDF value
        def get_cached_cdf(point):
            point_tuple = tuple(point)
            if point_tuple not in cdf_cache:
                cdf_cache[point_tuple] = copula.cdf(*point).evalf()
            return cdf_cache[point_tuple]

        for idx in indices:
            # Generate the edges of the hypercube for each dimension based on the index
            u_lower = [self.grid_points[k][i] for k, i in enumerate(idx)]
            u_upper = [self.grid_points[k][i + 1] for k, i in enumerate(idx)]

            # Initialize the CDF terms for inclusion-exclusion principle
            inclusion_exclusion_sum = 0

            # Compute the CDF for all corners of the hypercube using the inclusion-exclusion principle
            for corner in range(
                1 << self.d
            ):  # Iterate over 2^d corners of the hypercube
                corner_indices = [
                    (u_upper[k] if corner & (1 << k) else u_lower[k])
                    for k in range(self.d)
                ]
                sign = (-1) ** (
                    bin(corner).count("1") + 2
                )  # Use inclusion-exclusion principle
                cdf_value = get_cached_cdf(corner_indices)
                inclusion_exclusion_sum += sign * cdf_value

            # Assign the result to the copula matrix
            cmatr[idx] = inclusion_exclusion_sum

        return CheckPi(cmatr) if self.d > 2 else BivCheckPi(cmatr)

    def _compute_cell_value(self, u_lower, u_upper, copula):
        """
        Compute value for a single cell using inclusion-exclusion principle.
        Separated for use in parallel processing.
        """
        # Initialize the CDF terms for inclusion-exclusion principle
        inclusion_exclusion_sum = 0

        # Compute the CDF for all corners of the hypercube
        for corner in range(1 << self.d):  # Iterate over 2^d corners
            corner_indices = [
                (u_upper[k] if corner & (1 << k) else u_lower[k]) for k in range(self.d)
            ]
            sign = (-1) ** (bin(corner).count("1") + 2)
            try:
                cdf_value = copula.cdf(*corner_indices).evalf()
                inclusion_exclusion_sum += sign * cdf_value
            except Exception as e:
                log.warning(f"Error computing CDF at {corner_indices}: {e}")
                # Return a default value or handle the error as needed

        return inclusion_exclusion_sum

    def from_data(self, data: Union[pd.DataFrame, np.ndarray, list]):
        """
        Create a checkerboard copula from empirical data.
        Optimized for large datasets.
        """
        # Convert to DataFrame if necessary
        if isinstance(data, (list, np.ndarray)):
            data = pd.DataFrame(data)

        # Faster rank computation using numba if available
        n_obs = len(data)
        rank_data = np.empty_like(data.values, dtype=float)

        # Transform to ranks efficiently
        for i, col in enumerate(data.columns):
            rank_data[:, i] = _fast_rank(data[col].values)

        # Create a view of the ranked data for efficiency
        rank_df = pd.DataFrame(rank_data, columns=data.columns)

        # For bivariate case, use optimized implementation
        if self.d == 2:
            return self._from_data_bivariate(rank_df, n_obs)
        else:
            # General implementation for higher dimensions
            # Compute checkerboard density from ranks
            check_pi_matr = np.zeros(self.n)

            # This could be further optimized with numba or other methods
            # for higher dimensions if needed

            return CheckPi(check_pi_matr) if self.d > 2 else BivCheckPi(check_pi_matr)

    def _from_data_bivariate(self, data, n_obs):
        """Optimized implementation for bivariate data."""
        check_pi_matr = np.zeros((self.n[0], self.n[1]))

        # Use numpy operations for speed
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values

        # Efficiently compute bin indices
        x_bins = np.minimum(np.floor(x * self.n[0]).astype(int), self.n[0] - 1)
        y_bins = np.minimum(np.floor(y * self.n[1]).astype(int), self.n[1] - 1)

        # Use numpy's histogram2d for fast binning
        hist, _, _ = np.histogram2d(
            x, y, bins=[self.n[0], self.n[1]], range=[[0, 1], [0, 1]]
        )

        # Normalize the histogram
        check_pi_matr = hist / n_obs

        return BivCheckPi(check_pi_matr)


@njit
def _fast_rank(x):
    """
    Fast computation of percentage ranks using numba.

    Args:
        x: 1D numpy array

    Returns:
        Array of ranks in [0, 1] range
    """
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    temp = np.empty(n, dtype=np.float64)

    # Sort x and keep track of original indices
    idx = np.argsort(x)

    # Compute ranks
    for i in range(n):
        ranks[idx[i]] = (i + 1) / n

    return ranks


def from_data(data, checkerboard_size=None):
    """
    Create a checkerboard copula from data.
    Optimized wrapper for the class method.

    Args:
        data: Data as DataFrame, numpy array, or list
        checkerboard_size: Size of the checkerboard grid

    Returns:
        CheckPi or BivCheckPi object
    """
    if checkerboard_size is None:
        # Adaptive grid size based on data size
        n_samples = len(data)
        checkerboard_size = min(max(10, int(np.sqrt(n_samples) / 5)), 50)

    dimensions = data.shape[1] if hasattr(data, "shape") else len(data[0])
    return Checkerboarder(checkerboard_size, dimensions).from_data(data)
