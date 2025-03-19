import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

from copul.checkerboard.check_pi import CheckPi
from copul.checkerboard.check_min import CheckMin
from copul.checkerboard.biv_check_w import BivCheckW


log = logging.getLogger(__name__)


class Checkerboarder:
    def __init__(self, n: Union[int, list] = None, dim=2, checkerboard_type="CheckPi"):
        """
        Initialize a Checkerboarder instance.

        Parameters
        ----------
        n : int or list, optional
            Number of grid partitions per dimension. If an integer is provided,
            the same number of partitions is used for each dimension.
            If None, defaults to 20 partitions per dimension.
        dim : int, optional
            The number of dimensions for the checkerboard grid.
            Defaults to 2.

        Notes
        -----
        Pre-calculates grid points for each dimension upon initialization.
        """
        if n is None:
            n = 20
        if isinstance(n, (int, np.int_)):
            n = [n] * dim
        self.n = n
        self.d = len(self.n)
        self._checkerboard_type = checkerboard_type
        # Pre-compute common grid points for each dimension.
        self._precalculate_grid_points()

    def _precalculate_grid_points(self):
        """
        Pre-calculate grid points for each dimension.

        Computes linearly spaced grid points on the interval [0, 1] for each dimension
        and stores them in the instance attribute 'grid_points' to avoid redundant calculations.
        """
        self.grid_points = []
        for n_i in self.n:
            points = np.linspace(0, 1, n_i + 1)
            self.grid_points.append(points)

    def get_checkerboard_copula(self, copula, n_jobs=None):
        """
        Compute the checkerboard representation of a copula's CDF.

        The method computes a discrete approximation (checkerboard) of the copula by
        evaluating the CDF over a grid defined by pre-calculated grid points. It uses
        vectorized computation for 2D cases if available; otherwise, it defaults to parallel
        or serial processing based on grid size and the n_jobs parameter.

        Parameters
        ----------
        copula : object
            A copula object that must provide a 'cdf' method, and optionally a
            'cdf_vectorized' method for 2D cases.
        n_jobs : int, optional
            The number of parallel jobs to use. If None, the number of jobs is determined
            automatically based on the grid size. Setting n_jobs to 1 disables parallelization.

        Returns
        -------
        Checkerboard copula of type self._checkerboard_type
            The computed checkerboard representation.
        """
        log.debug("Computing checkerboard copula with grid sizes: %s", self.n)

        # Use vectorized computation for 2D if available.
        if hasattr(copula, "cdf_vectorized") and self.d <= 2:
            return self._compute_checkerboard_vectorized(copula)

        # Determine parallelization based on grid size.
        if n_jobs is None:
            total_cells = np.prod(self.n)
            n_jobs = max(1, min(8, total_cells // 1000))

        if n_jobs > 1 and np.prod(self.n) > 100:
            return self._compute_checkerboard_parallel(copula, n_jobs)
        else:
            return self._compute_checkerboard_serial(copula)

    def _compute_checkerboard_vectorized(self, copula):
        """
        Compute the checkerboard representation using fully vectorized operations (2D only).

        Uses the copula's cdf_vectorized method to compute all grid cell values in a single
        operation, significantly improving performance compared to cell-by-cell evaluation.

        Parameters
        ----------
        copula : object
            A copula object providing a 'cdf_vectorized' method.

        Returns
        -------
        CheckPi or BivCheckPi
            The checkerboard representation computed via vectorized operations.
        """
        if self.d != 2:
            warnings.warn("Vectorized computation only supported for 2D case")
            return self._compute_checkerboard_serial(copula)

        # Define grid edges for both dimensions
        x_lower = self.grid_points[0][:-1]
        x_upper = self.grid_points[0][1:]
        y_lower = self.grid_points[1][:-1]
        y_upper = self.grid_points[1][1:]

        # Create meshgrids for all corner combinations
        X_lower, Y_lower = np.meshgrid(x_lower, y_lower, indexing="ij")
        X_upper, Y_upper = np.meshgrid(x_upper, y_upper, indexing="ij")

        # Apply inclusion-exclusion principle using vectorized CDF evaluation
        # C(u_upper, v_upper) - C(u_upper, v_lower) - C(u_lower, v_upper) + C(u_lower, v_lower)
        cdf_upper_upper = copula.cdf_vectorized(X_upper, Y_upper)
        cdf_lower_lower = copula.cdf_vectorized(X_lower, Y_lower)
        cdf_upper_lower = copula.cdf_vectorized(X_upper, Y_lower)
        cdf_lower_upper = copula.cdf_vectorized(X_lower, Y_upper)

        # Compute cell probabilities using the inclusion-exclusion principle
        cmatr = cdf_upper_upper - cdf_upper_lower - cdf_lower_upper + cdf_lower_lower
        # round matr entries
        rounded_cmatr = np.clip(cmatr, 0, 1)
        return self._get_checkerboard_copula_for(rounded_cmatr)

    def _get_checkerboard_copula_for(self, cmatr):
        if self._checkerboard_type in ["CheckPi", "BivCheckPi"]:
            return CheckPi(cmatr)
        elif self._checkerboard_type in ["CheckMin", "BivCheckMin"]:
            return CheckMin(cmatr)
        elif self._checkerboard_type == ["BivCheckW", "CheckW"]:
            return BivCheckW(cmatr)
        else:
            raise ValueError(f"Unknown checkerboard type: {self._checkerboard_type}")

    def _compute_checkerboard_parallel(self, copula, n_jobs):
        """
        Compute the checkerboard representation in parallel.

        Divides the grid into cells, processes each cell concurrently, and then reassembles
        the results into the final checkerboard matrix.

        Parameters
        ----------
        copula : object
            A copula object providing a 'cdf' method.
        n_jobs : int
            Number of parallel jobs to use.

        Returns
        -------
        CheckPi or BivCheckPi
            The resulting checkerboard representation. For dimensions >2, returns a CheckPi;
            for 2D, returns a BivCheckPi.
        """
        # Generate list of indices for all grid cells.
        indices = list(np.ndindex(*self.n))

        # Process each cell in parallel.
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_cell)(idx, copula) for idx in indices
        )

        # Assemble results into a matrix.
        cmatr = np.zeros(self.n)
        for idx, value in zip(indices, results):
            cmatr[idx] = value

        return self._get_checkerboard_copula_for(cmatr)

    def _process_cell(self, idx, copula):
        """
        Process a single grid cell for parallel computation.

        For the given cell index, the method determines the corresponding hypercube
        (using precomputed grid points) and computes its value using inclusion-exclusion.

        Parameters
        ----------
        idx : tuple of int
            Indices representing the cell location in the grid.
        copula : object
            A copula object with a 'cdf' method.

        Returns
        -------
        float
            The computed CDF-based value for the cell.
        """
        u_lower = [self.grid_points[k][i] for k, i in enumerate(idx)]
        u_upper = [self.grid_points[k][i + 1] for k, i in enumerate(idx)]
        return self._compute_cell_value(u_lower, u_upper, copula)

    def _compute_checkerboard_serial(self, copula):
        """
        Compute the checkerboard representation serially with caching.

        Uses an internal cache for CDF evaluations to reduce redundant calculations,
        applying the inclusion-exclusion principle for each grid cell.

        Parameters
        ----------
        copula : object
            A copula object providing a 'cdf' method.

        Returns
        -------
        CheckPi or BivCheckPi
            The computed checkerboard representation. Returns a CheckPi for d > 2,
            otherwise a BivCheckPi.
        """
        cdf_cache = {}
        cmatr = np.zeros(self.n)
        indices = np.ndindex(*self.n)

        def get_cached_cdf(point):
            point_tuple = tuple(point)
            if point_tuple not in cdf_cache:
                cdf_value = copula.cdf(*point)
                if not isinstance(cdf_value, (float, int)):
                    cdf_value = cdf_value.evalf()
                cdf_cache[point_tuple] = cdf_value
            return cdf_cache[point_tuple]

        for idx in indices:
            u_lower = [self.grid_points[k][i] for k, i in enumerate(idx)]
            u_upper = [self.grid_points[k][i + 1] for k, i in enumerate(idx)]
            inclusion_exclusion_sum = 0
            # Iterate over all 2^d corners of the hypercube.
            for corner in range(1 << self.d):
                corner_indices = [
                    (u_upper[k] if corner & (1 << k) else u_lower[k])
                    for k in range(self.d)
                ]
                sign = (-1) ** (bin(corner).count("1") + 2)
                cdf_value = get_cached_cdf(corner_indices)
                inclusion_exclusion_sum += sign * cdf_value
            cmatr[idx] = inclusion_exclusion_sum

        return self._get_checkerboard_copula_for(cmatr)

    def _compute_cell_value(self, u_lower, u_upper, copula):
        """
        Compute the value for a single grid cell using the inclusion-exclusion principle.

        Evaluates the copula's CDF at each corner of the hypercube defined by u_lower and u_upper,
        applying alternating signs based on the inclusion-exclusion principle.

        Parameters
        ----------
        u_lower : list of float
            Lower bounds for the cell in each dimension.
        u_upper : list of float
            Upper bounds for the cell in each dimension.
        copula : object
            A copula object with a 'cdf' method.

        Returns
        -------
        float
            The computed cell value.
        """
        inclusion_exclusion_sum = 0
        for corner in range(1 << self.d):
            corner_indices = [
                (u_upper[k] if corner & (1 << k) else u_lower[k]) for k in range(self.d)
            ]
            sign = (-1) ** (bin(corner).count("1") + 2)
            try:
                cdf_value = copula.cdf(*corner_indices).evalf()
                inclusion_exclusion_sum += sign * cdf_value
            except Exception as e:
                log.warning(f"Error computing CDF at {corner_indices}: {e}")
        return inclusion_exclusion_sum

    def from_data(self, data: Union[pd.DataFrame, np.ndarray, list]):
        """
        Create a checkerboard copula from empirical data.

        Converts input data to a DataFrame (if needed), computes rank-transformed values using a
        fast numba-accelerated function, and then builds a checkerboard density estimate.

        Parameters
        ----------
        data : pd.DataFrame, np.ndarray, or list
            The empirical data used to estimate the copula.

        Returns
        -------
        CheckPi or BivCheckPi
            A checkerboard copula representation derived from the data.
        """
        if isinstance(data, (list, np.ndarray)):
            data = pd.DataFrame(data)

        n_obs = len(data)
        rank_data = np.empty_like(data.values, dtype=float)
        for i, col in enumerate(data.columns):
            rank_data[:, i] = _fast_rank(data[col].values)

        rank_df = pd.DataFrame(rank_data, columns=data.columns)

        if self.d == 2:
            return self._from_data_bivariate(rank_df, n_obs)
        else:
            check_pi_matr = np.zeros(self.n)
            return CheckPi(check_pi_matr)

    def _from_data_bivariate(self, data, n_obs):
        """
        Construct a bivariate checkerboard copula from rank-transformed data.

        Uses numpy’s histogram2d function to efficiently bin the data and normalizes the result.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing bivariate rank-transformed data.
        n_obs : int
            Total number of observations.

        Returns
        -------
        BivCheckPi
            A bivariate checkerboard copula representation.
        """
        check_pi_matr = np.zeros((self.n[0], self.n[1]))
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values

        # Compute histogram bins for the data.
        hist, _, _ = np.histogram2d(
            x, y, bins=[self.n[0], self.n[1]], range=[[0, 1], [0, 1]]
        )
        check_pi_matr = hist / n_obs
        return self._get_checkerboard_copula_for(check_pi_matr)


@njit
def _fast_rank(x):
    """
    Compute percentage ranks for a 1D numpy array using numba for speed.

    Parameters
    ----------
    x : np.ndarray
        A one-dimensional array of numerical values.

    Returns
    -------
    np.ndarray
        An array of normalized ranks in the range [0, 1].
    """
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    idx = np.argsort(x)
    for i in range(n):
        ranks[idx[i]] = (i + 1) / n
    return ranks


def from_data(data, checkerboard_size=None):
    """
    Create a checkerboard copula from empirical data using an adaptive grid size.

    This is an optimized wrapper that determines the grid size based on the number of samples,
    constructs a Checkerboarder instance, and generates the corresponding copula.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or list
        The empirical data from which to build the copula.
    checkerboard_size : int, optional
        Number of grid partitions per dimension. If not provided, an adaptive value is computed
        based on the data size.

    Returns
    -------
    CheckPi or BivCheckPi
        The constructed checkerboard copula representation.
    """
    if checkerboard_size is None:
        n_samples = len(data)
        checkerboard_size = min(max(10, int(np.sqrt(n_samples) / 5)), 50)

    dimensions = data.shape[1] if hasattr(data, "shape") else len(data[0])
    return Checkerboarder(checkerboard_size, dimensions).from_data(data)
