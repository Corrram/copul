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

# Import your BernsteinCopula class:
# (Adjust the import path if it's located elsewhere.)
from copul.checkerboard.bernstein import BernsteinCopula

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
        checkerboard_type : str, optional
            Specifies which checkerboard-based copula class to return.
            Possible values include:
              - "CheckPi", "BivCheckPi"
              - "CheckMin", "BivCheckMin"
              - "CheckW", "BivCheckW"
              - "Bernstein", "BernsteinCopula"
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
        """Pre-calculate grid points for each dimension, linearly spaced in [0,1]."""
        self.grid_points = []
        for n_i in self.n:
            points = np.linspace(0, 1, n_i + 1)
            self.grid_points.append(points)

    def get_checkerboard_copula(self, copula, n_jobs=None):
        """
        Compute the checkerboard representation of a copula's CDF.

        The method computes a discrete approximation (checkerboard) of the copula by
        evaluating the CDF over a grid defined by pre-calculated grid points. It uses
        vectorized computation for 2D cases if available; otherwise, it defaults to
        parallel or serial processing.

        Parameters
        ----------
        copula : object
            A copula object that must provide a 'cdf' method, and optionally a
            'cdf_vectorized' method for 2D cases.
        n_jobs : int, optional
            Number of parallel jobs to use. If None, a heuristic is used based
            on the grid size.

        Returns
        -------
        CheckPi / CheckMin / BivCheckW / BernsteinCopula
            The computed checkerboard representation, depending on
            self._checkerboard_type.
        """
        log.debug("Computing checkerboard copula with grid sizes: %s", self.n)

        # If 2D and copula has a 'cdf_vectorized' method, do vectorized approach
        if hasattr(copula, "cdf_vectorized") and self.d == 2:
            return self._compute_checkerboard_vectorized(copula)

        # Otherwise, decide on serial vs parallel
        if n_jobs is None:
            total_cells = np.prod(self.n)
            # Simple heuristic: up to 8 jobs, and only if large enough
            n_jobs = max(1, min(8, total_cells // 1000))

        if n_jobs > 1 and np.prod(self.n) > 100:
            return self._compute_checkerboard_parallel(copula, n_jobs)
        else:
            return self._compute_checkerboard_serial(copula)

    def _compute_checkerboard_vectorized(self, copula):
        """
        2D-only: Compute the checkerboard representation using a vectorized approach
        with the copula's cdf_vectorized(u, v).
        """
        if self.d != 2:
            warnings.warn("Vectorized computation only supported for 2D case.")
            return self._compute_checkerboard_serial(copula)

        # Define grid edges for both dimensions
        x_lower = self.grid_points[0][:-1]
        x_upper = self.grid_points[0][1:]
        y_lower = self.grid_points[1][:-1]
        y_upper = self.grid_points[1][1:]

        # Create meshgrids for all corner combinations
        X_lower, Y_lower = np.meshgrid(x_lower, y_lower, indexing="ij")
        X_upper, Y_upper = np.meshgrid(x_upper, y_upper, indexing="ij")

        # Inclusion-exclusion using vectorized cdf
        cdf_uu = copula.cdf_vectorized(X_upper, Y_upper)
        cdf_ll = copula.cdf_vectorized(X_lower, Y_lower)
        cdf_ul = copula.cdf_vectorized(X_upper, Y_lower)
        cdf_lu = copula.cdf_vectorized(X_lower, Y_upper)

        cmatr = cdf_uu - cdf_ul - cdf_lu + cdf_ll
        cmatr = np.clip(cmatr, 0, 1)
        return self._get_checkerboard_copula_for(cmatr)

    def _compute_checkerboard_parallel(self, copula, n_jobs):
        """
        Compute the checkerboard representation in parallel by subdividing the
        grid into cells and evaluating each cell’s measure with inclusion-exclusion.
        """
        indices = list(np.ndindex(*self.n))

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_cell)(idx, copula) for idx in indices
        )
        cmatr = np.zeros(self.n)
        for idx, value in zip(indices, results):
            cmatr[idx] = value

        return self._get_checkerboard_copula_for(cmatr)

    def _compute_checkerboard_serial(self, copula):
        """
        Compute checkerboard representation serially, with caching of CDF calls
        to avoid redundant evaluations.
        """
        cdf_cache = {}
        cmatr = np.zeros(self.n)
        indices = np.ndindex(*self.n)

        def get_cached_cdf(point):
            pt_tuple = tuple(point)
            if pt_tuple not in cdf_cache:
                val = copula.cdf(*point)
                # Convert to float if symbolic or array-like
                if not isinstance(val, (float, int)):
                    val = float(val)
                cdf_cache[pt_tuple] = val
            return cdf_cache[pt_tuple]

        for idx in indices:
            u_lower = [self.grid_points[k][i] for k, i in enumerate(idx)]
            u_upper = [self.grid_points[k][i + 1] for k, i in enumerate(idx)]

            # Inclusion-exclusion sum
            ie_sum = 0.0
            # 2^d corners
            for corner in range(1 << self.d):
                corner_point = []
                for dim_idx in range(self.d):
                    if corner & (1 << dim_idx):
                        corner_point.append(u_upper[dim_idx])
                    else:
                        corner_point.append(u_lower[dim_idx])
                # sign factor => (-1)^(# of upper corners), carefully offset
                sign = (-1) ** (bin(corner).count("1") + self.d)
                ie_sum += sign * get_cached_cdf(corner_point)

            cmatr[idx] = ie_sum

        return self._get_checkerboard_copula_for(cmatr)

    def _process_cell(self, idx, copula):
        """
        Process a single cell in the grid (for parallel).
        """
        u_lower = [self.grid_points[k][i] for k, i in enumerate(idx)]
        u_upper = [self.grid_points[k][i + 1] for k, i in enumerate(idx)]

        inclusion_exclusion_sum = 0.0
        for corner in range(1 << self.d):
            corner_point = [
                (u_upper[dim] if corner & (1 << dim) else u_lower[dim])
                for dim in range(self.d)
            ]
            sign = (-1) ** (bin(corner).count("1") + self.d)
            try:
                cdf_val = copula.cdf(*corner_point)
                cdf_val = float(cdf_val)
                inclusion_exclusion_sum += sign * cdf_val
            except Exception as e:
                log.warning(f"Error computing CDF at {corner_point}: {e}")
        return inclusion_exclusion_sum

    def _get_checkerboard_copula_for(self, cmatr):
        """
        Based on self._checkerboard_type, return the corresponding checkerboard-like copula
        or a BernsteinCopula.
        """
        # CheckPi, CheckMin, BivCheckW, or Bernstein
        if self._checkerboard_type in ["CheckPi", "BivCheckPi"]:
            return CheckPi(cmatr)
        elif self._checkerboard_type in ["CheckMin", "BivCheckMin"]:
            return CheckMin(cmatr)
        elif self._checkerboard_type in ["BivCheckW", "CheckW"]:
            return BivCheckW(cmatr)
        elif self._checkerboard_type in ["Bernstein", "BernsteinCopula"]:
            # Interpret 'cmatr' as the coefficient tensor for a Bernstein copula
            return BernsteinCopula(cmatr, check_theta=False)
        else:
            raise ValueError(f"Unknown checkerboard type: {self._checkerboard_type}")

    def from_data(self, data: Union[pd.DataFrame, np.ndarray, list]):
        """
        Create a checkerboard copula from empirical data (rank-based).

        Parameters
        ----------
        data : pd.DataFrame, np.ndarray, or list
            The empirical data used to estimate the copula.

        Returns
        -------
        CheckPi, BivCheckPi, or other
            A checkerboard copula representation derived from the data.
            (If self._checkerboard_type == "Bernstein", returns a BernsteinCopula.)
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
            # For d>2, a more general approach (e.g. d-dim histogram) would be needed.
            # For now, we just create an empty checkerboard or do a histogram-based approach.
            check_pi_matr = np.zeros(self.n)
            return self._get_checkerboard_copula_for(check_pi_matr)

    def _from_data_bivariate(self, data, n_obs):
        """
        Construct a bivariate checkerboard copula from rank-transformed data.

        Uses numpy’s histogram2d for binning, normalizes, and returns the
        appropriate checkerboard object (or Bernstein).
        """
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values

        hist, _, _ = np.histogram2d(x, y, bins=[self.n[0], self.n[1]], range=[[0, 1], [0, 1]])
        cmatr = hist / n_obs
        return self._get_checkerboard_copula_for(cmatr)


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
        # i-th smallest element gets rank (i+1)/n
        ranks[idx[i]] = (i + 1) / n
    return ranks


def from_data(data, checkerboard_size=None, checkerboard_type="CheckPi"):
    """
    Create a checkerboard-based copula from empirical data using an adaptive grid size.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or list
        Empirical data (samples) from which to build the copula.
    checkerboard_size : int, optional
        Number of grid partitions per dimension. If None, an adaptive size is computed.
    checkerboard_type : str, optional
        Which checkerboard-like object to construct ("CheckPi", "Bernstein", etc.).

    Returns
    -------
    Union[CheckPi, CheckMin, BivCheckW, BernsteinCopula]
        The resulting checkerboard copula approximation.
    """
    if checkerboard_size is None:
        n_samples = len(data)
        # Heuristic: at least 10, at most 50, scaled by sqrt of sample size
        checkerboard_size = min(max(10, int(np.sqrt(n_samples) / 5)), 50)

    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data)

    dimensions = data.shape[1]
    cb = Checkerboarder(n=checkerboard_size, dim=dimensions, checkerboard_type=checkerboard_type)
    return cb.from_data(data)
