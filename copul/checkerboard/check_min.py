import importlib  # Import at module level
import logging

import numpy as np

from copul.checkerboard.check import Check
from copul.exceptions import PropertyUnavailableException

log = logging.getLogger(__name__)


class CheckMin(Check):
    """
    Checkerboard "Min" Copula

    This copula implements a "min-fraction" approach for computing the cumulative
    distribution function (CDF) across all dimensions, and a fully discrete method for
    calculating conditional distributions.

    Key features:
      - CDF Calculation:
          Uses a min-fraction partial coverage over all dimensions to aggregate the CDF.

      - Conditional Distribution (cond_distr):
          For any given dimension i and input vector u:
            1. In dimension i, determine the cell index as floor(u[i] * dim[i]). The entire
               slice corresponding to this index constitutes the conditioning event (denominator).
            2. In every other dimension j ≠ i, only include cells where the cell index c[j] is
               strictly less than floor(u[j] * dim[j]); this avoids any partial coverage in these dimensions.
            3. Compute the conditional distribution as the ratio of the count of cells meeting the
               numerator condition (cells matching the target criteria) to the total count of cells
               in the conditioning event (denominator).

    Example:
      For a 2x2x2 grid and u = (0.5, 0.5, 0.5):
        - In dimension 0, we use floor(0.5 * 2) = 1, selecting the second layer.
        - Within that layer, for dimensions 1 and 2, only cells with indices less than floor(0.5 * 2) = 1
          are considered (i.e., only cells with index 0).
        - This results in 1 favorable cell out of 4 in the conditioning event, so:
              cond_distr(1, (0.5, 0.5, 0.5)) = 1 / 4 = 0.25.
    """

    def __new__(cls, matr, *args, **kwargs):
        """
        Create a new CheckMin instance or a BivCheckMin instance if dimension is 2.

        Parameters
        ----------
        matr : array-like
            Matrix of values that determine the copula's distribution.
        *args, **kwargs
            Additional arguments passed to the constructor.

        Returns
        -------
        CheckMin or BivCheckMin
            A CheckMin instance, or a BivCheckMin instance if dimension is 2.
        """
        # If this is the CheckMin class itself (not a subclass)
        if cls is CheckMin:
            # Convert matrix to numpy array to get its dimensionality
            matr_arr = np.asarray(matr)

            # Check if it's a 2D matrix (bivariate copula)
            if matr_arr.ndim == 2:
                # Import the BivCheckMin class here to avoid circular imports
                try:
                    bcp_module = importlib.import_module("copul.checkerboard.biv_check_min")
                    BivCheckMin = bcp_module.BivCheckMin
                    # Return a new BivCheckMin instance with the same arguments
                    return BivCheckMin(matr, *args, **kwargs)
                except (ImportError, ModuleNotFoundError, AttributeError):
                    # If the import fails, just continue with normal instantiation
                    pass

        # Create a normal instance by directly calling the parent's __new__
        # Using the actual class for clarity and to avoid MRO issues
        from copul.checkerboard.check import Check

        instance = Check.__new__(cls)
        return instance

    def __str__(self):
        return f"CheckMinCopula({self.matr.shape})"

    @property
    def is_absolutely_continuous(self) -> bool:
        # 'Min' copula is degenerate along lines, so not absolutely continuous in R^d
        return False

    # --------------------------------------------------------------------------
    # 1) CDF with 'min fraction' partial coverage
    # ------------------------------------------>--------------------------------
    def cdf(self, *args):
        """
        Compute the CDF at one or multiple points.

        This method handles both single-point and multi-point CDF evaluation
        in an efficient vectorized manner, using the 'min-fraction' approach
        specific to CheckMin.

        Parameters
        ----------
        *args : array-like or float
            Either:
            - Multiple separate coordinates (x, y, ...) of a single point
            - A single array-like object with coordinates of a single point
            - A 2D array where each row represents a separate point

        Returns
        -------
        float or numpy.ndarray
            If a single point is provided, returns a float.
            If multiple points are provided, returns an array of shape (n_points,).

        Examples
        --------
        # Single point as separate arguments
        value = copula.cdf(0.3, 0.7)

        # Single point as array
        value = copula.cdf([0.3, 0.7])

        # Multiple points as 2D array
        values = copula.cdf(np.array([[0.1, 0.2], [0.3, 0.4]]))
        """
        # Handle different input formats
        if len(args) == 0:
            raise ValueError("No arguments provided")

        elif len(args) == 1:
            # A single argument was provided - either a point or multiple points
            arg = args[0]

            if hasattr(arg, "ndim") and hasattr(arg, "shape"):
                # NumPy array or similar
                arr = np.asarray(arg, dtype=float)

                if arr.ndim == 1:
                    # 1D array - single point
                    if len(arr) != self.dim:
                        raise ValueError(
                            f"Expected point array of length {self.dim}, got {len(arr)}"
                        )
                    return self._cdf_single_point(arr)

                elif arr.ndim == 2:
                    # 2D array - multiple points
                    if arr.shape[1] != self.dim:
                        raise ValueError(
                            f"Expected points with {self.dim} dimensions, got {arr.shape[1]}"
                        )
                    return self._cdf_vectorized_impl(arr)

                else:
                    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array")

            elif hasattr(arg, "__len__"):
                # List, tuple, or similar sequence
                if len(arg) == self.dim:
                    # Single point as a sequence
                    return self._cdf_single_point(np.array(arg, dtype=float))
                else:
                    raise ValueError(f"Expected point with {self.dim} dimensions, got {len(arg)}")

            else:
                # Single scalar value - only valid for 1D case
                if self.dim == 1:
                    return self._cdf_single_point(np.array([arg], dtype=float))
                else:
                    raise ValueError(f"Single scalar provided but copula has {self.dim} dimensions")

        else:
            # Multiple arguments provided
            if len(args) == self.dim:
                # Separate coordinates for a single point
                return self._cdf_single_point(np.array(args, dtype=float))
            else:
                raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}")

    def _cdf_single_point(self, u):
        """CDF at a single point using the min-fraction approach."""
        return float(self._cdf_vectorized_impl(np.asarray(u, dtype=float)[None, :])[0])

    def _cdf_vectorized_impl(self, points, chunk_elems=4_000_000):
        """
        Vectorised min-fraction CDF for an ``(n_points, dim)`` array:
        ``C(u) = sum_c matr[c] * min_d F_d(u_d, c_d)``.

        Only cells with positive mass are used and points are processed in
        chunks, so memory stays bounded.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        n_points = points.shape[0]
        shape = np.array(self.matr.shape)
        cells = np.argwhere(self.matr > 0)  # (K, dim)
        masses = self.matr[tuple(cells.T)]
        out = np.zeros(n_points)
        if cells.size == 0:
            return out
        step = max(1, chunk_elems // (cells.shape[0] * self.dim))
        for s in range(0, n_points, step):
            pts = np.clip(points[s : s + step], 0.0, 1.0) * shape  # (n, dim)
            frac = np.clip(pts[None, :, :] - cells[:, None, :], 0.0, 1.0)
            out[s : s + step] = masses @ frac.min(axis=2)
        return out

    def cond_distr(self, i, *args):
        """
        Compute the conditional distribution for one or multiple points.

        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        *args : array-like or float
            Either:
            - Multiple separate coordinates (x, y, ...) of a single point
            - A single array-like object with coordinates of a single point
            - A 2D array where each row represents a separate point

        Returns
        -------
        float or numpy.ndarray
            If a single point is provided, returns a float.
            If multiple points are provided, returns an array of shape (n_points,).

        Examples
        --------
        # Single point as separate arguments
        value = copula.cond_distr(1, 0.3, 0.7)

        # Single point as array
        value = copula.cond_distr(1, [0.3, 0.7])

        # Multiple points as 2D array
        values = copula.cond_distr(1, np.array([[0.1, 0.2], [0.3, 0.4]]))
        """
        if i < 1 or i > self.dim:
            raise ValueError(f"Dimension {i} out of range 1..{self.dim}")

        # Handle different input formats
        if len(args) == 0:
            raise ValueError("No point coordinates provided")

        elif len(args) == 1:
            # A single argument was provided - either a point or multiple points
            arg = args[0]

            if hasattr(arg, "ndim") and hasattr(arg, "shape"):
                # NumPy array or similar
                arr = np.asarray(arg, dtype=float)

                if arr.ndim == 1:
                    # 1D array - single point
                    if len(arr) != self.dim:
                        raise ValueError(
                            f"Expected point array of length {self.dim}, got {len(arr)}"
                        )
                    return self._cond_distr_single(i, arr)

                elif arr.ndim == 2:
                    # 2D array - multiple points
                    if arr.shape[1] != self.dim:
                        raise ValueError(
                            f"Expected points with {self.dim} dimensions, got {arr.shape[1]}"
                        )
                    return self._cond_distr_vectorized(i, arr)

                else:
                    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array")

            elif hasattr(arg, "__len__"):
                # List, tuple, or similar sequence
                if len(arg) == self.dim:
                    # Single point as a sequence
                    return self._cond_distr_single(i, np.array(arg, dtype=float))
                else:
                    raise ValueError(f"Expected point with {self.dim} dimensions, got {len(arg)}")

            else:
                # Single scalar value - only valid for 1D case
                if self.dim == 1:
                    return self._cond_distr_single(i, np.array([arg], dtype=float))
                else:
                    raise ValueError(f"Single scalar provided but copula has {self.dim} dimensions")

        else:
            # Multiple arguments provided
            if len(args) == self.dim:
                # Separate coordinates for a single point
                return self._cond_distr_single(i, np.array(args, dtype=float))
            else:
                raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}")

    def _cond_distr_single(self, i, u):
        """Conditional distribution at a single point."""
        pts = np.asarray(u, dtype=float)[None, :]
        return float(self._cond_distr_vectorized(i, pts)[0])

    def _cond_distr_vectorized(self, i, points):
        """
        Vectorised conditional distribution for an ``(n_points, dim)`` array.

        In the slice ``c[i0] = floor(u_i0 * m_i0)`` a cell counts if, in every
        other dimension ``j``, ``u_j`` lies above the cell's lower edge and the
        cell-local fraction ``frac_j`` is at least the conditioning fraction
        ``frac_i`` (the cell mass sits on the cell diagonal).  The criterion
        factorises over the dimensions, so a separable contraction is used.
        """
        points = np.asarray(points, dtype=float)
        n_pts = points.shape[0]
        shape = self.matr.shape
        i0 = i - 1
        k = shape[i0]
        x = points[:, i0]
        xs = np.clip(x, 0.0, 1.0) * k
        idx = np.minimum(np.floor(xs).astype(np.intp), k - 1)
        frac_i = np.clip(xs - idx, 0.0, 1.0)
        onehot = np.zeros((n_pts, k))
        onehot[np.arange(n_pts), idx] = 1.0
        factors = []
        for d in range(self.dim):
            if d == i0:
                factors.append(onehot)
                continue
            kd = shape[d]
            y = np.asarray(points[:, d], dtype=float) * kd
            frac = y[:, None] - np.arange(kd)[None, :]
            q = (frac > 0) & ((frac >= 1) | (frac >= frac_i[:, None] - 1e-10))
            factors.append(q.astype(float))
        num = self._contract(factors)
        axes = tuple(d for d in range(self.dim) if d != i0)
        denom = self.matr.sum(axis=axes)[idx] if axes else self.matr[idx]
        out = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        if not axes:
            out = np.where(denom > 0, 1.0, 0.0)
        return np.where(x < 0, 0.0, out)

    @property
    def pdf(self):
        raise PropertyUnavailableException("PDF does not exist for CheckMin.")

    def rvs(self, n=1, random_state=None, **kwargs):
        """
        Draw ``n`` samples: pick a cell by mass, then a point on its diagonal.

        ``random_state`` may be an int, a numpy Generator or ``None`` (NumPy's
        global generator, never reseeded).
        """
        from copul.checkerboard._biv_engine import resolve_rng

        rng = resolve_rng(random_state)
        log.debug(f"Generating {n} random variates for {self}...")
        flat = np.asarray(self.matr, dtype=float).ravel()
        flat_idx = rng.choice(flat.size, size=int(n), p=flat / flat.sum())
        cells = np.column_stack(np.unravel_index(flat_idx, self.matr.shape))
        t = rng.random(int(n))
        return (cells + t[:, None]) / np.array(self.matr.shape)

    @staticmethod
    def _weighted_random_selection(matrix, num_samples, random_state=None):
        from copul.checkerboard.check_pi import CheckPi

        return CheckPi._weighted_random_selection(matrix, num_samples, random_state)


if __name__ == "__main__":
    ccop = CheckMin([[1, 2], [2, 1]])
    ccop.cdf((0.2, 0.2))
