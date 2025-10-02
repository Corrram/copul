import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --- keep your scipy import flag as-is ---
try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _row_targets_from_conditionals(
    cmatr: np.ndarray, method: str = "median"
) -> np.ndarray:
    """
    Compute target column indices y_i (1..n) for each row i using row conditionals.
    method: 'median' (robust) or 'mean' (smoother).
    """
    n = cmatr.shape[1]
    y = np.empty(cmatr.shape[0], dtype=float)

    if method == "mean":
        cols = np.arange(1, n + 1, dtype=float)
        row_sums = cmatr.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div0 for all-zero rows
        y = (cmatr @ cols) / row_sums[:, 0]
        return y

    # median (default)
    cdf = np.cumsum(cmatr, axis=1)
    # target quantile 0.5 per row
    for i in range(cmatr.shape[0]):
        if cdf[i, -1] <= 0:
            y[i] = (n + 1) / 2.0
        else:
            j = np.searchsorted(cdf[i, :], 0.5 * cdf[i, -1], side="left")
            y[i] = float(j + 1)  # 1-based
    return y


def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """
    Pool-Adjacent-Violators (PAV) isotonic regression.
    Enforces a nondecreasing fit and returns an array of the same length as y.
    """
    y = np.asarray(y, dtype=float)
    n = y.size

    # Maintain blocks as (mean, size), merging when monotonicity is violated
    means: list[float] = []
    sizes: list[int] = []

    for val in y:
        means.append(val)
        sizes.append(1)
        # Merge backwards while previous mean > last mean
        while len(means) >= 2 and means[-2] > means[-1]:
            m2, s2 = means.pop(), sizes.pop()
            m1, s1 = means.pop(), sizes.pop()
            m = (m1 * s1 + m2 * s2) / (s1 + s2)
            means.append(m)
            sizes.append(s1 + s2)

    # Reconstruct the fitted vector
    fit = np.empty(n, dtype=float)
    pos = 0
    for m, s in zip(means, sizes):
        fit[pos : pos + s] = m
        pos += s
    return fit


def _assignment_closest_to_targets(y_iso: np.ndarray, n: int) -> np.ndarray:
    """
    Pick permutation pi (1..n) closest to targets y_iso (floats in [1,n]) in least squares sense.
    Uses Hungarian if available; otherwise, greedy match sorted pairs.
    Returns 1-based permutation.
    """
    if _HAS_SCIPY:
        # Build cost matrix (i,j): (j - y_i)^2
        j_idx = np.arange(1, n + 1, dtype=float)
        # (i,j) cost computed via broadcasting
        C = (j_idx[None, :] - y_iso[:, None]) ** 2
        row_ind, col_ind = linear_sum_assignment(C)
        pi0 = col_ind  # 0-based
        return pi0 + 1

    # Greedy fallback: sort rows by target; assign nearest free column
    order = np.argsort(y_iso)
    available = list(range(1, n + 1))
    pi = np.empty(n, dtype=int)
    from bisect import bisect_left

    for i in order:
        t = y_iso[i]
        # find nearest available column to t
        pos = bisect_left(available, t)
        candidates = []
        if pos < len(available):
            candidates.append(available[pos])
        if pos > 0:
            candidates.append(available[pos - 1])
        # pick the closer one
        best = min(candidates, key=lambda c: abs(c - t))
        pi[i] = best
        available.remove(best)
    return pi


def _blend_cost_matrix(
    cmatr: np.ndarray, y_iso: np.ndarray, lam: float = 0.15
) -> np.ndarray:
    """
    Optional: blended cost that still considers large cell mass.
    Minimization cost = lam * (-normalized_mass) + (1-lam) * normalized_sq_distance_to_target
    """
    n = cmatr.shape[0]
    # normalize mass to [0,1]
    M = cmatr.copy()
    if M.size > 0:
        mmax = M.max()
        if mmax > 0:
            M = M / mmax
    j_idx = np.arange(1, n + 1, dtype=float)
    D = (j_idx[None, :] - y_iso[:, None]) ** 2
    D = D / (n**2)  # scale-invariant
    # we minimize: lam*(-M) + (1-lam)*D
    return lam * (-M) + (1.0 - lam) * D


def _best_permutation_from_mass(cmatr: np.ndarray, lookback: int = 0) -> np.ndarray:
    """
    Greedy L∞-CDF fitter for Shuffle-of-Min.
    Builds permutation π to minimize the uniform (sup) distance between
    the cumulative S(i,j) of `cmatr` and the cumulative H(i,j) of the
    permutation matrix (scaled by 1/n).

    Params
    ------
    cmatr : (n,n) nonnegative matrix summing to 1 (or any positive total).
    lookback : how many previous rows to include (in addition to row i)
               when evaluating candidate j. 0 = just current row;
               1..3 gives a bit more stability (costlier).

    Returns
    -------
    1-based permutation (np.ndarray of ints).
    """
    C = np.asarray(cmatr, dtype=float)
    n, m = C.shape
    if n != m:
        raise ValueError("cmatr must be square.")

    # Normalize to probability mass; S is prefix-sum surface.
    total = C.sum()
    if total <= 0:
        # degenerate: return identity
        return np.arange(1, n + 1, dtype=int)
    P = C / total
    S = P.cumsum(axis=0).cumsum(axis=1)  # S[i,j] is cumulative up to (i,j) (0-based)

    # H counters: for each j, how many assigned rows ≤ current i have π(row) ≤ j
    # We only need the last row of H each iteration; H_row[j] = (#)/n
    assigned_leq = np.zeros(n, dtype=int)  # cumulative counts over columns (≤ j)
    used = np.zeros(n, dtype=bool)
    pi = np.empty(n, dtype=int)

    for i in range(n):  # 0-based row index
        # Candidate evaluation
        best = None
        # Precompute the cumulative S rows we’ll compare against
        i0 = max(0, i - lookback)
        rows = range(i0, i + 1)

        # For speed, get the current H_row before adding row i
        # H_row[j] corresponds to H(i, j) after previous assignments
        H_prev = assigned_leq / n  # shape (n,)

        for j in range(n):  # candidate column (0-based)
            if used[j]:
                continue
            # New H after assigning π(i)=j:
            # increments by 1/n for all columns ≥ j
            # So H_new[k] = H_prev[k] + 1/n for k >= j; else unchanged
            # We need sup over columns for each considered row r
            # For rows < i, H doesn't change; for row i, H becomes H_new
            # But we can optionally include a small lookback window:
            # For rows r in rows, the "row cumulative" is H_prev for r<i and H_new for i.
            # We compute sup over j' of |S[r, j'] - H_row_r[j']|.
            # Efficiently: we only compare the row vectors.

            # Build H_row for the current row i:
            H_row_new = H_prev.copy()
            H_row_new[j:] += 1.0 / n

            # Evaluate sup error over columns for the current row (and lookback rows).
            # Note: S is cumulative over BOTH axes; row-slice S[r, :] is what we want.
            sup_err = 0.0
            # previous rows use H_prev; current row uses H_row_new
            for r in rows:
                H_row_r = H_prev if r < i else H_row_new
                # L∞ over columns at row r:
                # Use vectorized max abs diff
                e = np.max(np.abs(S[r, :] - H_row_r))
                if e > sup_err:
                    sup_err = e

            # Tie-breakers: minimize sup_err, then prefer bigger reduction from previous,
            # then prefer larger mass in the actual cell (i,j).
            # Compute previous sup for comparison
            base_sup = 0.0
            for r in rows:
                e0 = np.max(np.abs(S[r, :] - H_prev))
                if e0 > base_sup:
                    base_sup = e0
            reduction = base_sup - sup_err
            score = (sup_err, -reduction, -P[i, j])  # lexicographic

            if best is None or score < best[0]:
                best = (score, j, H_row_new)

        # Commit best choice
        _, j_star, H_row_new = best
        pi[i] = j_star + 1  # 1-based
        used[j_star] = True
        # Update assigned_leq (cumulative counts over columns)
        assigned_leq[j_star:] += 1

    return pi


class Checkerboarder:
    def __init__(self, n: Union[int, list] = None, dim=2, checkerboard_type="CheckPi"):  # noqa: E501
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
        """
        log.debug("Computing checkerboard copula with grid sizes: %s", self.n)

        # If 2D and copula has a 'cdf_vectorized' method, do vectorized approach
        if hasattr(copula, "cdf_vectorized") and self.d == 2:
            return self._compute_checkerboard_vectorized(copula)

        # Otherwise, decide on serial vs parallel
        if n_jobs is None:
            total_cells = np.prod(self.n)
            n_jobs = max(1, min(8, total_cells // 1000))

        return self._compute_checkerboard_serial(copula)

    def _compute_checkerboard_vectorized(self, copula, tol=1e-12):
        """
        Computes the checkerboard copula mass matrix in a highly optimized, vectorized manner.

        This method calculates the probability mass for each cell in the grid by first
        evaluating the copula's CDF at all grid intersection points in a single call.
        It then uses 2D finite differences on the resulting CDF grid to find the mass
        of each rectangular cell. This avoids redundant computations and is significantly
        faster than calculating the CDF for each cell's corners separately.
        """
        if self.d != 2:
            warnings.warn("Vectorized computation is only supported for the 2D case.")
            return self._compute_checkerboard_serial(copula)

        # 1. Get the unique grid points for each dimension.
        u_pts = self.grid_points[0]
        v_pts = self.grid_points[1]

        # 2. Create a meshgrid of all grid points.
        # This prepares the input for a single, comprehensive CDF evaluation.
        U, V = np.meshgrid(u_pts, v_pts, indexing="ij")

        # 3. Call the vectorized CDF function ONCE on the entire grid of points.
        # This is the core optimization, replacing four separate, expensive calls.
        cdf_grid = copula.cdf_vectorized(U, V)

        # 4. Compute the probability mass of each cell using fast 2D finite differences.
        # This is equivalent to the inclusion-exclusion principle for rectangles:
        # P(u_i<U<u_{i+1}, v_j<V<v_{j+1}) = C(u_{i+1},v_{j+1}) - C(u_{i+1},v_j) - C(u_i,v_{j+1}) + C(u_i,v_j)
        cmatr = (
            cdf_grid[1:, 1:]  # Upper-right corners of all cells
            - cdf_grid[1:, :-1]  # Upper-left corners
            - cdf_grid[:-1, 1:]  # Lower-right corners
            + cdf_grid[:-1, :-1]  # Lower-left corners
        )

        # 5. Handle potential floating-point inaccuracies.
        # The logic here remains the same as your original implementation.
        neg_mask = cmatr < 0
        if np.any(neg_mask):
            min_val = cmatr[neg_mask].min()
            if min_val < -tol:
                log.warning(
                    f"cmatr has {np.sum(neg_mask)} entries < -{tol:.1e}; "
                    f"most extreme = {min_val:.3e}"
                )
            cmatr[neg_mask] = 0.0  # Zero out any negative probabilities

        # Ensure the probabilities are clipped between 0 and 1.
        cmatr = np.clip(cmatr, 0, 1)

        return self._get_checkerboard_copula_for(cmatr)

    def _compute_checkerboard_serial(self, copula):
        cdf_cache = {}
        cmatr = np.zeros(self.n)
        indices = np.ndindex(*self.n)

        def get_cached_cdf(point):
            pt_tuple = tuple(point)
            if pt_tuple not in cdf_cache:
                val = copula.cdf(*point)
                if not isinstance(val, (float, int)):
                    val = float(val)
                cdf_cache[pt_tuple] = val
            return cdf_cache[pt_tuple]

        for idx in indices:
            u_lower = [self.grid_points[k][i] for k, i in enumerate(idx)]
            u_upper = [self.grid_points[k][i + 1] for k, i in enumerate(idx)]
            ie_sum = 0.0
            for corner in range(1 << self.d):
                corner_point = [
                    (u_upper[dim_idx] if corner & (1 << dim_idx) else u_lower[dim_idx])
                    for dim_idx in range(self.d)
                ]
                sign = (-1) ** (bin(corner).count("1") + self.d)
                ie_sum += sign * get_cached_cdf(corner_point)
            cmatr[idx] = ie_sum
        cmatr = np.clip(cmatr, 0, 1)
        return self._get_checkerboard_copula_for(cmatr)

    def _process_cell(self, idx, copula):
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
        Lazily import and return the appropriate checkerboard-like copula.
        """
        if self._checkerboard_type in ["CheckPi", "BivCheckPi"]:
            from copul.checkerboard.check_pi import CheckPi

            return CheckPi(cmatr)
        elif self._checkerboard_type in ["CheckMin", "BivCheckMin"]:
            from copul.checkerboard.check_min import CheckMin

            return CheckMin(cmatr)
        elif self._checkerboard_type in ["BivCheckW", "CheckW"]:
            from copul.checkerboard.biv_check_w import BivCheckW

            return BivCheckW(cmatr)
        elif self._checkerboard_type in ["Bernstein", "BernsteinCopula"]:
            from copul.checkerboard.bernstein import BernsteinCopula

            return BernsteinCopula(cmatr, check_theta=False)
        elif self._checkerboard_type in ["ShuffleOfMin"]:
            # Build a ShuffleOfMin of order n by solving the assignment on the mass matrix.
            from copul.checkerboard.shuffle_min import (
                ShuffleOfMin,
            )  # adjust import path if needed

            cmatr = np.asarray(cmatr, dtype=float)
            if cmatr.ndim != 2 or cmatr.shape[0] != cmatr.shape[1]:
                raise ValueError(
                    "ShuffleOfMin requires a square checkerboard mass matrix (n x n). "
                    f"Got shape {cmatr.shape}."
                )
            pi = _best_permutation_from_mass(cmatr)  # 1-based permutation
            return ShuffleOfMin(pi.tolist())
        else:
            raise ValueError(f"Unknown checkerboard type: {self._checkerboard_type}")

    def from_data(self, data: Union[pd.DataFrame, np.ndarray, list]):  # noqa: E501
        # Normalize input to DataFrame
        if isinstance(data, (list, np.ndarray)):
            data = pd.DataFrame(data)

        n_obs = len(data)

        # Rank -> pseudo-observations in (0,1]; one column at a time for speed (numba)
        rank_data = np.empty_like(data.values, dtype=float)
        for i, col in enumerate(data.columns):
            rank_data[:, i] = _fast_rank(data[col].values)

        # Use existing fast path for 2D
        if self.d == 2:
            rank_df = pd.DataFrame(rank_data, columns=data.columns)
            return self._from_data_bivariate(rank_df, n_obs)

        # General d-dimensional case (d > 2): histogram on the unit cube
        # Guard against any tiny floating overshoots by nudging 1.0 to the next lower float.
        right_inclusive = np.nextafter(1.0, 0.0)
        rank_data = np.clip(rank_data, 0.0, right_inclusive)

        # Build d-D histogram with per-dimension bins self.n and range [0,1] in each dim
        hist, _ = np.histogramdd(rank_data, bins=self.n, range=[(0.0, 1.0)] * self.d)

        # Convert counts to probabilities
        cmatr = hist / n_obs

        return self._get_checkerboard_copula_for(cmatr)

    def _from_data_bivariate(self, data, n_obs):
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        hist, _, _ = np.histogram2d(
            x, y, bins=[self.n[0], self.n[1]], range=[[0, 1], [0, 1]]
        )
        cmatr = hist / n_obs
        return self._get_checkerboard_copula_for(cmatr)

    def approximate_shuffle_of_min(self, copula=None, cmatr=None, order=None):
        """
        Fit a Shuffle-of-Min copula to a checkerboard mass matrix by solving
        an assignment problem that maximizes the captured mass.

        Parameters
        ----------
        copula : optional
            If provided, we first compute the checkerboard mass for this copula.
            Ignored if `cmatr` is provided.
        cmatr : np.ndarray, optional
            Precomputed checkerboard mass matrix. If provided, we use it directly.
        order : int, optional
            Desired SoM order. If None, uses the grid size if square; if not,
            tries to down/up-sample to a square matrix of size `order`.

        Returns
        -------
        ShuffleOfMin
        """
        from copul.checkerboard.shuffle_min import ShuffleOfMin

        if cmatr is None:
            if copula is None:
                raise ValueError("Provide either `copula` or `cmatr`.")
            # compute checkerboard mass with your existing path
            cb = self.get_checkerboard_copula(copula)
            # All your checkerboard-like classes should expose their mass matrix;
            # if not, add a property/accessor where they were constructed from `cmatr`.
            try:
                cmatr = cb.cmatr
            except AttributeError:
                raise AttributeError("Checkerboard copula does not expose `cmatr`.")

        cmatr = np.asarray(cmatr, dtype=float)
        if cmatr.ndim != 2:
            raise ValueError("cmatr must be 2D.")
        M, N = cmatr.shape

        # If order unspecified, require square matrix
        if order is None:
            if M != N:
                raise ValueError("cmatr must be square or specify `order`.")
            order = M

        # If sizes mismatch, aggregate or interpolate to square [order x order].
        if (M != order) or (N != order):
            # Simple block aggregation for downsampling;
            # for upsampling, average-nn interpolation to conserve total mass.
            cmatr = _resize_mass_matrix(cmatr, order, order)

        # Solve assignment and build SoM
        pi = _best_permutation_from_mass(cmatr)
        return ShuffleOfMin(pi.tolist())


# Utility to resize mass matrices while preserving total mass
def _resize_mass_matrix(C: np.ndarray, new_m: int, new_n: int) -> np.ndarray:
    """
    Resize a nonnegative matrix C to (new_m, new_n) approximately conserving total mass.
    Downsampling uses block sums; upsampling uses proportional split.
    """
    m, n = C.shape
    total = C.sum()
    if m == new_m and n == new_n:
        return C

    # Map each old cell to continuous [0,1]x[0,1] then partition into new grid bins.
    # Implemented via 1D resampling along axes with cumulative sums.
    def _resample_axis(A, old, new):
        # cumulative mass along axis, then linear interpolation at new bin edges
        np.cumsum(A, axis=0) if old == A.shape[0] else np.cumsum(A, axis=1)
        # Normalize to [0,total] along that axis and interpolate cuts,
        # but keeping things concise we’ll do simple averaging fallback:
        return A  # (keep simple; call twice would be overkill for now)

    # For brevity: nearest-block aggregation if both downsizing
    if new_m <= m and new_n <= n:
        fm = m / new_m
        fn = n / new_n
        out = np.zeros((new_m, new_n), dtype=float)
        for i in range(new_m):
            for j in range(new_n):
                i0 = int(np.floor(i * fm))
                i1 = int(np.floor((i + 1) * fm))
                j0 = int(np.floor(j * fn))
                j1 = int(np.floor((j + 1) * fn))
                out[i, j] = C[i0 : i1 or None, j0 : j1 or None].sum()
        # Normalize minor rounding
        s = out.sum()
        if s > 0 and total > 0:
            out *= total / s
        return out
    else:
        # Simple bilinear-like upsample with normalization
        # (cheap & cheerful; good enough for initialization)
        grid_u = np.linspace(0, 1, m, endpoint=False) + 0.5 / m
        grid_v = np.linspace(0, 1, n, endpoint=False) + 0.5 / n
        U, V = np.meshgrid(grid_u, grid_v, indexing="ij")
        uu = np.linspace(0, 1, new_m, endpoint=False) + 0.5 / new_m
        vv = np.linspace(0, 1, new_n, endpoint=False) + 0.5 / new_n
        UU, VV = np.meshgrid(uu, vv, indexing="ij")
        # nearest-neighbor for speed
        II = np.clip((UU * m).astype(int), 0, m - 1)
        J = np.clip((VV * n).astype(int), 0, n - 1)
        out = C[II, J]
        s = out.sum()
        if s > 0 and total > 0:
            out *= total / s
        return out


def _fast_rank(x):
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    idx = np.argsort(x)
    for i in range(n):
        ranks[idx[i]] = (i + 1) / n
    return ranks


def from_data(data, checkerboard_size=None, checkerboard_type="CheckPi"):  # noqa: E501
    if checkerboard_size is None:
        n_samples = len(data)
        checkerboard_size = min(max(10, int(np.sqrt(n_samples) / 5)), 50)
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data)
    dimensions = data.shape[1]
    cb = Checkerboarder(
        n=checkerboard_size, dim=dimensions, checkerboard_type=checkerboard_type
    )
    return cb.from_data(data)


def from_samples(samples, checkerboard_size=None, checkerboard_type="CheckPi"):  # noqa: E501
    return from_data(samples, checkerboard_size, checkerboard_type)


def from_matrix(matrix, checkerboard_size=None, checkerboard_type="CheckPi"):  # noqa: E501
    return from_data(matrix, checkerboard_size, checkerboard_type)
