import itertools
import logging

import numpy as np
import sympy

from copul.exceptions import PropertyUnavailableException

log = logging.getLogger(__name__)


class CheckMin:
    """
    Checkerboard 'Min' copula:

      - The CDF uses 'min-fraction' partial coverage across all dimensions.
      - The cond_distr(i,u) uses a purely discrete approach in *all* dimensions:
          * pick the single integer cell index in dim i = floor(u[i]*dim[i]),
            that entire slice is the conditioning event (denominator).
          * For j != i, we only include cells c where c[j] < floor(u[j]*dim[j])
            (no partial coverage).
          * ratio = numerator/denominator.

    This matches your test examples, e.g. in 2x2x2 with (0.5,0.5,0.5),
    cond_distr(1,(0.5,0.5,0.5))=1/4=0.25, because we pick c[0]=1
    (the 'second layer'), and among c[0]=1, only one of those four cells
    has c[1]=0,c[2]=0 => 1 out of 4 => 0.25.
    """

    params = []
    intervals = {}

    def __init__(self, matr):
        if isinstance(matr, list):
            matr = np.array(matr)
        matr_sum = sum(matr) if isinstance(matr, sympy.Matrix) else matr.sum()
        self.matr = matr / matr_sum

        self.dim = self.matr.shape
        self.d = len(self.dim)

    def __str__(self):
        return f"CheckerboardMinCopula({self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        # 'Min' copula is degenerate along lines, so not absolutely continuous in R^d
        return False

    # --------------------------------------------------------------------------
    # 1) CDF with 'min fraction' partial coverage
    # --------------------------------------------------------------------------
    def cdf(self, *args):
        """
        Evaluate the CDF at (u1,...,ud).  For each cell c:
          fraction = min_k( overlap_len(k) / cell_width(k) ),
        then sum cell_mass * fraction.
        """
        if len(args) != self.d:
            raise ValueError(f"cdf expects {self.d} coordinates, got {len(args)}.")

        # Quick boundaries
        if any(u <= 0 for u in args):
            return 0.0
        if all(u >= 1 for u in args):
            return 1.0

        total = 0.0
        for c in itertools.product(*(range(s) for s in self.dim)):
            cell_mass = self.matr[c]
            if cell_mass <= 0:
                continue

            # min-fraction across dims
            frac_cell = 1.0
            for k in range(self.d):
                lower_k = c[k] / self.dim[k]
                upper_k = (c[k] + 1) / self.dim[k]
                overlap_k = max(0.0, min(args[k], upper_k) - lower_k)
                if overlap_k <= 0:
                    frac_cell = 0.0
                    break
                width_k = 1.0 / self.dim[k]
                frac_k = overlap_k / width_k
                if frac_k > 1.0:
                    frac_k = 1.0
                if frac_k < frac_cell:
                    frac_cell = frac_k

            if frac_cell > 0:
                total += cell_mass * frac_cell

        return float(total)

    def cond_distr(self, i: int, u):
        """
        Compute F_{U_{-i} | U_i}(u_{-i} | u_i) using a 'cell-based' conditioning dimension i.

        Steps for dimension i (1-based):
              1) Find i0 = i - 1 (zero-based).
              2) Identify the cell index along dim i0 where u[i0] lies:
                 i_idx = floor(u[i0] * self.dim[i0])  (clamp if needed).
              3) The denominator = sum of masses of all cells that have index[i0] = i_idx,
                 *without* any partial fraction for that dimension i0.  We treat that entire
                 'slice' as the event {U_i is in that cell}.
              4) The numerator = among that same slice, we see how much of each cell is
                 below u[j] in the other dimensions j != i0, using partial-overlap logic
                 if 0 <= u[j] <= 1.  Sum that over the slice.
              5) cond_distr = numerator / denominator  (or 0 if denominator=0).

        Optimized version for CheckMin copula with improved performance.
        """
        if i < 1 or i > self.d:
            raise ValueError(f"Dimension {i} out of range 1..{self.d}")

        i0 = i - 1
        if len(u) != self.d:
            raise ValueError(f"Point u must have length {self.d}.")

        # Find which cell index along dim i0 the coordinate u[i0] falls into
        x_i = u[i0]
        if x_i < 0:
            return 0.0  # If 'conditioning coordinate' <0, prob is 0
        elif x_i >= 1:
            # If 'conditioning coordinate' >=1, then we pick the last cell index
            i_idx = self.dim[i0] - 1
        else:
            i_idx = int(np.floor(x_i * self.dim[i0]))
            # clamp (just in case)
            if i_idx < 0:
                i_idx = 0
            elif i_idx >= self.dim[i0]:
                i_idx = self.dim[i0] - 1

        # For safety, reset the intervals cache between calls
        self.intervals = {}

        # Cache key for the slice indices
        slice_key = (i, i_idx)

        # Check if we have cached slice indices for this dimension and index
        if slice_key in self.intervals:
            slice_indices = self.intervals[slice_key]

            # Calculate denominator - sum of all cells in the slice
            denom = 0.0
            for c in slice_indices:
                denom += self.matr[c]
        else:
            # Create more efficient slice iteration by only iterating through relevant dimensions
            indices = [range(s) for s in self.dim]
            indices[i0] = [i_idx]  # Fix the i0 dimension

            # Collect all cells in the slice
            denom = 0.0
            slice_indices = []
            for c in itertools.product(*indices):
                cell_mass = self.matr[c]
                denom += cell_mass
                # Store all cells for CheckMin (even zero mass cells might be needed for boundary checks)
                slice_indices.append(c)

            # Store in cache for future use
            self.intervals[slice_key] = slice_indices

        if denom <= 0:
            return 0.0

        # Precompute 1/dim values for faster calculations
        inv_dims = np.array([1.0 / dim for dim in self.dim])
        u_array = np.array(u)

        # Calculate the conditioning dimension's fraction once
        val_i0 = u_array[i0]
        lower_i0 = i_idx * inv_dims[i0]
        upper_i0 = (i_idx + 1) * inv_dims[i0]
        overlap_len_i0 = max(0.0, min(val_i0, upper_i0) - lower_i0)
        frac_i = overlap_len_i0 * self.dim[i0]  # Multiply by dim instead of dividing

        # Calculate numerator
        num = 0.0
        for c in slice_indices:
            cell_mass = self.matr[c]
            if cell_mass <= 0:
                continue

            qualifies = True

            for j in range(self.d):
                if j == i0:
                    continue  # Skip conditioning dimension

                val_j = u_array[j]
                lower_j = c[j] * inv_dims[j]

                # Early termination check - more efficient boundary comparison
                if val_j <= lower_j:
                    qualifies = False
                    break

                upper_j = (c[j] + 1) * inv_dims[j]

                # If val_j >= upper_j, entire cell dimension is included, so continue
                if val_j >= upper_j:
                    continue

                # Partial overlap - calculate fraction
                overlap_len = (
                    val_j - lower_j
                )  # No need for max() since we know val_j > lower_j
                frac_j = (
                    overlap_len * self.dim[j]
                )  # Multiply by dim instead of dividing

                # More efficient fraction comparison with numerical stability
                if frac_j < frac_i and abs(frac_j - frac_i) > 1e-10:
                    qualifies = False
                    break

            if qualifies:
                num += cell_mass

        return num / denom

    @property
    def pdf(self):
        raise PropertyUnavailableException("PDF does not exist for CheckMin.")

    def rvs(self, n=1):
        """
        More efficient implementation of random variate sampling.
        """
        log.info(f"Generating {n} random variates for {self}...")
        # Get cell indices according to their probability weights
        _, idxs = self._weighted_random_selection(self.matr, n)

        # Generate n random numbers uniformly in [0, 1]
        randoms = np.random.uniform(size=n)

        # Pre-allocate the output array for better performance
        out = np.zeros((n, self.d))

        # Pre-compute inverse dimensions (1/dim) for faster division
        inv_dims = np.array([1.0 / dim for dim in self.dim])

        # Process each selected cell more efficiently
        for i, (c, u) in enumerate(zip(idxs, randoms)):
            # Vectorized computation of lower bounds
            lower_bounds = np.array([c[d] * inv_dims[d] for d in range(self.d)])

            # Vectorized computation of ranges (upper - lower)
            ranges = np.array(
                [(c[d] + 1) * inv_dims[d] - lower_bounds[d] for d in range(self.d)]
            )

            # Directly compute the interpolated point and store in output array
            out[i] = lower_bounds + u * ranges

        return out

    @staticmethod
    def _weighted_random_selection(matrix, num_samples):
        arr = np.asarray(matrix, dtype=float).ravel()
        p = arr / arr.sum()

        flat_indices = np.random.choice(np.arange(arr.size), size=num_samples, p=p)
        shape = matrix.shape
        multi_idx = [np.unravel_index(ix, shape) for ix in flat_indices]
        selected_elements = matrix[tuple(np.array(multi_idx).T)]
        return selected_elements, multi_idx

    def lambda_L(self):
        return 1

    def lambda_U(self):
        return 1


if __name__ == "__main__":
    ccop = CheckMin([[1, 2], [2, 1]])
    ccop.cdf((0.2, 0.2))
