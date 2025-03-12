import logging

import numpy as np

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.exceptions import PropertyUnavailableException

log = logging.getLogger(__name__)


class BivCheckW(BivCheckPi):
    """
    Bivariate checkerboard W-copula (2D only).

    1) The CDF uses a piecewise 'W-fraction':
         frac_ij = max(0, frac_x + frac_y - 1),
       where frac_x, frac_y in [0,1] are the proportions of cell (i,j)
       that lie below (u,v).

    2) The conditional distribution uses a discrete approach:
       - Finds the cell-slice in the conditioning dimension
       - Denominator: sum of masses in that slice
       - Numerator: sum of slice cells that lie fully below the threshold
       - cond_distr = numerator / denominator
    """

    def __init__(self, matr):
        """
        Initialize the 2D W-copula with a matrix of nonnegative weights.

        :param matr: 2D array/list of nonnegative weights. Will be normalized to sum=1.
        """
        super().__init__(matr)

    def __str__(self):
        return f"BivCheckW(m={self.m}, n={self.n})"

    @property
    def is_absolutely_continuous(self):
        """Checkerboard W-copula is not absolutely continuous (it has jumps along cell edges)."""
        return False

    @property
    def is_symmetric(self):
        """Check if m = n and the matrix is symmetric about the diagonal."""
        if self.m != self.n:
            return False
        return np.allclose(self.matr, self.matr.T)

    def cdf(self, u, v):
        """
        Evaluate CDF at (u, v) in [0,1]^2, using:
           fraction_ij = max(0, fx + fy - 1),
        where:
          fx = overlap_x * m,
          fy = overlap_y * n.
        """
        # Quick boundary checks
        if u <= 0 or v <= 0:
            return 0.0
        if u >= 1 and v >= 1:
            return 1.0

        total = 0.0
        for i in range(self.m):
            for j in range(self.n):
                w_ij = self.matr[i, j]
                if w_ij <= 0:
                    continue

                x0, x1 = i / self.m, (i + 1) / self.m
                y0, y1 = j / self.n, (j + 1) / self.n

                overlap_x = max(0.0, min(u, x1) - x0)
                overlap_y = max(0.0, min(v, y1) - y0)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                fx = overlap_x * self.m
                fy = overlap_y * self.n
                frac_ij = max(0.0, fx + fy - 1.0)
                if frac_ij > 0:
                    total += w_ij * frac_ij

        return float(total)

    def cond_distr(self, i: int, u):
        """
        Compute conditional distribution for dimension i (1-based).
        """
        if i < 1 or i > 2:  # Hardcoded 2D
            raise ValueError(f"Dimension {i} out of range 1..2")

        i0 = i - 1
        if len(u) != 2:  # Hardcoded 2D
            raise ValueError("Point u must have length 2.")

        # Find which cell the conditioning coordinate falls into
        x_i = u[i0]
        if x_i < 0:
            return 0.0
        elif x_i >= 1:
            i_idx = self.m if i0 == 0 else self.n
            i_idx -= 1
        else:
            dim_size = self.m if i0 == 0 else self.n
            i_idx = int(np.floor(x_i * dim_size))
            # clamp
            if i_idx < 0:
                i_idx = 0
            elif i_idx >= dim_size:
                i_idx = dim_size - 1

        # For safety, reset the intervals cache between calls
        self.intervals = {}

        # Cache key for the slice indices
        slice_key = (i, i_idx)

        # Calculate denominator - sum of all cells in the slice
        if slice_key in self.intervals:
            slice_indices = self.intervals[slice_key]
            denom = sum(self.matr[c] for c in slice_indices)
        else:
            # Create slice iteration
            if i0 == 0:  # Fix row
                denom = 0.0
                slice_indices = []
                for j in range(self.n):
                    cell_mass = self.matr[i_idx, j]
                    denom += cell_mass
                    slice_indices.append((i_idx, j))
            else:  # Fix column
                denom = 0.0
                slice_indices = []
                for i in range(self.m):
                    cell_mass = self.matr[i, i_idx]
                    denom += cell_mass
                    slice_indices.append((i, i_idx))

            # Store in cache
            self.intervals[slice_key] = slice_indices

        if denom <= 0:
            return 0.0

        # Calculate the conditioning dimension's fraction
        val_i0 = u[i0]
        dim_size = self.m if i0 == 0 else self.n
        lower_i0 = i_idx / dim_size
        upper_i0 = (i_idx + 1) / dim_size
        overlap_len_i0 = max(0.0, min(val_i0, upper_i0) - lower_i0)
        frac_i = overlap_len_i0 * dim_size

        # Calculate numerator
        num = 0.0
        for c in slice_indices:
            cell_mass = self.matr[c]
            if cell_mass <= 0:
                continue

            # Check other dimension (not i0)
            j = 1 - i0  # If i0 is 0, j is 1; if i0 is 1, j is 0
            val_j = u[j]
            dim_size_j = self.n if j == 1 else self.m
            lower_j = c[j] / dim_size_j

            # Early exit if below threshold
            if val_j <= lower_j:
                continue

            upper_j = (c[j] + 1) / dim_size_j

            # If val_j >= upper_j, entire cell is included
            if val_j >= upper_j:
                num += cell_mass
                continue

            # Partial overlap - calculate fraction
            overlap_len = val_j - lower_j
            frac_j = overlap_len * dim_size_j

            # W-copula condition
            if frac_j + frac_i >= 1:
                num += cell_mass

        return num / denom

    def rvs(self, n=1):
        """Generate n random samples."""
        # Get cell indices according to their probability weights
        _, idxs = self._weighted_random_selection(self.matr, n)

        # Generate n random numbers uniformly in [0, 1]
        randoms = np.random.uniform(size=n)

        # Pre-allocate output array
        out = np.zeros((n, 2))  # Hardcoded 2D

        # Process each selected cell
        for i, (c, u) in enumerate(zip(idxs, randoms)):
            # Compute bounds and ranges
            lower_x = c[0] / self.m
            range_x = 1 / self.m
            lower_y = c[1] / self.n
            range_y = 1 / self.n

            # Compute the interpolated point
            out[i, 0] = lower_x + u * range_x
            out[i, 1] = lower_y + (1 - u) * range_y

        return out

    def lambda_L(self):
        """Lower tail dependence (0 for W-copula)."""
        return 0

    def lambda_U(self):
        """Upper tail dependence (1 for W-copula)."""
        return 1

    @property
    def pdf(self):
        raise PropertyUnavailableException(
            "No continuous PDF for discrete W-checkerboard."
        )
