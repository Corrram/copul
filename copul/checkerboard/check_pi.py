import itertools
import numpy as np
import sympy


class CheckPi:
    params = []
    intervals = {}

    def __init__(self, matr):
        """
        Initialize a checkerboard (piecewise-uniform) copula with the given weight matrix.

        Parameters
        ----------
        matr : array-like
            d-dimensional array of nonnegative weights (one per cell).
            They will be normalized so that the total sum is 1.
        """
        if isinstance(matr, list):
            matr = np.array(matr)

        # Normalize the matrix so it sums to 1
        matr_sum = sum(matr) if isinstance(matr, sympy.Matrix) else matr.sum()
        self.matr = matr / matr_sum

        # Store shape and dimension
        self.dim = self.matr.shape
        self.d = len(self.dim)

    def __str__(self):
        return f"CheckerboardCopula({self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def cdf(self, *args):
        """
        Evaluate the CDF at point (args) in [0,1]^d.

        We do a piecewise-uniform construction:
        Sum over all cells the fraction of that cell lying in the hyper-rectangle [0,args[1]] x ... x [0,args[d]].
        """
        if len(args) != self.d:
            raise ValueError(f"cdf expects {self.d} coordinates, got {len(args)}.")

        # Short-circuit checks (optional, not strictly needed):
        if any(u <= 0 for u in args):
            return 0.0
        if all(u >= 1 for u in args):
            return 1.0

        total = 0.0

        # Enumerate all possible cell indices in the grid
        all_indices = itertools.product(*(range(s) for s in self.dim))
        for idx in all_indices:
            cell_mass = self.matr[idx]
            if cell_mass <= 0:
                continue

            # Calculate how much of this cell is below the point 'args'
            fraction = 1.0
            for k in range(self.d):
                # Cell k-th dim covers [lower_k, upper_k)
                lower_k = idx[k] / self.dim[k]
                upper_k = (idx[k] + 1) / self.dim[k]
                # Overlap with [0, args[k]]
                overlap_len = max(0.0, min(args[k], upper_k) - lower_k)

                # Convert overlap length to fraction of cell's width in dim k
                cell_width = 1.0 / self.dim[k]
                frac_k = overlap_len / cell_width  # fraction in [0,1]
                if frac_k <= 0:
                    fraction = 0.0
                    break
                if frac_k > 1.0:
                    frac_k = 1.0  # clamp to the cell’s boundary

                fraction *= frac_k
                if fraction == 0.0:
                    break

            if fraction > 0.0:
                total += cell_mass * fraction

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

        # 1) DENOMINATOR: sum of all cells in "slice" c[i0] = i_idx
        #    ignoring partial coverage in dimension i0.
        denom = 0.0
        # We'll collect those cell indices for potential partial coverage in the numerator
        slice_indices = []
        for c in itertools.product(*(range(s) for s in self.dim)):
            if c[i0] == i_idx:
                # That cell is part of the conditioning event
                denom += self.matr[c]
                slice_indices.append(c)

        if denom <= 0:
            return 0.0

        # 2) NUMERATOR: Among that same slice, we see how much is below u[j] in each j != i0
        #    using partial overlap logic dimension by dimension, but skipping dimension i0.
        num = 0.0
        for c in slice_indices:
            cell_mass = self.matr[c]
            fraction = 1.0
            for j in range(self.d):
                if j == i0:
                    # No partial coverage in the conditioning dimension
                    continue

                # Cell j covers [c[j]/dim[j], (c[j]+1)/dim[j])
                lower_j = c[j] / self.dim[j]
                upper_j = (c[j] + 1) / self.dim[j]
                val_j = u[j]

                # Overlap with [0, val_j] in this dimension
                if val_j <= 0:
                    fraction = 0.0
                    break
                if val_j >= 1:
                    # entire cell dimension is included
                    continue
                # partial
                overlap_len = max(0.0, min(val_j, upper_j) - lower_j)
                cell_width = 1.0 / self.dim[j]
                frac_j = overlap_len / cell_width  # fraction in [0,1]
                if frac_j <= 0:
                    fraction = 0.0
                    break
                if frac_j > 1.0:
                    frac_j = 1.0
                fraction *= frac_j
                if fraction == 0.0:
                    break

            if fraction > 0.0:
                num += cell_mass * fraction

        return num / denom

    def cond_distr_1(self, u):
        """F_{U_{-1}|U_1}(u_{-1} | u_1)."""
        return self.cond_distr(1, u)

    def cond_distr_2(self, u):
        """F_{U_{-2}|U_2}(u_{-2} | u_2)."""
        return self.cond_distr(2, u)

    def pdf(self, *args):
        """
        Evaluate the piecewise "PDF" at the point (args).
        In a piecewise-uniform sense, the *true* PDF would be:
            self.matr[idx_cell] / volume_of_that_cell
        but some tests want just the cell's probability mass.
        """
        if len(args) != self.d:
            raise ValueError(f"pdf expects {self.d} coords, got {len(args)}.")
        if any(a < 0 or a > 1 for a in args):
            return 0.0

        # Identify which cell the point (args) falls into
        cell_idx = []
        for k, val in enumerate(args):
            ix = int(np.floor(val * self.dim[k]))
            if ix < 0:
                ix = 0
            if ix >= self.dim[k]:
                ix = self.dim[k] - 1
            cell_idx.append(ix)

        # Return just the cell's mass (not dividing by cell volume)
        return float(self.matr[tuple(cell_idx)])

    def rvs(self, n=1):
        """
        Draw random variates from the checkerboard copula.
        1) Choose a cell index by probabilities in self.matr.
        2) Uniformly pick a point within that cell.
        """
        sel_ele, sel_idx = self._weighted_random_selection(self.matr, n)

        # offsets for each dimension (uniform in [0,1/dim[k]])
        offsets = np.random.rand(n, self.d)
        for k in range(self.d):
            offsets[:, k] /= self.dim[k]

        # Convert cell indices to their lower corners in [0,1]
        base_points = []
        for idx_tuple in sel_idx:
            base_points.append([idx_tuple[k] / self.dim[k] for k in range(self.d)])
        base_points = np.array(base_points)

        # Final = lower corner + uniform offset
        return base_points + offsets

    @staticmethod
    def _weighted_random_selection(matrix, num_samples):
        """
        Select elements from 'matrix' with probability proportional to matrix entries.
        Return (selected_values, selected_multi_indices).
        """
        arr = np.asarray(matrix, dtype=float).ravel()
        p = arr / arr.sum()

        flat_indices = np.random.choice(np.arange(arr.size), size=num_samples, p=p)
        shape = matrix.shape
        multi_idx = [np.unravel_index(ix, shape) for ix in flat_indices]
        selected_elements = matrix[tuple(np.array(multi_idx).T)]
        return selected_elements, multi_idx

    def lambda_L(self):
        """Lower tail dependence (usually 0 for a checkerboard copula)."""
        return 0

    def lambda_U(self):
        """Upper tail dependence (usually 0 for a checkerboard copula)."""
        return 0


if __name__ == "__main__":
    matr = [[1, 5, 4], [5, 3, 2], [4, 2, 4]]
    copul = CheckPi(matr)
    copul.plot_cdf()
