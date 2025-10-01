import logging
import warnings

from typing_extensions import TypeAlias
import numpy as np

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.exceptions import PropertyUnavailableException


log = logging.getLogger(__name__)


class BivCheckW(BivCheckPi):
    """
    Bivariate checkerboard W-copula (2D only).

    The copula is defined as follows:

    1) **CDF**: Uses a piecewise 'W-fraction':

       ``frac_ij = max(0, frac_x + frac_y - 1)``

       where ``frac_x`` and ``frac_y`` (both in [0,1]) are the proportions of cell (i,j)
       that lie below (u,v).

    2) **Conditional Distribution**: Uses a discrete approach:

       - Finds the cell-slice in the conditioning dimension.
       - **Denominator**: Sum of masses in that slice.
       - **Numerator**: Sum of slice cells that lie fully below the threshold.
       - ``cond_distr = numerator / denominator``

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

    @classmethod
    def generate_randomly(cls, grid_size: int | list | None = None, n=1):
        generated = BivCheckPi.generate_randomly(grid_size, n)
        if n == 1:
            return cls(generated.matr)  # <-- use .matr
        else:
            return [cls(c.matr) for c in generated]

    def transpose(self):
        """
        Transpose the checkerboard matrix.
        """
        return BivCheckW(self.matr.T)

    def cdf(self, *args):
        """
        Compute the CDF at one or multiple points using the W-fraction approach.

        This method handles both single-point and multi-point CDF evaluation
        in an efficient vectorized manner.

        Parameters
        ----------
        *args : array-like or float
            Either:
            - Two separate coordinates (u, v) of a single point
            - A single array-like object with coordinates [u, v] of a single point
            - A 2D array where each row represents a separate point [u, v]

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
                    if len(arr) != 2:  # Hardcoded 2D
                        raise ValueError(
                            f"Expected point array of length 2, got {len(arr)}"
                        )
                    return self._cdf_single_point(arr[0], arr[1])

                elif arr.ndim == 2:
                    # 2D array - multiple points
                    if arr.shape[1] != 2:  # Hardcoded 2D
                        raise ValueError(
                            f"Expected points with 2 dimensions, got {arr.shape[1]}"
                        )
                    return self._cdf_vectorized_impl(arr)

                else:
                    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array")

            elif hasattr(arg, "__len__"):
                # List, tuple, or similar sequence
                if len(arg) == 2:  # Hardcoded 2D
                    # Single point as a sequence
                    return self._cdf_single_point(arg[0], arg[1])
                else:
                    raise ValueError(
                        f"Expected point with 2 dimensions, got {len(arg)}"
                    )

            else:
                # Single scalar value - not valid for 2D case
                raise ValueError(
                    "Single scalar provided but copula requires 2 dimensions"
                )

        elif len(args) == 2:
            # Two arguments provided for u and v coordinates
            return self._cdf_single_point(args[0], args[1])

        else:
            # Too many arguments
            raise ValueError(f"Expected 2 coordinates, got {len(args)}")

    def _cdf_single_point(self, u, v):
        """
        Helper method to compute CDF for a single point using the W-fraction approach.

        Parameters
        ----------
        u : float
            First coordinate.
        v : float
            Second coordinate.

        Returns
        -------
        float
            CDF value at the point.
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

    def _cdf_vectorized_impl(self, points):
        """
        Implementation of vectorized CDF for multiple points using the W-fraction approach.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (n_points, 2) where each row is a point.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_points,) with CDF values.
        """
        # Convert to numpy array
        points = np.asarray(points, dtype=float)

        n_points = points.shape[0]
        results = np.zeros(n_points)

        # Create mask arrays for special cases
        all_zeros_mask = np.any(points <= 0, axis=1)
        all_ones_mask = np.all(points >= 1, axis=1)

        # Set results for special cases
        results[all_zeros_mask] = 0.0
        results[all_ones_mask] = 1.0

        # Filter out points that need actual computation
        compute_mask = ~(all_zeros_mask | all_ones_mask)
        compute_points = points[compute_mask]

        if len(compute_points) == 0:
            return results

        # Create a 2D grid of cell indices for vectorized computation
        i_indices, j_indices = np.meshgrid(range(self.m), range(self.n), indexing="ij")
        i_indices = i_indices.ravel()
        j_indices = j_indices.ravel()

        # Get cell masses
        cell_masses = self.matr.ravel()

        # Filter out cells with zero mass
        positive_mask = cell_masses > 0
        if not np.any(positive_mask):
            return results

        i_indices = i_indices[positive_mask]
        j_indices = j_indices[positive_mask]
        cell_masses = cell_masses[positive_mask]

        # Calculate cell bounds
        x0 = i_indices[:, np.newaxis] / self.m
        x1 = (i_indices[:, np.newaxis] + 1) / self.m
        y0 = j_indices[:, np.newaxis] / self.n
        y1 = (j_indices[:, np.newaxis] + 1) / self.n

        # Calculate overlaps for all points
        u_values = compute_points[:, 0]
        v_values = compute_points[:, 1]

        u_values = u_values[np.newaxis, :]
        v_values = v_values[np.newaxis, :]

        # Calculate x and y overlaps
        overlap_x = np.maximum(0.0, np.minimum(u_values, x1) - x0)
        overlap_y = np.maximum(0.0, np.minimum(v_values, y1) - y0)

        # Calculate fractions - applying the W-copula specific formula
        fx = overlap_x * self.m
        fy = overlap_y * self.n
        frac_ij = np.maximum(0.0, fx + fy - 1.0)

        # Zero out contributions from cells with no overlap
        zero_overlap_mask = (overlap_x <= 0) | (overlap_y <= 0)
        frac_ij[zero_overlap_mask] = 0.0

        # Calculate weighted contributions
        weighted_fractions = cell_masses[:, np.newaxis] * frac_ij

        # Sum contributions for each point
        point_results = np.sum(weighted_fractions, axis=0)

        # Put results back in the output array
        results[compute_mask] = point_results

        return results

    def cond_distr(self, i, *args):
        """
        Compute the conditional distribution for one or multiple points.

        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on (1 or 2).
        *args : array-like or float
            Either:
            - Two separate coordinates (u, v) of a single point
            - A single array-like object with coordinates [u, v] of a single point
            - A 2D array where each row represents a separate point [u, v]

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
        if i < 1 or i > 2:  # Hardcoded 2D
            raise ValueError(f"Dimension {i} out of range 1..2")

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
                    if len(arr) != 2:  # Hardcoded 2D
                        raise ValueError(
                            f"Expected point array of length 2, got {len(arr)}"
                        )
                    return self._cond_distr_single(i, arr)

                elif arr.ndim == 2:
                    # 2D array - multiple points
                    if arr.shape[1] != 2:  # Hardcoded 2D
                        raise ValueError(
                            f"Expected points with 2 dimensions, got {arr.shape[1]}"
                        )
                    return self._cond_distr_vectorized(i, arr)

                else:
                    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array")

            elif hasattr(arg, "__len__"):
                # List, tuple, or similar sequence
                if len(arg) == 2:  # Hardcoded 2D
                    # Single point as a sequence
                    return self._cond_distr_single(i, np.array(arg, dtype=float))
                else:
                    raise ValueError(
                        f"Expected point with 2 dimensions, got {len(arg)}"
                    )

            else:
                # Single scalar value - not valid for 2D case
                raise ValueError(
                    "Single scalar provided but copula requires 2 dimensions"
                )

        elif len(args) == 2:
            # Two arguments provided for u and v coordinates
            return self._cond_distr_single(i, np.array([args[0], args[1]], dtype=float))

        else:
            # Too many arguments
            raise ValueError(f"Expected 2 coordinates, got {len(args)}")

    def _cond_distr_single(self, i, u):
        """
        Helper method for conditional distribution of a single point.
        This preserves the original implementation for W-copula for a single point.

        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        u : numpy.ndarray
            Single point as a 1D array of length 2.

        Returns
        -------
        float
            Conditional distribution value.
        """
        i0 = i - 1  # Convert to 0-based index

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
                for i_idx2 in range(self.m):
                    cell_mass = self.matr[i_idx2, i_idx]
                    denom += cell_mass
                    slice_indices.append((i_idx2, i_idx))

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

    def _cond_distr_vectorized(self, i, points):
        """
        Vectorized implementation of conditional distribution for multiple points.

        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        points : numpy.ndarray
            Multiple points as a 2D array of shape (n_points, 2).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_points,) with conditional distribution values.
        """
        # Convert to numpy array
        points = np.asarray(points, dtype=float)

        n_points = points.shape[0]
        results = np.zeros(n_points)

        # Convert to 0-based index
        i0 = i - 1

        # Process each point separately - the BivCheckW cond_distr algorithm isn't easily vectorizable
        # across multiple points at once due to its conditional logic
        for p_idx, u in enumerate(points):
            x_i = u[i0]

            # Special case: conditioning coordinate < 0
            if x_i < 0:
                results[p_idx] = 0.0
                continue

            # Determine the cell index along the conditioning dimension
            if x_i >= 1:
                i_idx = self.m if i0 == 0 else self.n
                i_idx -= 1
            else:
                dim_size = self.m if i0 == 0 else self.n
                i_idx = int(np.floor(x_i * dim_size))
                # clamp
                i_idx = max(0, min(i_idx, dim_size - 1))

            # Get the slice of cells along the conditioning dimension
            if i0 == 0:  # Fix row
                # Get all cells in this row
                slice_indices = [(i_idx, j) for j in range(self.n)]
            else:  # Fix column
                # Get all cells in this column
                slice_indices = [(i_row, i_idx) for i_row in range(self.m)]

            # Calculate denominator - sum of all cells in the slice
            denom = sum(self.matr[c] for c in slice_indices)

            if denom <= 0:
                results[p_idx] = 0.0
                continue

            # Calculate the conditioning dimension's fraction
            val_i0 = u[i0]
            dim_size = self.m if i0 == 0 else self.n
            lower_i0 = i_idx / dim_size
            upper_i0 = (i_idx + 1) / dim_size
            overlap_len_i0 = max(0.0, min(val_i0, upper_i0) - lower_i0)
            frac_i = overlap_len_i0 * dim_size

            # Calculate numerator using W-copula specific logic
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

            results[p_idx] = num / denom

        return results

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
        """
        Lower tail dependence (0 for W-copula).
        Must be lower-equal to BivCheckPi(self.matr).lambda_L() = 0.
        """
        return 0

    def lambda_U(self):
        """
        Upper tail dependence (0 for W-copula).
        Must be lower-equal to BivCheckPi(self.matr).lambda_U() = 0.
        """
        return 0

    @property
    def pdf(self):
        """PDF is not available for BivCheckW.

        Raises:
            PropertyUnavailableException: Always raised, since PDF does not exist for BivCheckMin.
        """
        raise PropertyUnavailableException("PDF does not exist for BivCheckW.")

    def spearmans_rho(self) -> float:
        return BivCheckPi.spearmans_rho(self) - 1 / (self.m * self.n)

    def kendalls_tau(self) -> float:
        return BivCheckPi.kendalls_tau(self) - np.trace(self.matr.T @ self.matr)

    def chatterjees_xi(
        self,
        condition_on_y: bool = False,
    ) -> float:
        m, n = (self.n, self.m) if condition_on_y else (self.m, self.n)
        return (
            super().chatterjees_xi(condition_on_y)
            + m * np.trace(self.matr.T @ self.matr) / n
        )

    def blests_nu(self) -> float:
        """
        Blest's measure (nu) for a BivCheckW copula.

        Decomposition:
            nu(W) = nu(CheckPi) - singular_add_on,

        where the singular add-on corresponds to the minimum/maximum completion’s
        diagonal segments integrated against the (1-u) weight. Consistent with
        the sign conventions in this class (see footrule/rho/tau), the add-on
        is subtracted for W.

        Closed forms:
            nu(CheckPi)     = (24 / (m^2 n)) * tr(Δ^T K) - 2,  (via BivCheckPi.blests_nu)
            singular_add_on = (24 / m^2) * sum_{i=1}^{min(m,n)} (m - i + 1/2) * Δ_{ii}.
        """
        # Absolutely-continuous (checkerboard Pi) contribution
        nu_pi = super().blests_nu()

        P = np.asarray(self.matr, dtype=float)
        m, n = P.shape

        # Diagonal weights: average of (1-u) along the unit diagonal segment in row i
        # weight_i = (m - i + 1/2) / m; overall scale produces 24 / m^2
        d = np.diag(P)  # length min(m, n)
        i = np.arange(1, d.size + 1, dtype=float)  # 1..min(m,n)
        # Use m in the weights (row-scaling) as in Min; consistent with your class conventions
        weight = m - i + 0.5

        singular_add_on = (24.0 / (m * m)) * np.dot(weight, d)

        # For W we subtract the singular component (mirrors footrule/rho/tau behavior)
        return float(nu_pi - singular_add_on)

    def spearmans_footrule(self) -> float:
        """
        Compute Spearman's Footrule (psi) for a BivCheckMin copula.

        The value is the footrule of the underlying CheckPi copula plus an
        add-on term accounting for the singular part of the distribution.
        Implemented for square checkerboard matrices.

        Returns:
            float: The value of Spearman's Footrule.
        """
        if self.m != self.n:
            warnings.warn(
                "Footrule analytical formula is implemented for square matrices only."
            )
            return np.nan

        # Calculate footrule for the absolutely continuous part (CheckPi)
        check_pi_footrule = super().spearmans_footrule()

        # Add-on term from the singular part of the copula
        # Add-on = (1/n) * trace(P)
        trace = np.trace(self.matr)
        add_on = trace / self.m

        return check_pi_footrule - add_on

    def ginis_gamma(self) -> float:
        """
        Compute Gini's Gamma for a BivCheckMin copula.

        This method corrects the value from the parent BivCheckPi class. The
        parent method incorrectly uses the overridden `footrule` method from
        this child class, leading to a "contaminated" result that already
        includes the add-on for the main diagonal integral. We correct this
        by adding only the missing component from the anti-diagonal integral.
        Implemented for square checkerboard matrices.

        Returns:
            float: The value of Gini's Gamma.
        """
        if self.m != self.n:
            warnings.warn(
                "Gini's Gamma analytical formula is implemented for square matrices only."
            )
            return np.nan

        # The super() call returns a value that has incorrectly incorporated the
        # diagonal add-on but not the anti-diagonal add-on.
        contaminated_gamma_pi = super().ginis_gamma()

        # We add only the part that was missing: the add-on for the
        # anti-diagonal integral C(u, 1-u).
        # Add-on = 4 * (Trace(Anti-Diagonal(P)) / (12n))
        anti_diag_trace = np.trace(np.fliplr(self.matr))

        add_on = anti_diag_trace / (3 * self.m)

        return contaminated_gamma_pi - add_on


CheckW: TypeAlias = BivCheckW

if __name__ == "__main__":
    matr = [[1, 1]]
    copula = BivCheckW(matr)
    ccop = copula.to_checkerboard()
    footrule = ccop.spearmans_footrule()
    rho = ccop.spearmans_rho()
    print(f"Footrule: {footrule}, Rho: {rho}")
