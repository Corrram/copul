import itertools
import logging
import importlib  # Import at module level

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
                    bcp_module = importlib.import_module(
                        "copul.checkerboard.bivcheckmin"
                    )
                    BivCheckMin = getattr(bcp_module, "BivCheckMin")
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
            
            if hasattr(arg, 'ndim') and hasattr(arg, 'shape'):
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
            
            elif hasattr(arg, '__len__'):
                # List, tuple, or similar sequence
                if len(arg) == self.dim:
                    # Single point as a sequence
                    return self._cdf_single_point(np.array(arg, dtype=float))
                else:
                    raise ValueError(
                        f"Expected point with {self.dim} dimensions, got {len(arg)}"
                    )
            
            else:
                # Single scalar value - only valid for 1D case
                if self.dim == 1:
                    return self._cdf_single_point(np.array([arg], dtype=float))
                else:
                    raise ValueError(
                        f"Single scalar provided but copula has {self.dim} dimensions"
                    )
        
        else:
            # Multiple arguments provided
            if len(args) == self.dim:
                # Separate coordinates for a single point
                return self._cdf_single_point(np.array(args, dtype=float))
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}"
                )

    def _cdf_single_point(self, u):
        """
        Helper method to compute CDF for a single point using the min-fraction approach.
        
        Parameters
        ----------
        u : numpy.ndarray
            1D array of length dim representing a single point.
            
        Returns
        -------
        float
            CDF value at the point.
        """
        # Quick boundaries
        if np.any(u <= 0):
            return 0.0
        if np.all(u >= 1):
            return 1.0
        
        total = 0.0
        for c in itertools.product(*(range(s) for s in self.matr.shape)):
            cell_mass = self.matr[c]
            if cell_mass <= 0:
                continue
            
            # min-fraction across dims
            frac_cell = 1.0
            for k in range(self.dim):
                lower_k = c[k] / self.matr.shape[k]
                upper_k = (c[k] + 1) / self.matr.shape[k]
                overlap_k = max(0.0, min(u[k], upper_k) - lower_k)
                if overlap_k <= 0:
                    frac_cell = 0.0
                    break
                width_k = 1.0 / self.matr.shape[k]
                frac_k = overlap_k / width_k
                if frac_k > 1.0:
                    frac_k = 1.0
                if frac_k < frac_cell:
                    frac_cell = frac_k
            
            if frac_cell > 0:
                total += cell_mass * frac_cell
        
        return float(total)

    def _cdf_vectorized_impl(self, points):
        """
        Implementation of vectorized CDF for multiple points using the min-fraction approach.
        
        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (n_points, dim) where each row is a point.
            
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
        
        # Precompute cell information for more efficient processing
        matr_shape = np.array(self.matr.shape)
        
        # Create a numpy meshgrid of indices for the cells
        indices = np.meshgrid(*[np.arange(s) for s in self.matr.shape], indexing='ij')
        indices = np.array([idx.ravel() for idx in indices]).T  # Shape: (n_cells, dim)
        
        # Only process cells with positive mass
        cell_masses = self.matr.ravel()
        positive_mass_mask = cell_masses > 0
        if not np.any(positive_mass_mask):
            return results  # All cells have zero mass
        
        indices = indices[positive_mass_mask]
        cell_masses = cell_masses[positive_mass_mask]
        
        # Reshape for broadcasting
        indices = indices[:, np.newaxis, :]  # Shape: (n_cells, 1, dim)
        compute_points = compute_points[np.newaxis, :, :]  # Shape: (1, n_points_to_compute, dim)
        cell_masses = cell_masses[:, np.newaxis]  # Shape: (n_cells, 1)
        
        # Calculate cell bounds
        lower_bounds = indices / matr_shape[np.newaxis, np.newaxis, :]
        upper_bounds = (indices + 1) / matr_shape[np.newaxis, np.newaxis, :]
        
        # Calculate overlap with [0, u] for each dimension and each point
        overlaps = np.maximum(0.0, np.minimum(compute_points, upper_bounds) - lower_bounds)
        
        # Convert to fraction of cell width
        cell_widths = 1.0 / matr_shape[np.newaxis, np.newaxis, :]
        fractions = overlaps / cell_widths
        fractions = np.clip(fractions, 0.0, 1.0)
        
        # Calculate the min fraction across dimensions for each cell and point
        # This is the key difference from CheckPi which uses a product of fractions
        min_fractions = np.min(fractions, axis=2)  # Shape: (n_cells, n_points_to_compute)
        
        # Early termination for cells with zero overlap in any dimension
        zero_overlap_mask = np.any(overlaps <= 0, axis=2)
        min_fractions[zero_overlap_mask] = 0.0
        
        # Calculate weighted sum for each point
        weighted_fractions = cell_masses * min_fractions  # Shape: (n_cells, n_points_to_compute)
        point_results = np.sum(weighted_fractions, axis=0)  # Shape: (n_points_to_compute,)
        
        # Put results back in the output array
        results[compute_mask] = point_results
        
        return results

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
            
            if hasattr(arg, 'ndim') and hasattr(arg, 'shape'):
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
            
            elif hasattr(arg, '__len__'):
                # List, tuple, or similar sequence
                if len(arg) == self.dim:
                    # Single point as a sequence
                    return self._cond_distr_single(i, np.array(arg, dtype=float))
                else:
                    raise ValueError(
                        f"Expected point with {self.dim} dimensions, got {len(arg)}"
                    )
            
            else:
                # Single scalar value - only valid for 1D case
                if self.dim == 1:
                    return self._cond_distr_single(i, np.array([arg], dtype=float))
                else:
                    raise ValueError(
                        f"Single scalar provided but copula has {self.dim} dimensions"
                    )
        
        else:
            # Multiple arguments provided
            if len(args) == self.dim:
                # Separate coordinates for a single point
                return self._cond_distr_single(i, np.array(args, dtype=float))
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}"
                )

    def _cond_distr_single(self, i, u):
        """
        Helper method for conditional distribution of a single point.
        This preserves the original implementation for CheckMin for a single point.
        
        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        u : numpy.ndarray
            Single point as a 1D array of length dim.
            
        Returns
        -------
        float
            Conditional distribution value.
        """
        i0 = i - 1  # Convert to 0-based index
        
        # Find which cell index along dim i0 the coordinate u[i0] falls into
        x_i = u[i0]
        if x_i < 0:
            return 0.0  # If 'conditioning coordinate' <0, prob is 0
        elif x_i >= 1:
            # If 'conditioning coordinate' >=1, then we pick the last cell index
            i_idx = self.matr.shape[i0] - 1
        else:
            i_idx = int(np.floor(x_i * self.matr.shape[i0]))
            # clamp (just in case)
            if i_idx < 0:
                i_idx = 0
            elif i_idx >= self.matr.shape[i0]:
                i_idx = self.matr.shape[i0] - 1
        
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
            indices = [range(s) for s in self.matr.shape]
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
        inv_dims = np.array([1.0 / dim for dim in self.matr.shape])
        u_array = np.array(u)
        
        # Calculate the conditioning dimension's fraction once
        val_i0 = u_array[i0]
        lower_i0 = i_idx * inv_dims[i0]
        upper_i0 = (i_idx + 1) * inv_dims[i0]
        overlap_len_i0 = max(0.0, min(val_i0, upper_i0) - lower_i0)
        frac_i = overlap_len_i0 * self.matr.shape[i0]  # Multiply by dim instead of dividing
        
        # Calculate numerator
        num = 0.0
        for c in slice_indices:
            cell_mass = self.matr[c]
            if cell_mass <= 0:
                continue
            
            qualifies = True
            
            for j in range(self.dim):
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
                overlap_len = val_j - lower_j  # No need for max() since we know val_j > lower_j
                frac_j = overlap_len * self.matr.shape[j]  # Multiply by dim instead of dividing
                
                # More efficient fraction comparison with numerical stability
                if frac_j < frac_i and abs(frac_j - frac_i) > 1e-10:
                    qualifies = False
                    break
            
            if qualifies:
                num += cell_mass
        
        return num / denom

    def _cond_distr_vectorized(self, i, points):
        """
        Vectorized implementation of conditional distribution for multiple points.
        Adapted for the CheckMin-specific logic.
        
        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        points : numpy.ndarray
            Multiple points as a 2D array of shape (n_points, dim).
            
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
        
        # Process each point separately - the CheckMin cond_distr algorithm isn't easily vectorizable
        # across multiple points at once due to its conditional logic
        for p_idx, u in enumerate(points):
            x_i = u[i0]
            
            # Special case: conditioning coordinate < 0
            if x_i < 0:
                results[p_idx] = 0.0
                continue
            
            # Determine the cell index along the conditioning dimension
            if x_i >= 1:
                i_idx = self.matr.shape[i0] - 1
            else:
                i_idx = int(np.floor(x_i * self.matr.shape[i0]))
                # clamp (just in case)
                i_idx = max(0, min(i_idx, self.matr.shape[i0] - 1))
            
            # Create a slice for efficient indexing
            slice_spec = [slice(None)] * self.dim
            slice_spec[i0] = i_idx
            
            # Get the slice of the matrix - all cells with index i_idx in dimension i0
            slice_matr = self.matr[tuple(slice_spec)]
            
            # Calculate denominator - sum of all cells in the slice
            denom = np.sum(slice_matr)
            
            if denom <= 0:
                results[p_idx] = 0.0
                continue
            
            # For numerator, we need to check each cell in the slice against CheckMin's criteria
            # Create indices for all cells in the slice
            remaining_dims = [d for d in range(self.dim) if d != i0]
            mesh_dims = [range(self.matr.shape[d]) for d in remaining_dims]
            
            if mesh_dims:  # Only create meshgrid if we have remaining dimensions
                # Create meshgrid for remaining dimensions
                mesh = np.meshgrid(*mesh_dims, indexing='ij')
                indices = np.array([idx.ravel() for idx in mesh]).T  # Shape: (n_cells_in_slice, len(remaining_dims))
                
                # Expand indices to include the fixed i0 dimension
                full_indices = []
                for idx in indices:
                    full_idx = []
                    rem_dim_idx = 0
                    for d in range(self.dim):
                        if d == i0:
                            full_idx.append(i_idx)
                        else:
                            full_idx.append(idx[rem_dim_idx])
                            rem_dim_idx += 1
                    full_indices.append(tuple(full_idx))
                
                # Precompute 1/dim values for faster calculations
                inv_dims = np.array([1.0 / dim for dim in self.matr.shape])
                
                # Calculate the conditioning dimension's fraction once
                val_i0 = u[i0]
                lower_i0 = i_idx * inv_dims[i0]
                upper_i0 = (i_idx + 1) * inv_dims[i0]
                overlap_len_i0 = max(0.0, min(val_i0, upper_i0) - lower_i0)
                frac_i = overlap_len_i0 * self.matr.shape[i0]
                
                # Calculate numerator using CheckMin's specific logic
                numerator = 0.0
                for c in full_indices:
                    cell_mass = self.matr[c]
                    if cell_mass <= 0:
                        continue
                    
                    qualifies = True
                    
                    for j in range(self.dim):
                        if j == i0:
                            continue  # Skip conditioning dimension
                        
                        val_j = u[j]
                        lower_j = c[j] * inv_dims[j]
                        
                        # Early termination check
                        if val_j <= lower_j:
                            qualifies = False
                            break
                        
                        upper_j = (c[j] + 1) * inv_dims[j]
                        
                        # If val_j >= upper_j, entire cell is included
                        if val_j >= upper_j:
                            continue
                        
                        # Partial overlap - calculate fraction
                        overlap_len = val_j - lower_j
                        frac_j = overlap_len * self.matr.shape[j]
                        
                        # CheckMin-specific criterion
                        if frac_j < frac_i and abs(frac_j - frac_i) > 1e-10:
                            qualifies = False
                            break
                    
                    if qualifies:
                        numerator += cell_mass
                
                results[p_idx] = numerator / denom
            else:
                # Special case: only one dimension
                results[p_idx] = 1.0 if denom > 0 else 0.0
        
        return results

    def pdf(self, *args):
        """
        Evaluate the piecewise PDF at one or multiple points.
        
        Note: PDF does not properly exist for CheckMin copula, so this method
        always returns 0 for any inputs.
        
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
            Always returns 0 (or array of zeros for multiple points).
        
        Examples
        --------
        # Single point
        value = copula.pdf(0.3, 0.7)  # Returns 0.0
        
        # Multiple points
        values = copula.pdf(np.array([[0.1, 0.2], [0.3, 0.4]]))  # Returns array of zeros
        """
        # Handle different input formats and return appropriate zeros
        if len(args) == 0:
            raise ValueError("No arguments provided")
        
        elif len(args) == 1:
            # A single argument was provided - either a point or multiple points
            arg = args[0]
            
            if hasattr(arg, 'ndim') and hasattr(arg, 'shape'):
                # NumPy array or similar
                arr = np.asarray(arg, dtype=float)
                
                if arr.ndim == 1:
                    # 1D array - single point
                    if len(arr) != self.dim:
                        raise ValueError(
                            f"Expected point array of length {self.dim}, got {len(arr)}"
                        )
                    return 0.0
                    
                elif arr.ndim == 2:
                    # 2D array - multiple points
                    if arr.shape[1] != self.dim:
                        raise ValueError(
                            f"Expected points with {self.dim} dimensions, got {arr.shape[1]}"
                        )
                    return np.zeros(arr.shape[0])
                    
                else:
                    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array")
            
            elif hasattr(arg, '__len__'):
                # List, tuple, or similar sequence
                if len(arg) == self.dim:
                    # Single point as a sequence
                    return 0.0
                else:
                    raise ValueError(
                        f"Expected point with {self.dim} dimensions, got {len(arg)}"
                    )
            
            else:
                # Single scalar value - only valid for 1D case
                if self.dim == 1:
                    return 0.0
                else:
                    raise ValueError(
                        f"Single scalar provided but copula has {self.dim} dimensions"
                    )
        
        else:
            # Multiple arguments provided
            if len(args) == self.dim:
                # Separate coordinates for a single point
                return 0.0
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}"
                )
    @property
    def pdf(self):
        raise PropertyUnavailableException("PDF does not exist for CheckMin.")

    def rvs(self, n=1, random_state=None):
        """
        More efficient implementation of random variate sampling.
        """
        if random_state is not None:
            np.random.seed(random_state)
        log.info(f"Generating {n} random variates for {self}...")
        # Get cell indices according to their probability weights
        _, idxs = self._weighted_random_selection(self.matr, n)

        # Generate n random numbers uniformly in [0, 1]
        randoms = np.random.uniform(size=n)

        # Pre-allocate the output array for better performance
        out = np.zeros((n, self.dim))

        # Pre-compute inverse dimensions (1/dim) for faster division
        inv_dims = np.array([1.0 / dim for dim in self.matr.shape])

        # Process each selected cell more efficiently
        for i, (c, u) in enumerate(zip(idxs, randoms)):
            # Vectorized computation of lower bounds
            lower_bounds = np.array([c[d] * inv_dims[d] for d in range(self.dim)])

            # Vectorized computation of ranges (upper - lower)
            ranges = np.array(
                [(c[d] + 1) * inv_dims[d] - lower_bounds[d] for d in range(self.dim)]
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


if __name__ == "__main__":
    ccop = CheckMin([[1, 2], [2, 1]])
    ccop.cdf((0.2, 0.2))
