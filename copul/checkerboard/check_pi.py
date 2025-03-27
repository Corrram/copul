import itertools

import numpy as np

from copul.checkerboard.check import Check


class CheckPi(Check):
    def __new__(cls, matr, *args, **kwargs):
        """
        Create a new CheckPi instance or a BivCheckPi instance if dimension is 2.

        Parameters
        ----------
        matr : array-like
            Matrix of values that determine the copula's distribution.
        *args, **kwargs
            Additional arguments passed to the constructor.

        Returns
        -------
        CheckPi or BivCheckPi
            A CheckPi instance, or a BivCheckPi instance if dimension is 2.
        """
        # If this is the CheckPi class itself (not a subclass)
        if cls is CheckPi:
            # Convert matrix to numpy array to get its dimensionality
            matr_arr = np.asarray(matr)

            # Check if it's a 2D matrix (bivariate copula)
            if matr_arr.ndim == 2:
                # Import the BivCheckPi class here to avoid circular imports
                try:
                    # Use importlib approach for better testability
                    import importlib

                    bcp_module = importlib.import_module(
                        "copul.checkerboard.biv_check_pi"
                    )
                    BivCheckPi = getattr(bcp_module, "BivCheckPi")
                    # Return a new BivCheckPi instance with the same arguments
                    return BivCheckPi(matr, *args, **kwargs)
                except (ImportError, ModuleNotFoundError, AttributeError):
                    # If the import fails, just continue with normal instantiation
                    pass

        # Otherwise, create a normal instance of the class
        instance = super().__new__(cls)
        return instance

    def __str__(self):
        return f"CheckPiCopula({self.matr.shape})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def cdf(self, *args):
        """
        Compute the CDF at one or multiple points.

        This method handles both single-point and multi-point CDF evaluation
        in an efficient vectorized manner.

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
                raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}")

    def _cdf_single_point(self, u):
        """
        Helper method to compute CDF for a single point using vectorized operations.

        Parameters
        ----------
        u : numpy.ndarray
            1D array of length dim representing a single point.

        Returns
        -------
        float
            CDF value at the point.
        """
        # Short-circuit checks
        if np.any(u <= 0):
            return 0.0
        if np.all(u >= 1):
            return 1.0

        # Create a numpy meshgrid of indices for vectorized computation
        indices = np.meshgrid(*[np.arange(s) for s in self.matr.shape], indexing="ij")

        # Calculate the fraction matrix - vectorized approach
        fraction_matrix = np.ones_like(self.matr, dtype=float)

        for d in range(self.dim):
            # Get indices array for this dimension
            idx_d = indices[d]

            # Calculate lower and upper bounds
            lower_d = idx_d / self.matr.shape[d]
            upper_d = (idx_d + 1) / self.matr.shape[d]

            # Calculate overlap with [0, u[d]]
            overlap_d = np.maximum(0.0, np.minimum(u[d], upper_d) - lower_d)

            # Convert to fraction of cell width
            cell_width_d = 1.0 / self.matr.shape[d]
            frac_d = overlap_d / cell_width_d
            # Clip to ensure fractions are in [0, 1]
            frac_d = np.clip(frac_d, 0.0, 1.0)

            # Multiply into the fraction matrix
            fraction_matrix *= frac_d

        # Multiply by cell masses and sum
        result = np.sum(self.matr * fraction_matrix)

        return float(result)

    def _cdf_vectorized_impl(self, points):
        """
        Implementation of vectorized CDF for multiple points.

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

        # Precompute cell information
        matr_shape = np.array(self.matr.shape)

        # Create a numpy meshgrid of indices for the cells
        indices = np.meshgrid(*[np.arange(s) for s in self.matr.shape], indexing="ij")
        indices = np.array([idx.ravel() for idx in indices]).T  # Shape: (n_cells, dim)

        # Reshape for broadcasting against points
        indices = indices[:, np.newaxis, :]  # Shape: (n_cells, 1, dim)
        compute_points = compute_points[
            np.newaxis, :, :
        ]  # Shape: (1, n_points_to_compute, dim)

        # Calculate cell bounds
        lower_bounds = indices / matr_shape[np.newaxis, np.newaxis, :]
        upper_bounds = (indices + 1) / matr_shape[np.newaxis, np.newaxis, :]

        # Calculate overlap with [0, u] for each dimension and each point
        overlaps = np.maximum(
            0.0, np.minimum(compute_points, upper_bounds) - lower_bounds
        )

        # Convert to fraction of cell width
        cell_widths = 1.0 / matr_shape[np.newaxis, np.newaxis, :]
        fractions = overlaps / cell_widths
        fractions = np.clip(fractions, 0.0, 1.0)

        # Calculate the total fraction for each cell and point
        cell_fractions = np.prod(
            fractions, axis=2
        )  # Shape: (n_cells, n_points_to_compute)

        # Get cell masses
        cell_masses = self.matr.ravel()[:, np.newaxis]  # Shape: (n_cells, 1)

        # Calculate weighted sum for each point
        weighted_fractions = (
            cell_masses * cell_fractions
        )  # Shape: (n_cells, n_points_to_compute)
        point_results = np.sum(
            weighted_fractions, axis=0
        )  # Shape: (n_points_to_compute,)

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
                raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}")

    def cond_distr_1(self, u):
        """F_{U_{-1}|U_1}(u_{-1} | u_1)."""
        return self.cond_distr(1, u)

    def cond_distr_2(self, u):
        """F_{U_{-2}|U_2}(u_{-2} | u_2)."""
        return self.cond_distr(2, u)

    def _cond_distr_single(self, i, u):
        """
        Helper method for conditional distribution of a single point.
        This preserves the original implementation for a single point.

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

        # Clear cached results from previous calls to avoid test interference
        self.intervals = {}

        # Cache key for the slice indices only
        slice_key = (i, i_idx)

        # Check if we've cached the slice indices for this dimension and index
        if slice_key in self.intervals:
            slice_indices = self.intervals[slice_key]
            # Recalculate denominator (the sum of all cells in the slice)
            denom = sum(self.matr[c] for c in slice_indices)
        else:
            # 1) DENOMINATOR: sum of all cells in "slice" c[i0] = i_idx
            denom = 0.0
            slice_indices = []

            # This is more efficient than using itertools.product for the whole space
            # when we're only interested in a specific slice
            indices = [range(s) for s in self.matr.shape]
            indices[i0] = [i_idx]  # Fix the i0 dimension

            for c in itertools.product(*indices):
                cell_mass = self.matr[c]
                denom += cell_mass
                if cell_mass > 0:  # Only track positive mass cells for numerator
                    slice_indices.append(c)

            # Cache the slice indices for future use
            self.intervals[slice_key] = slice_indices

        if denom <= 0:
            return 0.0

        # 2) NUMERATOR: Among that same slice, we see how much is below u[j] in each j != i0
        num = 0.0
        for c in slice_indices:
            cell_mass = self.matr[c]
            fraction = 1.0
            for j in range(self.dim):
                if j == i0:
                    # No partial coverage in the conditioning dimension
                    continue

                # Use exactly the same calculation method as the original
                lower_j = c[j] / self.matr.shape[j]
                upper_j = (c[j] + 1) / self.matr.shape[j]
                val_j = u[j]

                # Overlap with [0, val_j] in this dimension
                if val_j <= 0:
                    fraction = 0.0
                    break
                if val_j >= 1:
                    # entire cell dimension is included
                    continue

                # Calculate exactly as in the original implementation
                overlap_len = max(0.0, min(val_j, upper_j) - lower_j)
                cell_width = 1.0 / self.matr.shape[j]
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

    def _cond_distr_vectorized(self, i, points):
        """
        Vectorized implementation of conditional distribution for multiple points.

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

        # Process each point separately (vectorizing all calculations for a single point)
        # This is more efficient than trying to vectorize across all points at once for cond_distr
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

            # For numerator, we need to compute fractions for each cell in the slice
            # First, create indices for all cells in the slice
            remaining_dims = [d for d in range(self.dim) if d != i0]
            mesh_dims = [range(self.matr.shape[d]) for d in remaining_dims]

            if mesh_dims:  # Only create meshgrid if we have remaining dimensions
                # Create meshgrid for remaining dimensions
                mesh = np.meshgrid(*mesh_dims, indexing="ij")
                indices = np.array(
                    [idx.ravel() for idx in mesh]
                ).T  # Shape: (n_cells_in_slice, len(remaining_dims))

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

                # Calculate fractions for each cell in the slice
                numerator = 0.0
                for c in full_indices:
                    cell_mass = self.matr[c]
                    if cell_mass <= 0:
                        continue

                    fraction = 1.0
                    for j in range(self.dim):
                        if j == i0:
                            continue  # Skip conditioning dimension

                        lower_j = c[j] / self.matr.shape[j]
                        upper_j = (c[j] + 1) / self.matr.shape[j]
                        val_j = u[j]

                        if val_j <= 0:
                            fraction = 0.0
                            break
                        if val_j >= 1:
                            continue

                        overlap_len = max(0.0, min(val_j, upper_j) - lower_j)
                        cell_width = 1.0 / self.matr.shape[j]
                        frac_j = overlap_len / cell_width
                        frac_j = min(1.0, max(0.0, frac_j))

                        fraction *= frac_j
                        if fraction <= 0:
                            break

                    if fraction > 0:
                        numerator += cell_mass * fraction

                results[p_idx] = numerator / denom
            else:
                # Special case: only one dimension
                results[p_idx] = 1.0 if denom > 0 else 0.0

        return results

    def pdf(self, *args):
        """
        Evaluate the piecewise PDF at one or multiple points.

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
        value = copula.pdf(0.3, 0.7)

        # Single point as array
        value = copula.pdf([0.3, 0.7])

        # Multiple points as 2D array
        values = copula.pdf(np.array([[0.1, 0.2], [0.3, 0.4]]))
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
                    return self._pdf_single_point(arr)

                elif arr.ndim == 2:
                    # 2D array - multiple points
                    if arr.shape[1] != self.dim:
                        raise ValueError(
                            f"Expected points with {self.dim} dimensions, got {arr.shape[1]}"
                        )
                    return self._pdf_vectorized(arr)

                else:
                    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D array")

            elif hasattr(arg, "__len__"):
                # List, tuple, or similar sequence
                if len(arg) == self.dim:
                    # Single point as a sequence
                    return self._pdf_single_point(np.array(arg, dtype=float))
                else:
                    raise ValueError(
                        f"Expected point with {self.dim} dimensions, got {len(arg)}"
                    )

            else:
                # Single scalar value - only valid for 1D case
                if self.dim == 1:
                    return self._pdf_single_point(np.array([arg], dtype=float))
                else:
                    raise ValueError(
                        f"Single scalar provided but copula has {self.dim} dimensions"
                    )

        else:
            # Multiple arguments provided
            if len(args) == self.dim:
                # Separate coordinates for a single point
                return self._pdf_single_point(np.array(args, dtype=float))
            else:
                raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}")

    def _pdf_single_point(self, u):
        """
        Helper method to compute PDF for a single point.

        Parameters
        ----------
        u : numpy.ndarray
            1D array of length dim representing a single point.

        Returns
        -------
        float
            PDF value at the point.
        """
        if np.any(u < 0) or np.any(u > 1):
            return 0.0

        # Identify which cell the point falls into
        cell_idx = []
        for k, val in enumerate(u):
            ix = int(np.floor(val * self.matr.shape[k]))
            ix = max(0, min(ix, self.matr.shape[k] - 1))
            cell_idx.append(ix)

        # Return the cell's mass
        return float(self.matr[tuple(cell_idx)])

    def _pdf_vectorized(self, points):
        """
        Vectorized implementation of PDF for multiple points.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (n_points, dim) where each row is a point.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_points,) with PDF values.
        """
        # Convert to numpy array
        points = np.asarray(points, dtype=float)

        n_points = points.shape[0]
        results = np.zeros(n_points)

        # Filter out points outside [0,1]^d
        valid_mask = np.all((points >= 0) & (points <= 1), axis=1)

        if not np.any(valid_mask):
            return results  # All points are outside [0,1]^d

        # Process only valid points
        valid_points = points[valid_mask]

        # Calculate cell indices for each point - this is the key vectorized operation
        indices = np.floor(valid_points * np.array(self.matr.shape)).astype(int)

        # Clip indices to valid ranges
        for d in range(self.dim):
            indices[:, d] = np.clip(indices[:, d], 0, self.matr.shape[d] - 1)

        # Get PDF values using advanced indexing
        for i, idx in enumerate(indices):
            results[valid_mask][i] = self.matr[tuple(idx)]

        return results

    def rvs(self, n=1, random_state=None, **kwargs):
        """
        Draw random variates from the d-dimensional checkerboard copula efficiently.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        random_state : int, optional
            Seed for random number generator.

        Returns
        -------
        np.ndarray
            Array of shape (n, d) containing n samples in d dimensions.
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Flatten the matrix and create probability distribution
        flat_matrix = np.asarray(self.matr, dtype=float).ravel()
        total = flat_matrix.sum()

        if total <= 0:
            raise ValueError("Matrix contains no positive values, cannot sample")

        probs = flat_matrix / total

        # Sample flat indices according to cell probabilities
        flat_indices = np.random.choice(np.arange(len(probs)), size=n, p=probs)

        # Convert flat indices to multi-indices as a (d, n) array
        indices_arrays = np.unravel_index(flat_indices, self.matr.shape)

        # Transform indices to a (n, d) array
        indices = np.column_stack(indices_arrays)

        # Generate uniform jitter for each dimension
        jitter = np.random.rand(n, self.dim)

        # Combine indices and jitter to get final coordinates
        return (indices + jitter) / np.array(self.matr.shape)

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
