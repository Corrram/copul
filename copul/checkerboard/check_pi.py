import numpy as np

from copul.checkerboard.check import Check
from copul.family.core.copula_approximator_mixin import CopulaApproximatorMixin
from copul.family.core.copula_plotting_mixin import CopulaPlottingMixin


class CheckPi(Check, CopulaPlottingMixin, CopulaApproximatorMixin):
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

                    bcp_module = importlib.import_module("copul.checkerboard.biv_check_pi")
                    BivCheckPi = bcp_module.BivCheckPi
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

    @property
    def is_symmetric(self) -> bool:
        return np.allclose(self.matr, self.matr.T)

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
        """CDF at a single point (1D array of length ``dim``)."""
        return float(self._cdf_vectorized_impl(np.asarray(u, dtype=float)[None, :])[0])

    def _cdf_vectorized_impl(self, points):
        """
        Vectorised CDF for an ``(n_points, dim)`` array.

        Uses the separable structure ``C(u) = sum_c matr[c] prod_d F_d(u_d, c_d)``
        with the per-axis cell fractions ``F_d``; cost ``O(N * #cells)`` with
        bounded memory.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if self.dim == 2:
            from copul.checkerboard import _biv_engine as eng

            return eng.cdf(self.matr, None, points[:, 0], points[:, 1])
        factors = [self._axis_fractions(points[:, d], d) for d in range(self.dim)]
        return self._contract(factors)

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

    def cond_distr_1(self, u):
        """F_{U_{-1}|U_1}(u_{-1} | u_1)."""
        return self.cond_distr(1, u)

    def cond_distr_2(self, u):
        """F_{U_{-2}|U_2}(u_{-2} | u_2)."""
        return self.cond_distr(2, u)

    def _cond_distr_single(self, i, u):
        """Conditional distribution at a single point."""
        pts = np.asarray(u, dtype=float)[None, :]
        return float(self._cond_distr_vectorized(i, pts)[0])

    def _cond_distr_vectorized(self, i, points):
        """
        Vectorised conditional distribution for an ``(n_points, dim)`` array.

        For conditioning axis ``i0 = i - 1`` the slice ``c[i0] = floor(u_i0 * m_i0)``
        is selected; the result is the slice mass below ``u`` in the other
        coordinates (partial cell fractions) divided by the slice mass.
        """
        points = np.asarray(points, dtype=float)
        i0 = i - 1
        k = self.matr.shape[i0]
        x = points[:, i0]
        idx = np.minimum(np.floor(np.clip(x, 0.0, 1.0) * k).astype(np.intp), k - 1)
        onehot = np.zeros((points.shape[0], k))
        onehot[np.arange(points.shape[0]), idx] = 1.0
        factors = [
            onehot if d == i0 else self._axis_fractions(points[:, d], d) for d in range(self.dim)
        ]
        num = self._contract(factors)
        axes = tuple(d for d in range(self.dim) if d != i0)
        denom = self.matr.sum(axis=axes)[idx] if axes else self.matr[idx]
        out = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        return np.where(x < 0, 0.0, out)

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
                    raise ValueError(f"Expected point with {self.dim} dimensions, got {len(arg)}")

            else:
                # Single scalar value - only valid for 1D case
                if self.dim == 1:
                    return self._pdf_single_point(np.array([arg], dtype=float))
                else:
                    raise ValueError(f"Single scalar provided but copula has {self.dim} dimensions")

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
        return float(self.matr[tuple(cell_idx)]) * np.prod(self.matr.shape)

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
        points = np.asarray(points, dtype=float)
        shape = np.array(self.matr.shape)
        valid = np.all((points >= 0) & (points <= 1), axis=1)
        idx = np.floor(points * shape).astype(np.intp)
        idx = np.clip(idx, 0, shape - 1)
        vals = self.matr[tuple(idx.T)]
        return np.where(valid, vals, 0.0) * np.prod(shape)

    def rvs(self, n=1, random_state=None, **kwargs):
        """
        Draw random variates from the d-dimensional checkerboard copula.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        random_state : int, numpy Generator or None, optional
            Source of randomness.  ``None`` uses NumPy's global generator
            (never reseeded).

        Returns
        -------
        np.ndarray
            Array of shape (n, d) containing n samples in d dimensions.
        """
        from copul.checkerboard._biv_engine import resolve_rng

        rng = resolve_rng(random_state)
        flat_matrix = np.asarray(self.matr, dtype=float).ravel()
        total = flat_matrix.sum()
        if total <= 0:
            raise ValueError("Matrix contains no positive values, cannot sample")
        flat_indices = rng.choice(flat_matrix.size, size=int(n), p=flat_matrix / total)
        indices = np.column_stack(np.unravel_index(flat_indices, self.matr.shape))
        jitter = rng.random((int(n), self.dim))
        return (indices + jitter) / np.array(self.matr.shape)

    @staticmethod
    def _weighted_random_selection(matrix, num_samples, random_state=None):
        """
        Select elements from 'matrix' with probability proportional to matrix entries.
        Return (selected_values, selected_multi_indices).
        """
        from copul.checkerboard._biv_engine import resolve_rng

        rng = resolve_rng(random_state)
        matrix = np.asarray(matrix)
        arr = np.asarray(matrix, dtype=float).ravel()
        flat_indices = rng.choice(arr.size, size=int(num_samples), p=arr / arr.sum())
        idx_arrays = np.unravel_index(flat_indices, matrix.shape)
        multi_idx = list(zip(*idx_arrays))
        return matrix[idx_arrays], multi_idx

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
