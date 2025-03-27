import itertools
import warnings

import numpy as np
from scipy.special import comb
from scipy.integrate import quad

# If your base Check class lives elsewhere, adjust the import accordingly:
from copul.checkerboard.check import Check


class BernsteinCopula(Check):
    """
    Represents a d-dimensional Bernstein Copula with potentially different
    degrees m_i along each dimension i.

    The coefficient tensor `theta` has shape (m1 + 1, m2 + 1, ..., md + 1),
    where m_i is the degree along dimension i. Thus `theta` has `d` dimensions,
    and dimension i must have size (m_i + 1).

    Parameters
    ----------
    theta : array-like
        Coefficient tensor of shape (m1+1, m2+1, ..., md+1).
    check_theta : bool, optional
        If True (default), perform shape checks on the `theta` tensor.
        More advanced validity checks (related to non-negativity, etc.)
        are not implemented here.

    Attributes
    ----------
    theta : np.ndarray
        The coefficient tensor.
    degrees : list of int
        The list `[m1, m2, ..., md]` of polynomial degrees.
    dim : int
        The dimension `d` of the copula.
    """

    def __new__(cls, theta, *args, **kwargs):
        """
        Create a new BernsteinCopula instance or a BivBernsteinCopula instance
        if dimension is 2 and that specialized class is available.
        """
        if cls is BernsteinCopula:
            theta_arr = np.asarray(theta)
            if theta_arr.ndim == 2:
                # Attempt to import specialized BivBernsteinCopula
                try:
                    import importlib
                    bbc_module = importlib.import_module(
                        "copul.bernstein.biv_bernstein"  # or your correct path
                    )
                    BivBernsteinCopula = getattr(bbc_module, "BivBernsteinCopula")
                    return BivBernsteinCopula(theta, *args, **kwargs)
                except (ImportError, ModuleNotFoundError, AttributeError) as e:
                    warnings.warn(
                        f"Could not import BivBernsteinCopula, falling back "
                        f"to generic BernsteinCopula for 2D case. Error: {e}"
                    )
                    # If import fails, continue with normal instantiation
        return super().__new__(cls)

    def __init__(self, theta, check_theta=True):
        """
        Initialize the Bernstein Copula with potentially different degrees.
        """
        self.theta = np.asarray(theta, dtype=float)
        self.dim = self.theta.ndim
        if self.dim == 0:
            raise ValueError("Theta must have at least one dimension.")

        # For each dimension i, the size is (m_i + 1)
        shape = self.theta.shape
        self.degrees = [s - 1 for s in shape]
        if any(d < 0 for d in self.degrees):
            raise ValueError("Each dimension must have size >= 1.")

        # Optional shape check: not all shapes must match, so we skip the old check.
        if check_theta:
            # (You could add checks for non-negativity or boundary conditions here)
            pass

        # Precompute binomial coefficients for the CDF
        # For dimension i, we need comb(m_i, k) for k=0..m_i
        self._binom_coeffs_cdf = [
            comb(m_i, np.arange(m_i + 1), exact=True) if m_i >= 0 else np.array([1.0])
            for m_i in self.degrees
        ]

        # Precompute binomial coefficients for the PDF
        # For dimension i, we need comb(m_i - 1, k) for k=0..(m_i - 1) if m_i>0
        self._binom_coeffs_pdf = [
            comb(m_i - 1, np.arange(m_i), exact=True) if m_i > 0 else np.array([1.0])
            for m_i in self.degrees
        ]

        # Compute the forward finite differences (one difference per dimension)
        self._delta_theta = self._compute_finite_differences(self.theta)  # shape: (m1, m2, ..., md)

        # Inherit from Check, storing theta as 'matr' for compatibility
        super().__init__(matr=self.theta)

    def __str__(self):
        return f"BernsteinCopula(degrees={self.degrees}, dim={self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    # --------------------------------------------------------------------------
    #                           Helper Functions
    # --------------------------------------------------------------------------

    @staticmethod
    def _bernstein_poly(m, k, u, binom_coeff=None):
        """Compute the scalar Bernstein basis polynomial P_{m,k}(u)."""
        if k < 0 or k > m:
            return 0.0
        if binom_coeff is None:
            binom_coeff = comb(m, k, exact=True)

        if u <= 0.0:
            return binom_coeff if k == 0 else 0.0
        if u >= 1.0:
            return binom_coeff if k == m else 0.0

        return binom_coeff * (u ** k) * ((1.0 - u) ** (m - k))

    def _bernstein_poly_vec(self, m, k_vals, u, binom_coeffs):
        """
        Vectorized Bernstein polynomial for multiple k values at a single scalar u.
        Returns an array of same length as k_vals.
        """
        u = float(u)  # ensure scalar
        res = np.zeros_like(k_vals, dtype=float)

        # Quick boundary checks
        if u <= 0.0:
            # nonzero only for k=0
            mask_k0 = (k_vals == 0)
            res[mask_k0] = binom_coeffs[0]
            return res
        if u >= 1.0:
            # nonzero only for k=m
            mask_km = (k_vals == m)
            if m >= 0 and len(binom_coeffs) > m:
                res[mask_km] = binom_coeffs[m]
            return res

        # General interior case
        # For each k in k_vals, B_{m,k}(u) = comb(m,k)*u^k*(1-u)^(m-k)
        # but we have binom_coeffs array matching k_vals index
        # Typically binom_coeffs[k] = comb(m,k).
        res = binom_coeffs[k_vals] * (u ** k_vals) * ((1 - u) ** (m - k_vals))
        return res

    def _compute_finite_differences(self, arr):
        """
        Compute the d-dimensional forward finite difference by
        differencing once along each axis in turn.
        After each difference along axis i, that dimension's size reduces by 1.
        """
        diff_arr = np.array(arr, copy=True)
        for axis in range(self.dim):
            diff_arr = np.diff(diff_arr, n=1, axis=axis)
        return diff_arr  # shape: (m1, m2, ..., md)

    # --------------------------------------------------------------------------
    #                               CDF
    # --------------------------------------------------------------------------
    def cdf(self, *args):
        """
        Compute the CDF at one or multiple points.
        This is the same interface as in the base CheckPi, allowing:
          - cdf(x) if dim=1
          - cdf([x1, x2, ..., x_d]) if dim=d
          - cdf([[...], [...], ...]) for multiple points, shape (n, d)
        """
        # Input handling logic (similar to CheckPi).
        if len(args) == 0:
            raise ValueError("No arguments provided to cdf().")
        elif len(args) == 1:
            arg = args[0]
            arr = np.asarray(arg, dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._cdf_single_point(arr)
                else:
                    raise ValueError(
                        f"Single 1D array must match dimension {self.dim}, got shape {arr.shape}."
                    )
            elif arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError(
                        f"Second dimension must match copula dimension {self.dim}, got {arr.shape[1]}."
                    )
                return self._cdf_vectorized_impl(arr)
            else:
                # Single scalar if dim=1?
                if self.dim == 1 and arr.size == 1:
                    return self._cdf_single_point(arr.ravel())
                raise ValueError(
                    f"cdf() only supports 1D or 2D arrays, got {arr.ndim}D."
                )
        else:
            # multiple separate coords cdf(u1, u2, ..., ud)
            if len(args) == self.dim:
                return self._cdf_single_point(np.array(args, dtype=float))
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}."
                )

    def _cdf_single_point(self, u):
        """Compute CDF at a single d-dimensional point u."""
        if np.any(u < 0.0) or np.any(u > 1.0):
            # For points strictly out of [0,1], typical Bernstein copula is 0 or 1
            # Here, we enforce a strict domain check.
            raise ValueError("All coordinates must be in [0,1].")

        # Handle boundary quickly
        if np.any(u <= 0):
            return 0.0
        if np.all(u >= 1):
            return 1.0

        # Precompute the Bernstein polynomials in each dimension
        bern_vals = []
        for j in range(self.dim):
            m_j = self.degrees[j]
            k_arr = np.arange(m_j + 1)
            # comb array is self._binom_coeffs_cdf[j]
            bv = self._bernstein_poly_vec(m_j, k_arr, u[j], self._binom_coeffs_cdf[j])
            bern_vals.append(bv)

        # Accumulate sum over all multi-indices in [0..m1] x [0..m2] x ... x [0..md]
        total = 0.0
        shape = self.theta.shape  # (m1+1, m2+1, ..., md+1)
        theta_flat = self.theta.ravel(order='C')

        # product(...) over ranges:
        for k_tuple in itertools.product(*(range(m_j + 1) for m_j in self.degrees)):
            # map k_tuple -> flattened index
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            coef = theta_flat[flat_idx]
            # multiply the 1D bernstein contributions
            bprod = 1.0
            for j, kj in enumerate(k_tuple):
                bprod *= bern_vals[j][kj]
            total += coef * bprod

        return float(np.clip(total, 0.0, 1.0))

    def _cdf_vectorized_impl(self, points):
        """Vectorized CDF for multiple d-dimensional points."""
        n_points = points.shape[0]
        results = np.zeros(n_points, dtype=float)

        # Check domain, we will raise error if any out of [0,1]
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError("All coordinates must be in [0,1].")

        # Boundary mask
        all_zero_or_less = np.all(points <= 0, axis=1)
        all_one_or_more = np.all(points >= 1, axis=1)
        results[all_zero_or_less] = 0.0
        results[all_one_or_more] = 1.0

        # Points strictly inside (0,1)
        compute_mask = ~(all_zero_or_less | all_one_or_more)
        pts_in = points[compute_mask]
        n_in = pts_in.shape[0]
        if n_in == 0:
            return results

        # Precompute Bernstein polynomials for each dimension
        # We'll store a list of arrays [dim], each array shape = (n_in, m_i+1)
        bern_vals = []
        for j in range(self.dim):
            m_j = self.degrees[j]
            k_arr = np.arange(m_j + 1)
            binom_arr = self._binom_coeffs_cdf[j]
            bv_j = np.zeros((n_in, m_j + 1), dtype=float)
            for i in range(n_in):
                bv_j[i, :] = self._bernstein_poly_vec(m_j, k_arr, pts_in[i, j], binom_arr)
            bern_vals.append(bv_j)

        # Accumulate
        shape = self.theta.shape
        theta_flat = self.theta.ravel(order='C')
        cdf_in = np.zeros(n_in, dtype=float)

        for k_tuple in itertools.product(*(range(m_j + 1) for m_j in self.degrees)):
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            coef = theta_flat[flat_idx]
            # multiply dimension contributions: shape = (n_in,)
            # i.e. prod over j: bern_vals[j][:, k_tuple[j]]
            factor = np.ones(n_in, dtype=float)
            for j, kj in enumerate(k_tuple):
                factor *= bern_vals[j][:, kj]
            cdf_in += coef * factor

        # Clip and assign
        cdf_in = np.clip(cdf_in, 0.0, 1.0)
        results[compute_mask] = cdf_in
        return results

    # --------------------------------------------------------------------------
    #                               PDF
    # --------------------------------------------------------------------------
    def pdf(self, *args):
        """
        Evaluate the PDF at one or multiple points.
        Similar input interface as `cdf()`.
        """
        if len(args) == 0:
            raise ValueError("No arguments provided to pdf().")
        elif len(args) == 1:
            arg = args[0]
            arr = np.asarray(arg, dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._pdf_single_point(arr)
                else:
                    raise ValueError(
                        f"Single 1D array must match dimension {self.dim}, got {arr.shape}."
                    )
            elif arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError(
                        f"Second dimension must match copula dimension {self.dim}, got {arr.shape[1]}."
                    )
                return self._pdf_vectorized(arr)
            else:
                # Single scalar if dim=1?
                if self.dim == 1 and arr.size == 1:
                    return self._pdf_single_point(arr.ravel())
                raise ValueError(
                    f"pdf() only supports 1D or 2D arrays, got {arr.ndim}D."
                )
        else:
            if len(args) == self.dim:
                return self._pdf_single_point(np.array(args, dtype=float))
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}."
                )

    def _pdf_single_point(self, u):
        """Compute PDF for a single point u."""
        # If any coordinate is outside (0,1), PDF=0 for typical Bernsteins
        if np.any(u <= 0.0) or np.any(u >= 1.0):
            return 0.0

        # If any dimension has m_i=0, then that dimension is effectively "C(u)=u",
        # so the overall PDF is still 1 if all dims are m_i=0. If some are > 0, we keep going.
        # But let's handle the full logic with _delta_theta:
        # c(u) = sum_{k1=0..m1-1} ... sum_{kd=0..md-1} Delta_theta(...) * ∏_j B_{m_j-1, k_j}(u_j),
        # multiplied by ∏_j (m_j).
        # If all m_j=0, that implies shape == (1,1,...), then self._delta_theta is empty,
        # but effectively the copula is the identity if it is "valid" with coefficient=1.
        # We'll handle that below.

        # Check if all degrees == 0 => "constant" copula = Independence in 1D => PDF=1
        if all(m == 0 for m in self.degrees):
            # Typically that means theta[...] = 1. Then PDF=1.0
            return 1.0 if np.allclose(self.theta, 1.0) else 0.0

        # Precompute B_{m_j - 1, k} for each dimension j
        bern_vals = []
        for j in range(self.dim):
            mj = self.degrees[j]
            if mj > 0:
                k_arr = np.arange(mj)
                bv = self._bernstein_poly_vec(
                    mj - 1, k_arr, u[j], self._binom_coeffs_pdf[j]
                )
            else:
                # dimension with m_j=0 => B_{-1,k} is not well-defined. But effectively it's 1.
                bv = np.array([1.0])  # shape(1,)
            bern_vals.append(bv)

        total_pdf = 0.0
        shape = self._delta_theta.shape  # (m1, m2, ..., md)
        delta_flat = self._delta_theta.ravel(order='C')

        # Loop over k in [0..m_j-1]
        for k_tuple in itertools.product(*(range(max(mj, 0)) for mj in self.degrees)):
            # flatten index
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            dtheta = delta_flat[flat_idx]
            # product of B_{m_j-1, k_j}(u_j)
            bprod = 1.0
            for j, kj in enumerate(k_tuple):
                bprod *= bern_vals[j][kj]  # shape-matching
            total_pdf += dtheta * bprod

        # multiply by ∏(m_j)
        mprod = 1
        for mj in self.degrees:
            if mj > 0:
                mprod *= mj

        pdf_val = total_pdf * mprod
        # Clip for safety
        return max(0.0, pdf_val)

    def _pdf_vectorized(self, points):
        """Vectorized PDF for multiple points."""
        n_points = points.shape[0]
        results = np.zeros(n_points, dtype=float)

        # Outside (0,1) => PDF=0
        outside_mask = np.any((points <= 0.0) | (points >= 1.0), axis=1)
        results[outside_mask] = 0.0

        # The rest
        inside_mask = ~outside_mask
        pts_in = points[inside_mask]
        n_in = pts_in.shape[0]
        if n_in == 0:
            return results

        # If all degrees == 0 => PDF=1 if theta=1
        if all(m == 0 for m in self.degrees):
            # If it's a valid "all-0-degree" copula, presumably theta=1 => PDF=1
            # If not, then PDF=0. We'll check the top-left entry or the entire array:
            val = 1.0 if np.allclose(self.theta, 1.0) else 0.0
            results[inside_mask] = val
            return results

        # Precompute B_{m_j-1, k} for each dimension j, shape => (n_in, m_j)
        bern_vals = []
        for j in range(self.dim):
            mj = self.degrees[j]
            if mj > 0:
                k_arr = np.arange(mj)
                bc = self._binom_coeffs_pdf[j]
                tmp = np.zeros((n_in, mj), dtype=float)
                for i in range(n_in):
                    tmp[i, :] = self._bernstein_poly_vec(mj - 1, k_arr, pts_in[i, j], bc)
            else:
                # dimension with m_j=0 => effectively just 1
                tmp = np.ones((n_in, 1), dtype=float)
            bern_vals.append(tmp)

        shape = self._delta_theta.shape  # (m1, m2, ..., md)
        delta_flat = self._delta_theta.ravel(order='C')
        pdf_in = np.zeros(n_in, dtype=float)

        # accumulate sums
        for k_tuple in itertools.product(*(range(max(mj, 0)) for mj in self.degrees)):
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            dtheta = delta_flat[flat_idx]

            factor = np.ones(n_in, dtype=float)
            for j, kj in enumerate(k_tuple):
                factor *= bern_vals[j][:, kj]
            pdf_in += dtheta * factor

        # multiply by product of m_j
        mprod = 1
        for mj in self.degrees:
            if mj > 0:
                mprod *= mj

        pdf_in *= mprod
        pdf_in = np.maximum(pdf_in, 0.0)
        results[inside_mask] = pdf_in
        return results

    # --------------------------------------------------------------------------
    #                     Conditional Distribution
    # --------------------------------------------------------------------------
    def cond_distr(self, i, *args):
        """
        Compute C_{i|(-i)}(u_i | u_{-i}) numerically by:
            Integral_0^{u_i} c(u_1, ..., x, ..., u_d) dx
            ---------------------------------------------
            Integral_0^1    c(u_1, ..., x, ..., u_d) dx
        using `scipy.integrate.quad`.

        For multiple points, we loop over them. This can be slow if many points
        are requested.
        """
        if not (1 <= i <= self.dim):
            raise ValueError(f"i must be in [1, {self.dim}]")
        i0 = i - 1

        def integrand(x, fixed_minus_i):
            # build the full point
            full_point = np.insert(fixed_minus_i, i0, x)
            return self._pdf_single_point(full_point)

        # parse input
        if len(args) == 0:
            raise ValueError("No arguments provided to cond_distr().")
        elif len(args) == 1:
            arr = np.asarray(args[0], dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._cond_distr_single(i0, arr, integrand)
                else:
                    raise ValueError(
                        f"Expected array of length {self.dim}, got {arr.size}."
                    )
            elif arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError(
                        f"Expected points with dimension {self.dim}, got {arr.shape[1]}."
                    )
                out = []
                for row in arr:
                    out.append(self._cond_distr_single(i0, row, integrand))
                return np.array(out)
            else:
                raise ValueError("cond_distr() input must be 1D or 2D array.")
        else:
            # treat them as separate coords
            if len(args) == self.dim:
                point = np.array(args, dtype=float)
                return self._cond_distr_single(i0, point, integrand)
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}."
                )

    def _cond_distr_single(self, i0, u, integrand_func):
        """Compute the conditional distribution for a single point."""
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("Coordinates must be in [0,1].")

        ui = u[i0]
        if ui <= 0.0:
            return 0.0
        if ui >= 1.0:
            return 1.0

        u_minus_i = np.delete(u, i0)

        # denominator: integral from 0..1
        try:
            denom, denom_err = quad(integrand_func, 0, 1, args=(u_minus_i,),
                                    limit=100, epsabs=1e-9, epsrel=1e-9)
        except Exception as e:
            warnings.warn(
                f"Integration failed for denominator at point {u}: {e}."
            )
            return np.nan

        if denom < 1e-14:
            # near-zero denominator => undefined
            warnings.warn(
                f"Denominator ~0 for conditional distribution at point {u}."
            )
            return np.nan

        # numerator: integral from 0..ui
        try:
            num, num_err = quad(integrand_func, 0, ui, args=(u_minus_i,),
                                limit=100, epsabs=1e-9, epsrel=1e-9)
        except Exception as e:
            warnings.warn(
                f"Integration failed for numerator at point {u}: {e}."
            )
            return np.nan

        # final ratio
        val = num / denom
        return float(np.clip(val, 0.0, 1.0))

    # Helpers for convenience
    def cond_distr_1(self, u):
        return self.cond_distr(1, u)

    def cond_distr_2(self, u):
        if self.dim < 2:
            raise ValueError("cond_distr_2 requires dim >= 2.")
        return self.cond_distr(2, u)

    # --------------------------------------------------------------------------
    #                          Random Variates
    # --------------------------------------------------------------------------
    def rvs(self, n=1, random_state=None, **kwargs):
        """
        Draw random samples from this Bernstein copula.
        For d=1: Uniform(0,1).
        For d=2: A specialized algorithm might exist in BivBernsteinCopula.
        For d>2: Not implemented.
        """
        if self.dim > 2:
            raise NotImplementedError(
                "Sampling for d>2 is not implemented here."
            )
        elif self.dim == 1:
            rng = np.random.default_rng(random_state)
            return rng.uniform(0, 1, size=(n, 1))
        else:  # dim=2
            warnings.warn(
                "Sampling in 2D is typically handled by BivBernsteinCopula. "
                "Use that class directly for an efficient method."
            )
            raise NotImplementedError("2D sampling not implemented in generic class.")
