import itertools
import warnings

import numpy as np
from scipy.special import comb
from scipy.integrate import quad

# If your base Check class lives elsewhere, adjust the import accordingly:
from copul.checkerboard.check import Check
from copul.families.core.copula_plotting_mixin import CopulaPlottingMixin


class BernsteinCopula(Check, CopulaPlottingMixin):
    """
    Represents a d-dimensional Bernstein Copula with potentially different
    degrees m_i along each dimension i.

    NOTE: This version uses cumsum logic in the CDF and skips k=0,
    matching the updated _cdf_single_point method.
    """

    def __new__(cls, theta, *args, **kwargs):
        if cls is BernsteinCopula:
            theta_arr = np.asarray(theta)
            if theta_arr.ndim == 2:
                # Attempt specialized BivBernsteinCopula
                try:
                    import importlib
                    bbc_module = importlib.import_module(
                        "copul.checkerboard.biv_bernstein"
                    )
                    BivBernsteinCopula = getattr(bbc_module, "BivBernsteinCopula")
                    return BivBernsteinCopula(theta, *args, **kwargs)
                except (ImportError, ModuleNotFoundError, AttributeError) as e:
                    warnings.warn(
                        f"Could not import BivBernsteinCopula, "
                        f"falling back to generic BernsteinCopula. Error: {e}"
                    )
        return super().__new__(cls)

    def __init__(self, theta, check_theta=True):
        theta = np.asarray(theta, dtype=float)
        # Normalize so sum of theta is 1
        total_mass = np.sum(theta)
        if total_mass > 0:
            theta /= total_mass

        self.theta = theta
        self.dim = self.theta.ndim
        if self.dim == 0:
            raise ValueError("Theta must have at least one dimension.")

        # For each dimension i, the size is (m_i + 1)
        shape = self.theta.shape
        self.degrees = [s - 1 for s in shape]
        if any(d < 0 for d in self.degrees):
            raise ValueError("Each dimension must have size >= 1.")

        if check_theta:
            pass  # e.g. check for negativity, etc.

        # Precompute binomial coefficients for CDF & PDF
        self._binom_coeffs_cdf = [
            np.array([comb(m_i, k, exact=True) for k in range(m_i + 1)])
            for m_i in self.degrees
        ]
        self._binom_coeffs_pdf = [
            np.array([comb(m_i - 1, k, exact=True) for k in range(m_i)])
            if m_i > 0 else np.array([1.0])
            for m_i in self.degrees
        ]

        # Forward finite differences
        self._delta_theta = self._compute_finite_differences(self.theta)

        # Let base Check store 'matr'
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
        """Same as before, but skipping boundary inside 0<u<1 handled outside."""
        if k < 0 or k > m:
            return 0.0
        if binom_coeff is None:
            binom_coeff = comb(m, k, exact=True)
        return binom_coeff * (u ** k) * ((1 - u) ** (m - k))

    def _bernstein_poly_vec(self, m, k_vals, u, binom_coeffs):
        """Compute vector of Bernstein polynomials for k in k_vals, skipping boundary."""
        return binom_coeffs[k_vals] * (u ** k_vals) * ((1 - u) ** (m - k_vals))
    
    def _bernstein_poly_vec_cd(self, m, k_vals, u, binom_coeffs):
        """Compute vector of Bernstein polynomials for k in k_vals, skipping boundary."""
        sum1 = binom_coeffs[k_vals] * k_vals*( u ** (k_vals-1)) * ((1 - u) ** (m - k_vals))
        if k_vals[-1] == m:
            k_vals = k_vals[:-1]
        sum2 = binom_coeffs[k_vals] * (m-k_vals) * ( u ** (k_vals)) * ((1 - u) ** (m - k_vals-1))
        return sum1 + sum2

    def _compute_finite_differences(self, arr):
        diff_arr = np.array(arr, copy=True)
        for axis in range(self.dim):
            diff_arr = np.diff(diff_arr, n=1, axis=axis)
        return diff_arr

    # --------------------------------------------------------------------------
    #                               CDF
    # --------------------------------------------------------------------------
    def cdf(self, *args):
        if len(args) == 0:
            raise ValueError("No arguments provided to cdf().")

        if len(args) == 1:
            arr = np.asarray(args[0], dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._cdf_single_point(arr)
                else:
                    raise ValueError(
                        f"Single 1D array must match dimension {self.dim}, "
                        f"got shape {arr.shape}."
                    )
            elif arr.ndim == 2:
                # shape (n, d)
                if arr.shape[1] != self.dim:
                    raise ValueError(
                        f"Second dimension must match copula dimension {self.dim}, "
                        f"got {arr.shape[1]}."
                    )
                return self._cdf_vectorized_impl(arr)
            else:
                raise ValueError("cdf() only supports 1D or 2D arrays.")
        else:
            # cdf(u1,u2,...,ud)
            if len(args) == self.dim:
                return self._cdf_single_point(np.array(args))
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}."
                )

    def _cdf_single_point(self, u):
        # boundary checks
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("All coordinates must be in [0,1].")
        if np.any(u == 0):
            return 0.0
        if np.all(u == 1):
            return 1.0

        # "Fixed" logic: skip k=0, cumsum of theta
        # We only do the 2D cumsum if dim=2. For higher dims, we do
        # repeated cumsum across each axis. (If you only do 2D, adapt as needed.)
        # Example for any dim:
        theta_cs = self.theta.copy()
        for ax in range(self.dim):
            theta_cs = np.cumsum(theta_cs, axis=ax)

        # Precompute the 1..m_j bernstein polynomials
        bern_vals = []
        for j in range(self.dim):
            m_j = self.degrees[j]
            # skip k=0 => range(1..m_j)
            k_arr = np.arange(1, m_j + 1)
            bc = self._binom_coeffs_cdf[j]
            bv = self._bernstein_poly_vec(m_j, k_arr, u[j], bc)
            bern_vals.append(bv)

        # accumulate
        shape = theta_cs.shape
        theta_flat = theta_cs.ravel(order='C')
        total = 0.0

        for k_tuple in itertools.product(*(range(1, m_j + 1) for m_j in self.degrees)):
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            coef = theta_flat[flat_idx]

            bprod = 1.0
            for j, kj in enumerate(k_tuple):
                # indexing for bern_vals[j]: we used k_arr=1..m_j => index=kj-1
                bprod *= bern_vals[j][kj - 1]
            total += coef * bprod

        return float(np.clip(total, 0, 1))

    def _cdf_vectorized_impl(self, points):
        n_points = points.shape[0]
        results = np.zeros(n_points, dtype=float)

        # boundary
        if np.any(points < 0) or np.any(points > 1):
            raise ValueError("All coordinates must be in [0,1].")

        # any row that has a 0 => cdf=0, all=1 => cdf=1
        any_zero = np.any(points == 0, axis=1)
        all_one = np.all(points == 1, axis=1)
        results[any_zero] = 0.0
        results[all_one] = 1.0

        inside = ~(any_zero | all_one)
        pts_in = points[inside]
        n_in = len(pts_in)
        if n_in == 0:
            return results

        # cumsum of self.theta across each axis
        theta_cs = self.theta.copy()
        for ax in range(self.dim):
            theta_cs = np.cumsum(theta_cs, axis=ax)

        # Precompute bernstein polynomials for each dimension
        bern_vals = []
        for j in range(self.dim):
            m_j = self.degrees[j]
            k_arr = np.arange(1, m_j + 1)
            bc = self._binom_coeffs_cdf[j]
            # shape => (n_in, m_j)
            bvals_j = np.zeros((n_in, m_j), dtype=float)
            for i in range(n_in):
                bvals_j[i,:] = self._bernstein_poly_vec(m_j, k_arr, pts_in[i,j], bc)
            bern_vals.append(bvals_j)

        # accumulate
        shape = theta_cs.shape
        theta_flat = theta_cs.ravel(order='C')
        cdf_in = np.zeros(n_in, dtype=float)

        for k_tuple in itertools.product(*(range(1, m_j + 1) for m_j in self.degrees)):
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            coef = theta_flat[flat_idx]

            factor = np.ones(n_in, dtype=float)
            for j, kj in enumerate(k_tuple):
                factor *= bern_vals[j][:, kj - 1]
            cdf_in += coef * factor

        cdf_in = np.clip(cdf_in, 0.0, 1.0)
        results[inside] = cdf_in
        return results

    # --------------------------------------------------------------------------
    #                               PDF
    # --------------------------------------------------------------------------
    def pdf(self, *args):
        if len(args) == 0:
            raise ValueError("No arguments provided to pdf().")
        elif len(args) == 1:
            arr = np.asarray(args[0], dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._pdf_single_point(arr)
                else:
                    raise ValueError("Wrong shape for 1D input.")
            elif arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError("Second dimension must match copula dim.")
                return self._pdf_vectorized(arr)
            else:
                raise ValueError("pdf() only supports 1D or 2D arrays.")
        else:
            if len(args) == self.dim:
                return self._pdf_single_point(np.array(args))
            else:
                raise ValueError(
                    f"Expected {self.dim} coordinates, got {len(args)}."
                )

    def _pdf_single_point(self, u):
        # boundary => 0
        if np.any(u <= 0) or np.any(u >= 1):
            return 0.0

        # check if all deg=0 => trivial
        if all(d==0 for d in self.degrees):
            return 1.0 if np.allclose(self.theta,1.0) else 0.0

        # This version parallels the cumsum approach, skipping k=0
        # We'll do a forward-difference approach on cumsum-theta.
        # For 2D only? Or general dims. We'll cumsum across each axis again:
        theta_cs = self._delta_theta.copy()
        # Typically, you do difference, then cumsum? We'll just do difference once.
        # If your tests want cumsum approach, adapt similarly.

        # We skip k=0 => range(1..m_j)
        bern_vals = []
        for j in range(self.dim):
            mj = self.degrees[j]
            # PDF uses m_j-1 in standard, but we'll keep it consistent:
            if mj>0:
                k_arr = np.arange(1,mj)  # skip k=0 => 1..(m_j-1)
                if len(k_arr)==0:
                    # means m_j=1 => no interior PDF except boundary
                    return 0.0
                bc = self._binom_coeffs_pdf[j]
                bv = self._bernstein_poly_vec(mj-1, k_arr, u[j], bc)
            else:
                # deg=0 => 1
                bv = np.array([1.0])
            bern_vals.append(bv)

        # accumulate
        shape = theta_cs.shape
        delta_flat = theta_cs.ravel(order='C')
        total_pdf = 0.0

        for k_tuple in itertools.product(*(range(1,mj) for mj in self.degrees if mj>0)):
            # we skip 0..(m_j-1) => 1..(m_j-1)
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            dtheta = delta_flat[flat_idx]
            bprod = 1.0
            # multiply dimension contributions
            for j, kj in enumerate(k_tuple):
                bprod *= bern_vals[j][kj-1]
            total_pdf += dtheta * bprod

        # multiply by ∏(m_j)
        mprod = 1
        for mj in self.degrees:
            if mj>0:
                mprod *= mj
        pdf_val = total_pdf*mprod
        return max(0.0,pdf_val)

    def _pdf_vectorized(self, points):
        n = len(points)
        results = np.zeros(n, dtype=float)

        outside = np.any((points<=0)|(points>=1),axis=1)
        results[outside] = 0.0
        inside_mask=~outside
        if not np.any(inside_mask):
            return results

        pts_in = points[inside_mask]
        n_in = len(pts_in)
        if all(d==0 for d in self.degrees):
            val = 1.0 if np.allclose(self.theta,1.0) else 0.0
            results[inside_mask] = val
            return results

        # cumsum(diffs) approach => _delta_theta is the difference. We'll assume we skip k=0
        delta_cs = self._delta_theta.copy()
        shape = delta_cs.shape
        delta_flat = delta_cs.ravel(order='C')

        # precompute bernstein polynomials
        bern_vals = []
        for j in range(self.dim):
            mj = self.degrees[j]
            if mj>0:
                k_arr = np.arange(1,mj) # skip 0 => 1..(m_j-1)
                if len(k_arr)==0:
                    # no interior
                    return results
                bc = self._binom_coeffs_pdf[j]
                tmp = np.zeros((n_in,len(k_arr)),dtype=float)
                for i in range(n_in):
                    tmp[i,:] = self._bernstein_poly_vec(mj-1, k_arr, pts_in[i,j], bc)
                bern_vals.append(tmp)
            else:
                # deg=0 => all ones
                bern_vals.append(np.ones((n_in,1),dtype=float))

        pdf_in = np.zeros(n_in, dtype=float)

        # product over k=1..(m_j-1)
        # handle the case if any m_j=1 => empty range => we skip
        prod_ranges = []
        for mj in self.degrees:
            if mj>0:
                if mj==1:
                    # no interior
                    return results
                else:
                    prod_ranges.append(range(1,mj))

        for k_tuple in itertools.product(*prod_ranges):
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            dtheta = delta_flat[flat_idx]
            factor = np.ones(n_in,dtype=float)
            for j,kj in enumerate(k_tuple):
                factor *= bern_vals[j][:,kj-1]
            pdf_in += dtheta * factor

        mprod=1
        for mj in self.degrees:
            if mj>0:
                mprod*=mj
        pdf_in*=mprod
        pdf_in = np.maximum(pdf_in,0.0)
        results[inside_mask] = pdf_in
        return results

    # --------------------------------------------------------------------------
    #                     Conditional Distribution
    # --------------------------------------------------------------------------
    def cond_distr_1(self, *args):
        return self.cond_distr(1, *args)
    
    def cond_distr_2(self, *args):
        return self.cond_distr(2, *args)
        
    def cond_distr(self, i, *args):
        """
        Numerically compute C_{i|(-i)}(u_i|u_{-i}) by:
            numerator = ∫_0^{u_i} pdf(..., x, ...)
            denominator = ∫_0^1 pdf(..., x, ...)
        """
        if not (1 <= i <= self.dim):
            raise ValueError(f"i must be in [1, {self.dim}]")
        i0 = i-1

        def integrand(x, fixed_minus_i):
            full_point = np.insert(fixed_minus_i,i0,x)
            return self._pdf_single_point(full_point)

        if len(args)==0:
            raise ValueError("No arguments to cond_distr().")
        if len(args)==1:
            arr = np.asarray(args[0],dtype=float)
            if arr.ndim==1:
                if arr.size==self.dim:
                    return self._cond_distr_single(i0, arr, integrand)
                else:
                    raise ValueError("Expected 1D array with length=dim.")
            elif arr.ndim==2:
                if arr.shape[1]!=self.dim:
                    raise ValueError("Second dimension mismatch.")
                out=[]
                for row in arr:
                    out.append(self._cond_distr_single(i0,row,integrand))
                return np.array(out)
            else:
                raise ValueError("cond_distr() only supports 1D or 2D array.")
        else:
            if len(args)==self.dim:
                pt = np.array(args,dtype=float)
                return self._cond_distr_single(i0, pt, integrand)
            else:
                raise ValueError("Wrong number of coords.")

    def _cond_distr_single(self, u,):
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("All coordinates must be in [0,1].")
        if np.any(u == 0):
            return 0.0
        if np.all(u == 1):
            return 1.0

        # "Fixed" logic: skip k=0, cumsum of theta
        # We only do the 2D cumsum if dim=2. For higher dims, we do
        # repeated cumsum across each axis. (If you only do 2D, adapt as needed.)
        # Example for any dim:
        theta_cs = self.theta.copy()
        for ax in range(self.dim):
            theta_cs = np.cumsum(theta_cs, axis=ax)

        # Precompute the 1..m_j bernstein polynomials
        bern_vals = []
        for j in range(self.dim):
            m_j = self.degrees[j]
            # skip k=0 => range(1..m_j)
            k_arr = np.arange(1, m_j + 1)
            bc = self._binom_coeffs_cdf[j]
            bv = self._bernstein_poly_vec_cd(m_j, k_arr, u[j], bc)
            bern_vals.append(bv)

        # accumulate
        shape = theta_cs.shape
        theta_flat = theta_cs.ravel(order='C')
        total = 0.0

        for k_tuple in itertools.product(*(range(1, m_j + 1) for m_j in self.degrees)):
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            coef = theta_flat[flat_idx]

            bprod = 1.0
            for j, kj in enumerate(k_tuple):
                # indexing for bern_vals[j]: we used k_arr=1..m_j => index=kj-1
                bprod *= bern_vals[j][kj - 1]
            total += coef * bprod

        return float(np.clip(total, 0, 1))
    # --------------------------------------------------------------------------
    #                          Random Variates
    # --------------------------------------------------------------------------
    def rvs(self, n=1, random_state=None, **kwargs):
        if self.dim>2:
            raise NotImplementedError("Sampling for d>2 not implemented.")
        elif self.dim==1:
            rng = np.random.default_rng(random_state)
            return rng.uniform(0,1,(n,1))
        else:
            warnings.warn("Use BivBernsteinCopula for 2D sampling.")
            raise NotImplementedError("2D sampling not in generic class.")
