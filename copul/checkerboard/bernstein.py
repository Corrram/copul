import itertools
import warnings
import numpy as np
from scipy.special import comb
from typing import TypeAlias

# Adjust these imports as needed.
from copul.checkerboard.check import Check
from copul.families.core.copula_plotting_mixin import CopulaPlottingMixin


class BernsteinCopula(Check, CopulaPlottingMixin):
    """
    Represents a d-dimensional Bernstein Copula with possibly different degrees m_i per dimension.
    This version uses cumulative sum logic (skipping k=0) to compute the CDF/PDF.
    """

    def __new__(cls, theta, *args, **kwargs):
        theta_arr = np.asarray(theta)
        if cls is BernsteinCopula and theta_arr.ndim == 2:
            try:
                import importlib
                bbc_module = importlib.import_module("copul.checkerboard.biv_bernstein")
                BivBernsteinCopula = getattr(bbc_module, "BivBernsteinCopula")
                return BivBernsteinCopula(theta, *args, **kwargs)
            except (ImportError, ModuleNotFoundError, AttributeError) as e:
                warnings.warn(
                    f"Could not import BivBernsteinCopula, falling back to generic BernsteinCopula. Error: {e}"
                )
        return super().__new__(cls)

    def __init__(self, theta, check_theta=True):
        theta = np.asarray(theta, dtype=float)
        total_mass = np.sum(theta)
        if total_mass > 0:
            theta /= total_mass
        self.theta = theta
        self.dim = self.theta.ndim
        if self.dim == 0:
            raise ValueError("Theta must have at least one dimension.")

        # Each dimension's degree m_i is one less than the size along that axis.
        self.degrees = [s for s in self.theta.shape]
        if any(d < 0 for d in self.degrees):
            raise ValueError("Each dimension must have size >= 1.")

        if check_theta:
            # Optionally add additional checks here (e.g., negativity).
            pass

        # Precompute binomial coefficients for each dimension.
        self._binom_coeffs_cdf = [
            np.array([comb(m_i, k, exact=True) for k in range(m_i + 1)])
            for m_i in self.degrees
        ]

        # Let base Check store 'matr'
        super().__init__(matr=self.theta)

    def __str__(self):
        return f"BernsteinCopula(degrees={self.degrees}, dim={self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    # --- Helper Functions -----------------------------------------------------

    @staticmethod
    def _bernstein_poly_vec(m, k_vals, u, binom_coeffs):
        """Compute vector [B_{m,k}(u)] for k in k_vals."""
        return binom_coeffs[k_vals] * (u ** k_vals) * ((1 - u) ** (m - k_vals))

    def _bernstein_poly_vec_cd(self, m, k_vals, u, binom_coeffs):
        """Compute derivative vector for [B_{m,k}(u)] for k in k_vals."""
        term1 = binom_coeffs[k_vals] * k_vals * (u ** (k_vals - 1)) * ((1 - u) ** (m - k_vals))
        term2 = binom_coeffs[k_vals] * (m - k_vals) * (u ** k_vals) * ((1 - u) ** (m - k_vals - 1))
        return term1 - term2

    def _cumsum_theta(self):
        """Return the cumulative sum of theta along each axis."""
        theta_cs = self.theta.copy()
        for ax in range(self.dim):
            theta_cs = np.cumsum(theta_cs, axis=ax)
        return theta_cs

    def _prepare_bern_vals(self, u, use_deriv=False, cond_index=None):
        """
        For a given point u (1D array of length dim), return a list of Bernstein polynomial
        vectors (skipping k=0) for each dimension. If use_deriv is True, use the derivative version.
        If cond_index is not None, then for dimension cond_index use the derivative function.
        """
        bern_vals = []
        for j in range(self.dim):
            m_j = self.degrees[j]
            k_arr = np.arange(1, m_j + 1)
            bc = self._binom_coeffs_cdf[j]
            if cond_index is not None:
                # For the condition index, use derivative; else regular.
                if j == cond_index:
                    bern_vals.append(self._bernstein_poly_vec_cd(m_j, k_arr, u[j], bc))
                else:
                    bern_vals.append(self._bernstein_poly_vec(m_j, k_arr, u[j], bc))
            else:
                if use_deriv:
                    bern_vals.append(self._bernstein_poly_vec_cd(m_j, k_arr, u[j], bc))
                else:
                    bern_vals.append(self._bernstein_poly_vec(m_j, k_arr, u[j], bc))
        return bern_vals

    def _accumulate_value(self, bern_vals, theta_cs):
        """
        Sum over all multi-indices (skipping k=0) the product of the corresponding
        cumulative theta and Bernstein polynomial values.
        """
        shape = theta_cs.shape
        theta_flat = theta_cs.ravel(order='C')
        total = 0.0
        ranges = [range(m) for m in self.degrees]
        for k_tuple in itertools.product(*ranges):
            flat_idx = np.ravel_multi_index(k_tuple, shape, order='C')
            coef = theta_flat[flat_idx]
            prod = 1.0
            for j, kj in enumerate(k_tuple):
                prod *= bern_vals[j][kj]
            total += coef * prod
        return float(total)

    # --- CDF Methods ----------------------------------------------------------

    def cdf(self, *args):
        # Support cdf(u1,...,ud), cdf([u1,...,ud]), or cdf([[u1,...,ud], ...])
        if not args:
            raise ValueError("No arguments provided to cdf().")
        if len(args) == 1:
            arr = np.asarray(args[0], dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._cdf_single_point(arr)
                else:
                    raise ValueError(f"Input length must equal {self.dim}.")
            elif arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError(f"Second dimension must be {self.dim}.")
                # Simplified vectorized version: apply single-point method per row.
                return np.array([self._cdf_single_point(row) for row in arr])
            else:
                raise ValueError("cdf() supports 1D or 2D arrays only.")
        elif len(args) == self.dim:
            return self._cdf_single_point(np.array(args, dtype=float))
        else:
            raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}.")

    def _cdf_single_point(self, u):
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("All coordinates must be in [0,1].")
        if np.any(u == 0):
            return 0.0
        if np.all(u == 1):
            return 1.0

        theta_cs = self._cumsum_theta()
        bern_vals = self._prepare_bern_vals(u, use_deriv=False)
        return self._accumulate_value(bern_vals, theta_cs)

    # --- PDF Methods ----------------------------------------------------------

    def pdf(self, *args):
        # Support pdf(u1,...,ud), pdf([u1,...,ud]), or pdf([[u1,...,ud], ...])
        if not args:
            raise ValueError("No arguments provided to pdf().")
        if len(args) == 1:
            arr = np.asarray(args[0], dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._pdf_single_point(arr)
                else:
                    raise ValueError("Wrong shape for 1D input.")
            elif arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError("Second dimension must match copula dim.")
                return np.array([self._pdf_single_point(row) for row in arr])
            else:
                raise ValueError("pdf() supports 1D or 2D arrays only.")
        elif len(args) == self.dim:
            return self._pdf_single_point(np.array(args, dtype=float))
        else:
            raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}.")

    def _pdf_single_point(self, u):
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("All coordinates must be in [0,1].")
        if np.any(u == 0) or np.all(u == 1):
            return 0.0  # For boundaries, follow the original logic.

        theta_cs = self._cumsum_theta()
        bern_vals = self._prepare_bern_vals(u, use_deriv=True)
        return self._accumulate_value(bern_vals, theta_cs)

    # --- Conditional Distribution Methods -----------------------------------

    def cond_distr_1(self, *args):
        return self.cond_distr(1, *args)

    def cond_distr_2(self, *args):
        return self.cond_distr(2, *args)

    def cond_distr(self, i, *args):
        """
        Numerically compute C_{i|(-i)}(u_i|u_{-i}) using the ratio:
            C(u1,...,u_{i-1}, u_i, 1,...,1) / C(u1,...,u_{i-1}, 1,...,1)
        """
        if not (1 <= i <= self.dim):
            raise ValueError(f"i must be between 1 and {self.dim}")
        if not args:
            raise ValueError("No arguments provided to cond_distr().")
        if len(args) == 1:
            arr = np.asarray(args[0], dtype=float)
            if arr.ndim == 1:
                if arr.size == self.dim:
                    return self._cond_distr_single(arr, i)
                else:
                    raise ValueError("Expected 1D array of length dim.")
            elif arr.ndim == 2:
                return np.array([self._cond_distr_single(row, i) for row in arr])
            else:
                raise ValueError("cond_distr() supports 1D or 2D arrays only.")
        elif len(args) == self.dim:
            return self._cond_distr_single(np.array(args, dtype=float), i)
        else:
            raise ValueError("Wrong number of coordinates.")

    def _cond_distr_single(self, u, i):
        if np.any(u < 0) or np.any(u > 1):
            raise ValueError("All coordinates must be in [0,1].")
        if np.any(u == 0) or np.all(u == 1):
            return 0.0

        theta_cs = self._cumsum_theta()
        # Use derivative only for the conditional coordinate (i-1)
        bern_vals = self._prepare_bern_vals(u, use_deriv=False, cond_index=i - 1)
        return self._accumulate_value(bern_vals, theta_cs)


Bernstein: TypeAlias = BernsteinCopula
