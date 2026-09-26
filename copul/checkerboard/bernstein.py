r"""
d-dimensional Bernstein copulas.

For a nonnegative array :math:`\theta` of shape :math:`(m_1,\dots,m_d)`
(normalised to total mass one) with cumulated array
:math:`D_{k} = \sum_{i\le k}\theta_i` the Bernstein copula is

.. math::

   C(u) = \sum_{k_1=1}^{m_1}\cdots\sum_{k_d=1}^{m_d} D_{k}
          \prod_{j=1}^d B_{m_j,k_j}(u_j),\qquad
   B_{m,k}(u) = \binom{m}{k}u^k(1-u)^{m-k},

i.e. the Bernstein smoothing of the checkerboard copula with masses
:math:`\theta`.  Equivalently :math:`C` is the mixture, with weights
:math:`\theta_i`, of independent :math:`\mathrm{Beta}(i_j+1, m_j-i_j)` margins
(0-based ``i``), which gives an exact sampler.
"""

import warnings
from typing import TypeAlias

import numpy as np
from scipy.special import comb

from copul.checkerboard.check import Check
from copul.family.core.copula_plotting_mixin import CopulaPlottingMixin


def bernstein_basis(m, u):
    """``(N, m)`` array ``[B_{m,k}(u)]_{k=1..m}`` (stable at the boundary)."""
    u = np.asarray(u, dtype=float).reshape(-1, 1)
    k = np.arange(1, m + 1)[None, :]
    return comb(m, k) * u**k * (1.0 - u) ** (m - k)


def bernstein_basis_deriv(m, u):
    """``(N, m)`` array of ``d/du B_{m,k}(u)`` for ``k = 1..m``.

    Uses ``B'_{m,k} = m (B_{m-1,k-1} - B_{m-1,k})`` which has no negative
    powers (the naive product rule produces ``0 * inf = nan`` at ``u = 1``).
    """
    u = np.asarray(u, dtype=float).reshape(-1, 1)
    k = np.arange(1, m + 1)[None, :]
    lower = comb(m - 1, k - 1) * u ** (k - 1) * (1.0 - u) ** (m - k)
    upper = np.where(
        k <= m - 1,
        comb(m - 1, k) * u**k * (1.0 - u) ** np.maximum(m - 1 - k, 0),
        0.0,
    )
    return m * (lower - upper)


class BernsteinCopula(Check, CopulaPlottingMixin):
    """
    Represents a d-dimensional Bernstein Copula with possibly different degrees m_i per dimension.
    The degree along axis ``j`` equals ``theta.shape[j]``.
    """

    def __new__(cls, theta, *args, **kwargs):
        theta_arr = np.asarray(theta)
        if cls is BernsteinCopula and theta_arr.ndim == 2:
            try:
                import importlib

                bbc_module = importlib.import_module("copul.checkerboard.biv_bernstein")
                BivBernsteinCopula = bbc_module.BivBernsteinCopula
                return BivBernsteinCopula(theta, *args, **kwargs)
            except (ImportError, ModuleNotFoundError, AttributeError) as e:
                warnings.warn(
                    f"Could not import BivBernsteinCopula, falling back to generic BernsteinCopula. Error: {e}"
                )
        return super().__new__(cls)

    def __init__(self, theta, check_theta=True):
        # never modify the caller's array
        theta = np.array(theta, dtype=float)
        if theta.ndim == 0:
            raise ValueError("Theta must have at least one dimension.")
        if check_theta and np.any(theta < 0):
            raise ValueError("Theta must be nonnegative.")
        total_mass = np.sum(theta)
        matr = theta.copy()
        if total_mass > 0:
            theta = theta / total_mass
        self.theta = theta
        self.dim = self.theta.ndim

        # Each dimension's degree equals the size along that axis.
        self.degrees = [int(s) for s in self.theta.shape]
        if any(d < 1 for d in self.degrees):
            raise ValueError("Each dimension must have size >= 1.")

        # Binomial coefficients (kept for backwards compatibility).
        self._binom_coeffs_cdf = [
            np.array([comb(m_i, k, exact=True) for k in range(m_i + 1)]) for m_i in self.degrees
        ]

        # Let base Check store 'matr'
        super().__init__(matr=matr)
        self._theta_cs = self._cumsum_theta()

    def __str__(self):
        return f"BernsteinCopula(degrees={self.degrees}, dim={self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    # --- Helper Functions -----------------------------------------------------

    @staticmethod
    def _bernstein_poly_vec(m, k_vals, u, binom_coeffs):
        """Compute vector [B_{m,k}(u)] for k in k_vals."""
        return binom_coeffs[k_vals] * (u**k_vals) * ((1 - u) ** (m - k_vals))

    def _bernstein_poly_vec_cd(self, m, k_vals, u, binom_coeffs):
        """Compute derivative vector for [B_{m,k}(u)] for k in k_vals (nan-free)."""
        return bernstein_basis_deriv(m, u)[0, np.asarray(k_vals) - 1]

    def _cumsum_theta(self, with_zeros=False):
        """Return the cumulative sum of theta along each axis.

        If with_zeros=True, add a leading zero row and column.
        """
        theta_cs = self.theta.copy()
        for ax in range(self.dim):
            theta_cs = np.cumsum(theta_cs, axis=ax)

        if with_zeros:
            theta_cs = np.pad(theta_cs, pad_width=[(1, 0)] * self.dim, mode="constant")

        return theta_cs

    def _evaluate(self, points, deriv_axes=()):
        """Contract the cumulated coefficients with (derivative) bases."""
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[None, :]
        if points.shape[1] != self.dim:
            raise ValueError(f"Expected points with {self.dim} coordinates.")
        if np.any(points < 0) or np.any(points > 1):
            raise ValueError("All coordinates must be in [0,1].")
        factors = []
        for j, m_j in enumerate(self.degrees):
            if j in deriv_axes:
                factors.append(bernstein_basis_deriv(m_j, points[:, j]))
            else:
                factors.append(bernstein_basis(m_j, points[:, j]))
        return self._contract(factors, tensor=self._theta_cs)

    def _parse_points(self, args, name):
        if not args:
            raise ValueError(f"No arguments provided to {name}().")
        if len(args) == 1:
            arr = np.asarray(args[0], dtype=float)
            if arr.ndim == 1:
                if arr.size != self.dim:
                    raise ValueError(f"Input length must equal {self.dim}.")
                return arr[None, :], True
            if arr.ndim == 2:
                if arr.shape[1] != self.dim:
                    raise ValueError(f"Second dimension must be {self.dim}.")
                return arr, False
            raise ValueError(f"{name}() supports 1D or 2D arrays only.")
        if len(args) == self.dim:
            arrs = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
            scalar = arrs[0].ndim == 0
            return np.column_stack([a.ravel() for a in arrs]), scalar
        raise ValueError(f"Expected {self.dim} coordinates, got {len(args)}.")

    # --- CDF / PDF / conditional distributions --------------------------------

    def cdf(self, *args):
        """CDF; supports ``cdf(u1,...,ud)``, ``cdf([u1,...,ud])`` and ``cdf(P)``
        with ``P`` of shape ``(N, d)``."""
        pts, scalar = self._parse_points(args, "cdf")
        out = self._evaluate(pts)
        return float(out[0]) if scalar else out

    def _cdf_single_point(self, u):
        return float(self._evaluate(np.asarray(u, dtype=float)[None, :])[0])

    def pdf(self, *args):
        """Density; same call conventions as :meth:`cdf`."""
        pts, scalar = self._parse_points(args, "pdf")
        out = self._evaluate(pts, deriv_axes=tuple(range(self.dim)))
        return float(out[0]) if scalar else out

    def _pdf_single_point(self, u):
        return float(self._evaluate(np.asarray(u, dtype=float)[None, :], tuple(range(self.dim)))[0])

    def cond_distr_1(self, *args):
        return self.cond_distr(1, *args)

    def cond_distr_2(self, *args):
        return self.cond_distr(2, *args)

    def cond_distr(self, i, *args):
        r"""
        Conditional distribution :math:`\partial C / \partial u_i`, i.e. the
        distribution of the other coordinates given :math:`U_i = u_i`.
        """
        if not (1 <= i <= self.dim):
            raise ValueError(f"i must be between 1 and {self.dim}")
        pts, scalar = self._parse_points(args, "cond_distr")
        out = self._evaluate(pts, deriv_axes=(i - 1,))
        return float(out[0]) if scalar else out

    def _cond_distr_single(self, u, i):
        return float(self._evaluate(np.asarray(u, dtype=float)[None, :], (i - 1,))[0])

    # --- sampling ------------------------------------------------------------

    def rvs(self, n=1, random_state=None, **kwargs):
        """Exact sampling via the Beta-mixture representation.

        A cell ``i`` is drawn with probability ``theta_i``; then the
        coordinates are independent ``Beta(i_j + 1, m_j - i_j)`` (0-based
        ``i_j``).  ``random_state``: int, Generator or ``None`` (NumPy's global
        generator, never reseeded).  Extra keywords (e.g. ``approximate``) are
        accepted and ignored.
        """
        from copul.checkerboard._biv_engine import resolve_rng

        rng = resolve_rng(random_state)
        flat = self.theta.ravel()
        idx = rng.choice(flat.size, size=int(n), p=flat / flat.sum())
        cells = np.unravel_index(idx, self.theta.shape)
        out = np.empty((int(n), self.dim))
        for j, m_j in enumerate(self.degrees):
            out[:, j] = rng.beta(cells[j] + 1.0, m_j - cells[j])
        return out


Bernstein: TypeAlias = BernsteinCopula
