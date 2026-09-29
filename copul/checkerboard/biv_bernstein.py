r"""
Bivariate Bernstein copulas with exact, vectorised numerics.

With the cumulated coefficient matrix :math:`D` (``m x n``) and the Bernstein
basis vectors :math:`b_m(u) = (B_{m,1}(u),\dots,B_{m,m}(u))`,

.. math::

   C(u,v) = b_m(u)^\top D\, b_n(v),\quad
   \partial_1 C(u,v) = b_m'(u)^\top D\, b_n(v),\quad
   \partial_2 C(u,v) = b_m(u)^\top D\, b_n'(v).

All dependence measures reduce to integrals of products of Bernstein
polynomials (polynomials of degree at most ``2 max(m, n)``), which are
evaluated *exactly* by Gauss--Legendre quadrature or Beta-function identities:

* ``rho  = 12 sum_ij theta_ij (m - i)(n - j) / ((m+1)(n+1)) - 3``
* ``nu   = 12 sum_ij theta_ij (m - i)(m - i + 1)(n - j) / ((m+1)(m+2)(n+1)) - 2``
* ``tau  = 1 - 4 tr(D^T A_m D A_n^T)``, ``A_m[k,l] = int b'_k b_l``
* ``xi   = 6 tr(D^T Omega_m D Lambda_n) - 2``, ``Omega_m = int b' b'^T``,
  ``Lambda_n = int b b^T``
* footrule / gini from
  ``int B_{m,k}(t) B_{n,l}(t) dt = C(m,k) C(n,l) B(k+l+1, m+n-k-l+1)`` and
  ``int B_{m,k}(t) B_{n,l}(1-t) dt = C(m,k) C(n,l) B(k+n-l+1, m-k+l+1)``

(``i, j`` 0-based cell indices of ``theta``).
"""

import math
from typing import TypeAlias

import numpy as np
from scipy.special import betaln, gammaln

from copul.checkerboard import _biv_engine as eng
from copul.checkerboard.bernstein import (
    BernsteinCopula,
    bernstein_basis,
    bernstein_basis_deriv,
)
from copul.family.core.biv_core_copula import BivCoreCopula
from copul.family.core.copula_sampling_mixin import CopulaSamplingMixin


def _gl_nodes(deg):
    """Gauss-Legendre nodes/weights on [0, 1] exact for polynomials of degree ``deg``."""
    k = deg // 2 + 1
    x, w = np.polynomial.legendre.leggauss(k)
    return 0.5 * (x + 1.0), 0.5 * w


def _logcomb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


class BivBernsteinCopula(BernsteinCopula, BivCoreCopula, CopulaSamplingMixin):
    #: exact vectorized evaluation methods (passed through by the numeric API)
    _numeric_native = True

    def __init__(self, theta, check_theta=True):
        BernsteinCopula.__init__(self, theta, check_theta)
        BivCoreCopula.__init__(self)
        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]

    # ------------------------------------------------------------------
    # evaluation (call conventions as for the checkerboard copulas)
    # ------------------------------------------------------------------
    def _eval2(self, u, v, du=False, dv=False):
        if np.any(u < 0) or np.any(u > 1) or np.any(v < 0) or np.any(v > 1):
            raise ValueError("All coordinates must be in [0,1].")
        shape = np.shape(u)
        uf, vf = np.ravel(u), np.ravel(v)
        Bu = bernstein_basis_deriv(self.m, uf) if du else bernstein_basis(self.m, uf)
        Bv = bernstein_basis_deriv(self.n, vf) if dv else bernstein_basis(self.n, vf)
        return (((Bu @ self._theta_cs) * Bv).sum(axis=1)).reshape(shape)

    def cdf(self, *args, **kwargs):
        """``C(u, v) = b_m(u)^T D b_n(v)`` (vectorised)."""
        u, v, scalar = eng.parse_uv(args, kwargs)
        return eng.finish(self._eval2(u, v), scalar)

    def cdf_vectorized(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        return self._eval2(u, v)

    def pdf(self, *args, **kwargs):
        u, v, scalar = eng.parse_uv(args, kwargs)
        return eng.finish(self._eval2(u, v, du=True, dv=True), scalar)

    def cond_distr(self, i, *args, **kwargs):
        """``i = 1``: ``P(V <= v | U = u)``; ``i = 2``: ``P(U <= u | V = v)``."""
        if i not in (1, 2):
            raise ValueError(f"i must be between 1 and {self.dim}")
        u, v, scalar = eng.parse_uv(args, kwargs)
        return eng.finish(self._eval2(u, v, du=(i == 1), dv=(i == 2)), scalar)

    def cond_distr_1(self, *args, **kwargs):
        return self.cond_distr(1, *args, **kwargs)

    def cond_distr_2(self, *args, **kwargs):
        return self.cond_distr(2, *args, **kwargs)

    def rvs(self, n=1, random_state=None, **kwargs):
        """Exact sampling (Beta mixture), see :meth:`BernsteinCopula.rvs`."""
        if "size" in kwargs and kwargs["size"] is not None:
            n = kwargs["size"]
        return BernsteinCopula.rvs(self, n, random_state=random_state)

    def transpose(self):
        return BivBernsteinCopula(self.theta.T)

    # ------------------------------------------------------------------
    # exact integral matrices
    # ------------------------------------------------------------------
    @staticmethod
    def _gram(m, deriv_left, deriv_right):
        t, w = _gl_nodes(2 * m + 2)
        L = bernstein_basis_deriv(m, t) if deriv_left else bernstein_basis(m, t)
        R = bernstein_basis_deriv(m, t) if deriv_right else bernstein_basis(m, t)
        return (L * w[:, None]).T @ R

    # ------------------------------------------------------------------
    # dependence measures
    # ------------------------------------------------------------------
    def spearmans_rho(self, *args, **kwargs) -> float:
        """Spearman's rho: ``12 sum theta_ij (m-i)(n-j)/((m+1)(n+1)) - 3``."""
        m, n = self.m, self.n
        wi = (m - np.arange(m)) / (m + 1.0)
        wj = (n - np.arange(n)) / (n + 1.0)
        return float(12.0 * wi @ self.theta @ wj - 3.0)

    def kendalls_tau(self, *args, **kwargs) -> float:
        """Kendall's tau ``1 - 4 int int d1C d2C`` (exact)."""
        D = self._theta_cs
        Am = self._gram(self.m, True, False)  # int b'_k b_l
        An = self._gram(self.n, False, True)  # int b_k b'_l
        # int d1C d2C = sum D_kl D_k'l' int b'_k b_k' int b_l b'_l'
        return float(1.0 - 4.0 * np.sum(D * (Am @ D @ An.T)))

    def chatterjees_xi(self, *, condition_on_y: bool = False) -> float:
        """Chatterjee's xi ``6 int int (d_1 C)^2 - 2`` (exact);
        ``condition_on_y=True`` gives xi(U | V)."""
        D = self._theta_cs.T if condition_on_y else self._theta_cs
        m, n = D.shape
        Omega = self._gram(m, True, True)
        Lambda = self._gram(n, False, False)
        return float(6.0 * np.trace(D.T @ Omega @ D @ Lambda) - 2.0)

    def blests_nu(self, *args, **kwargs) -> float:
        """Blest's nu ``24 int int (1-u) C - 2`` (exact)."""
        m, n = self.m, self.n
        i = np.arange(m)
        j = np.arange(n)
        wi = (m - i) * (m - i + 1.0) / ((m + 1.0) * (m + 2.0))
        wj = (n - j) / (n + 1.0)
        return float(12.0 * wi @ self.theta @ wj - 2.0)

    def _diag_weights(self, anti=False):
        m, n = self.m, self.n
        k = np.arange(1, m + 1)[:, None]
        l_ = np.arange(1, n + 1)[None, :]
        if anti:
            lb = betaln(k + n - l_ + 1, m - k + l_ + 1)
        else:
            lb = betaln(k + l_ + 1, m + n - k - l_ + 1)
        return np.exp(_logcomb(m, k) + _logcomb(n, l_) + lb)

    def spearmans_footrule(self, *args, **kwargs) -> float:
        """Spearman's footrule ``6 int C(t,t) dt - 2`` (exact)."""
        return float(6.0 * np.sum(self._theta_cs * self._diag_weights()) - 2.0)

    def ginis_gamma(self, *args, **kwargs) -> float:
        """Gini's gamma ``4 (int C(t,t) + int C(t,1-t)) - 2`` (exact)."""
        i1 = np.sum(self._theta_cs * self._diag_weights())
        i2 = np.sum(self._theta_cs * self._diag_weights(anti=True))
        return float(4.0 * (i1 + i2) - 2.0)

    def spearman_footrule(self, *args, **kwargs) -> float:
        return self.spearmans_footrule()

    def gini_gamma(self, *args, **kwargs) -> float:
        return self.ginis_gamma()

    def blomqvists_beta(self, *args, **kwargs) -> float:
        return float(4.0 * self.cdf(0.5, 0.5) - 1.0)

    # ------------------------------------------------------------------
    # legacy closed-form constructors (kept for backwards compatibility)
    # ------------------------------------------------------------------
    @staticmethod
    def _construct_theta(m: int) -> np.ndarray:
        Theta = np.zeros((m, m), dtype=float)
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                numerator = (i - j) * math.comb(m, i) * math.comb(m, j)
                denom = (2 * m - i - j) * math.comb(2 * m - 1, i + j - 1)
                if denom == 0:
                    Theta[i - 1, j - 1] = 0.0 if (numerator != 0) else 1.0
                else:
                    Theta[i - 1, j - 1] = numerator / denom
        return Theta

    @staticmethod
    def _construct_omega(m: int) -> np.ndarray:
        """``Omega[k, l] = int_0^1 B'_{m,k} B'_{m,l}`` (exact quadrature)."""
        return BivBernsteinCopula._gram(m, True, True)

    @staticmethod
    def _construct_lambda(n: int) -> np.ndarray:
        """``Lambda[k, l] = int_0^1 B_{n,k} B_{n,l}`` (exact quadrature)."""
        return BivBernsteinCopula._gram(n, False, False)

    def lambda_L(self):
        """
        Lower tail dependence is zero by 2016 Pfeifer, Tsatedem, Mändle and Girschig - Example 1
        """
        return 0

    def lambda_U(self):
        """
        Upper tail dependence is zero by 2016 Pfeifer, Tsatedem, Mändle and Girschig - Example 1
        """
        return 0


BivBernstein: TypeAlias = BivBernsteinCopula
