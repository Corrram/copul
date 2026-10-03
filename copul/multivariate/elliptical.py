r"""
:math:`d`-dimensional Gaussian and Student-t copulas.

For a correlation matrix :math:`R` (symmetric, positive definite, unit
diagonal) the Gaussian copula is the copula of :math:`N_d(0,R)`,

.. math::

   C_R(u) = \Phi_R\bigl(\Phi^{-1}(u_1),\dots,\Phi^{-1}(u_d)\bigr),\qquad
   c_R(u) = |R|^{-1/2}\exp\Bigl(-\tfrac12 x^\top(R^{-1}-I)x\Bigr),
   \quad x_i=\Phi^{-1}(u_i),

and the Student-t copula with :math:`\nu>0` degrees of freedom is the copula
of the multivariate :math:`t_\nu(0,R)` distribution,

.. math::

   c_{R,\nu}(u) = \frac{\Gamma(\frac{\nu+d}2)\,\Gamma(\frac\nu2)^{d-1}}
   {\Gamma(\frac{\nu+1}2)^d\,|R|^{1/2}}
   \Bigl(1+\frac{x^\top R^{-1}x}{\nu}\Bigr)^{-\frac{\nu+d}2}
   \prod_{i=1}^d\Bigl(1+\frac{x_i^2}{\nu}\Bigr)^{\frac{\nu+1}2},
   \quad x_i = t_\nu^{-1}(u_i)

(Demarta & McNeil, 2005; Joe, 2014, §4.3--4.4).  Both are radially
symmetric, their bivariate margins are the bivariate Gaussian / t copulas
with the corresponding entries of :math:`R`, and for every elliptical copula
Kendall's tau of the :math:`(i,j)` margin is
:math:`\tau_{ij}=\tfrac2\pi\arcsin R_{ij}` (Lindskog, McNeil & Schmock,
2003).  The coefficient of (upper and lower) tail dependence of a t margin is
:math:`\lambda_{ij} = 2\,t_{\nu+1}\bigl(-\sqrt{(\nu+1)(1-R_{ij})/(1+R_{ij})}\bigr)`
and zero for the Gaussian copula (Embrechts, Lindskog & McNeil, 2003).

The cdf uses the closed bivariate formulas of copul for :math:`d=2` and
SciPy's (quasi-Monte Carlo) multivariate normal / t distribution functions
for :math:`d\ge 3` (absolute error about :math:`10^{-5}` / :math:`10^{-4}`,
reproducible through a fixed seed); densities, sampling and margins are
exact.

References
----------
* Demarta, S. and McNeil, A. J. (2005). The t copula and related copulas.
  *Int. Statist. Rev.* 73, 111--129.
* Embrechts, P., Lindskog, F. and McNeil, A. (2003). Modelling dependence
  with copulas and applications to risk management. In *Handbook of Heavy
  Tailed Distributions in Finance*, ch. 8, 329--384. Elsevier.
* Genz, A. and Bretz, F. (2009). *Computation of Multivariate Normal and t
  Probabilities*. Springer.
* Joe, H. (2014). *Dependence Modeling with Copulas*. CRC Press.
* Lindskog, F., McNeil, A. and Schmock, U. (2003). Kendall's tau for
  elliptical distributions. In *Credit Risk*, 149--156. Physica, Heidelberg.
* Klaassen, C. A. J. and Wellner, J. A. (1997). Efficient estimation in
  the bivariate normal copula model: normal margins are least favourable.
  *Bernoulli* 3, 55--77.
* Kruskal, W. H. (1958). Ordinal measures of association. *JASA* 53,
  814--861.
* Mashal, R. and Zeevi, A. (2002). Beyond correlation: extreme co-movements
  between financial assets. Technical report, Columbia University.
* Rosenblatt, M. (1952). Remarks on a multivariate transformation.
  *Ann. Math. Statist.* 23, 470--472.
* Rousseeuw, P. J. and Molenberghs, G. (1993). Transformation of non
  positive semidefinite correlation matrices. *Comm. Statist. Theory
  Methods* 22, 965--984.
* Sibuya, M. (1960). Bivariate extreme statistics, I. *Ann. Inst.
  Statist. Math.* 11, 195--210.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg, optimize, stats
from scipy.special import gammaln, ndtr, ndtri, stdtr, stdtrit

from copul.multivariate.base import CopulaND

__all__ = ["EllipticalCopulaND", "GaussianND", "StudentTND", "nearest_correlation"]

_CDF_SEED = 20240611


def _validate_corr(R: Any) -> np.ndarray:
    R = np.array(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] < 2:
        raise ValueError(f"the correlation matrix must be square of size >= 2, got {R.shape}.")
    if not np.allclose(R, R.T, atol=1e-10):
        raise ValueError("the correlation matrix must be symmetric.")
    if not np.allclose(np.diag(R), 1.0, atol=1e-10):
        raise ValueError("the correlation matrix must have a unit diagonal.")
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    try:
        np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        raise ValueError("the correlation matrix must be positive definite.") from None
    return R


def nearest_correlation(A: Any, eps: float = 1e-8) -> np.ndarray:
    r"""Positive definite correlation matrix close to a symmetric matrix ``A``.

    Eigenvalues are clipped at ``eps`` and the result is rescaled to a unit
    diagonal (Rousseeuw & Molenberghs, 1993), the standard repair of
    rank-based correlation estimates such as :math:`\sin(\pi\hat\tau/2)`.
    """
    A = 0.5 * (np.asarray(A, dtype=float) + np.asarray(A, dtype=float).T)
    w, V = np.linalg.eigh(A)
    if w.min() > eps:
        out = A
    else:
        out = (V * np.maximum(w, eps)) @ V.T
    s = np.sqrt(np.diag(out))
    out = out / np.outer(s, s)
    np.fill_diagonal(out, 1.0)
    return out


def _pairwise_tau(U: np.ndarray) -> np.ndarray:
    d = U.shape[1]
    T = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            T[i, j] = T[j, i] = stats.kendalltau(U[:, i], U[:, j]).statistic
    return T


def _prepare_uniform(data: Any, pseudo_obs: bool) -> np.ndarray:
    from copul.stats._utils import as_data
    from copul.stats.pseudo_obs import pseudo_obs as _po

    X = as_data(data, min_dim=2)
    if pseudo_obs:
        return _po(X)
    if np.any(X <= 0) or np.any(X >= 1):
        raise ValueError("with pseudo_obs=False the data must lie in the open unit cube.")
    return X


class EllipticalCopulaND(CopulaND):
    """Common base of :class:`GaussianND` and :class:`StudentTND`."""

    radially_symmetric = True

    def __init__(self, corr: Any) -> None:
        R = _validate_corr(corr)
        super().__init__(R.shape[0])
        self.corr = R
        self._L = np.linalg.cholesky(R)
        self._logdet = 2.0 * float(np.sum(np.log(np.diag(self._L))))
        off = R[~np.eye(self.dim, dtype=bool)]
        self.exchangeable = bool(np.allclose(off, off[0]))

    @property
    def corr_matrix(self) -> np.ndarray:
        """The correlation matrix :math:`R`."""
        return self.corr

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def _quad(self, X: np.ndarray) -> np.ndarray:
        """Quadratic form :math:`x^\top R^{-1}x` row-wise."""
        Y = linalg.solve_triangular(self._L, X.T, lower=True, check_finite=False)
        return np.sum(Y * Y, axis=0)

    def kendalls_tau_matrix(self) -> np.ndarray:
        r""":math:`\tau_{ij} = \frac2\pi\arcsin R_{ij}` (Lindskog, McNeil & Schmock, 2003)."""
        return 2.0 / np.pi * np.arcsin(self.corr)

    def _sub(self, idx):
        return self.corr[np.ix_(list(idx), list(idx))]

    def _bivariate_cdf(self, U):
        biv = self.__dict__.get("_biv")
        if biv is None:
            biv = self.__dict__["_biv"] = self._margin((0, 1))
        return np.asarray(biv.cdf(U[:, 0], U[:, 1]), dtype=float)


class GaussianND(EllipticalCopulaND):
    r""":math:`d`-dimensional Gaussian copula :math:`C_R` (see module docstring).

    Parameters
    ----------
    corr : array_like of shape (d, d)
        Positive definite correlation matrix :math:`R`.

    Examples
    --------
    >>> import numpy as np
    >>> from copul.multivariate import GaussianND
    >>> R = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]])
    >>> C = GaussianND(R)
    >>> C.margin(0, 1)            # the bivariate copul Gaussian copula
    Gaussian(rho=0.5)
    >>> X = C.rvs(1000, random_state=0)
    >>> X.shape
    (1000, 3)
    """

    _cheap_cdf = False

    def __init__(self, corr: Any) -> None:
        super().__init__(corr)
        if self.dim == 2:
            self._cheap_cdf = True

    def __repr__(self) -> str:
        return f"GaussianND(dim={self.dim})"

    def _cdf(self, U):
        if self.dim == 2:
            return self._bivariate_cdf(U)
        out = np.zeros(U.shape[0])
        inside = np.all(U > 0.0, axis=1)
        if np.any(inside):
            X = ndtri(U[inside])
            dist = stats.multivariate_normal(mean=np.zeros(self.dim), cov=self.corr)
            out[inside] = np.atleast_1d(dist.cdf(X, rng=np.random.default_rng(_CDF_SEED)))
        return out

    def _logpdf(self, U):
        X = ndtri(U)
        return -0.5 * self._logdet - 0.5 * (self._quad(X) - np.sum(X * X, axis=1))

    def _rvs(self, n, rng):
        Z = rng.standard_normal((n, self.dim)) @ self._L.T
        return ndtr(Z)

    def _margin(self, idx):
        if len(idx) == 2:
            from copul.family.elliptical.gaussian import Gaussian

            return Gaussian(float(self.corr[idx[0], idx[1]]))
        return GaussianND(self._sub(idx))

    def spearmans_rho_matrix(self) -> np.ndarray:
        r""":math:`\rho^S_{ij} = \frac6\pi\arcsin(R_{ij}/2)` (Kruskal, 1958)."""
        return 6.0 / np.pi * np.arcsin(self.corr / 2.0)

    def tail_dependence_matrix(self) -> np.ndarray:
        """Tail dependence coefficients: zero off the diagonal (Sibuya, 1960)."""
        return np.eye(self.dim)

    def rosenblatt(self, U: Any) -> np.ndarray:
        r"""Rosenblatt (1952) transform :math:`(C_1(u_1), C_{2|1}(u_2|u_1),\dots)`.

        For the Gaussian copula :math:`C_{k|1..k-1}(u_k|u_{<k}) =
        \Phi(z_k)` with :math:`z = L^{-1}x`, :math:`x_i=\Phi^{-1}(u_i)` and the
        Cholesky factor :math:`R = LL^\top`.  Maps a sample of :math:`C_R` to
        independent uniforms.
        """
        U = np.atleast_2d(np.asarray(U, dtype=float))
        X = ndtri(np.clip(U, 1e-300, 1 - 1e-16))
        Z = linalg.solve_triangular(self._L, X.T, lower=True).T
        return ndtr(Z)

    @classmethod
    def fit(cls, data: Any, method: str = "itau", pseudo_obs: bool = True) -> GaussianND:
        r"""Estimate :math:`R` from data.

        Parameters
        ----------
        data : array_like of shape (n, d)
        method : {"itau", "irho", "normal_scores"}
            * ``"itau"``: :math:`\hat R_{ij}=\sin(\pi\hat\tau_{ij}/2)`
              (Lindskog, McNeil & Schmock, 2003);
            * ``"irho"``: :math:`\hat R_{ij}=2\sin(\pi\hat\rho^S_{ij}/6)`;
            * ``"normal_scores"``: correlation matrix of the normal scores
              :math:`\Phi^{-1}(\hat U_{ij})` (Klaassen & Wellner, 1997).

            The estimate is repaired to a positive definite correlation
            matrix by :func:`nearest_correlation` if necessary.
        pseudo_obs : bool
            Rank-transform the data (default) or use them as given (values in
            :math:`(0,1)`).
        """
        U = _prepare_uniform(data, pseudo_obs)
        method = method.lower()
        if method == "itau":
            R = np.sin(np.pi * _pairwise_tau(U) / 2.0)
        elif method == "irho":
            S = stats.spearmanr(U).statistic
            S = np.array([[1.0, S], [S, 1.0]]) if np.ndim(S) == 0 else np.asarray(S)
            R = 2.0 * np.sin(np.pi * S / 6.0)
        elif method in ("normal_scores", "ns"):
            R = np.corrcoef(ndtri(U), rowvar=False)
        else:
            raise ValueError("method must be 'itau', 'irho' or 'normal_scores'.")
        return cls(nearest_correlation(R))


class StudentTND(EllipticalCopulaND):
    r""":math:`d`-dimensional Student-t copula :math:`C_{R,\nu}` (see module docstring).

    Parameters
    ----------
    corr : array_like of shape (d, d)
        Positive definite correlation matrix :math:`R`.
    nu : float
        Degrees of freedom :math:`\nu>0`.

    Examples
    --------
    >>> import numpy as np
    >>> from copul.multivariate import StudentTND
    >>> C = StudentTND(np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]]), nu=4)
    >>> C.margin(0, 2)
    StudentT(rho=0.3, nu=4)
    >>> round(float(C.tail_dependence_matrix()[0, 1]), 4)
    0.2532
    """

    _cheap_cdf = False

    def __init__(self, corr: Any, nu: float) -> None:
        super().__init__(corr)
        nu = float(nu)
        if not nu > 0:
            raise ValueError("nu must be positive.")
        self.nu = nu
        d = self.dim
        self._logk = (
            gammaln((nu + d) / 2.0) + (d - 1) * gammaln(nu / 2.0) - d * gammaln((nu + 1) / 2.0)
        )
        if d == 2:
            self._cheap_cdf = True

    def __repr__(self) -> str:
        return f"StudentTND(dim={self.dim}, nu={self.nu:g})"

    def _cdf(self, U):
        if self.dim == 2:
            return self._bivariate_cdf(U)
        out = np.zeros(U.shape[0])
        inside = np.all(U > 0.0, axis=1)
        if np.any(inside):
            X = stdtrit(self.nu, U[inside])
            dist = stats.multivariate_t(loc=np.zeros(self.dim), shape=self.corr, df=self.nu)
            out[inside] = np.atleast_1d(dist.cdf(X, random_state=_CDF_SEED))
        return out

    def _logpdf(self, U):
        nu, d = self.nu, self.dim
        X = stdtrit(nu, U)
        return (
            self._logk
            - 0.5 * self._logdet
            - 0.5 * (nu + d) * np.log1p(self._quad(X) / nu)
            + 0.5 * (nu + 1) * np.sum(np.log1p(X * X / nu), axis=1)
        )

    def _rvs(self, n, rng):
        Z = rng.standard_normal((n, self.dim)) @ self._L.T
        W = np.sqrt(self.nu / rng.chisquare(self.nu, n))
        return stdtr(self.nu, Z * W[:, None])

    def _margin(self, idx):
        if len(idx) == 2:
            from copul.family.elliptical.student_t import StudentT

            return StudentT(rho=float(self.corr[idx[0], idx[1]]), nu=self.nu)
        return StudentTND(self._sub(idx), self.nu)

    def tail_dependence_matrix(self) -> np.ndarray:
        r""":math:`\lambda_{ij}=2t_{\nu+1}\bigl(-\sqrt{(\nu+1)(1-R_{ij})/(1+R_{ij})}\bigr)`
        (Embrechts, Lindskog & McNeil, 2003); upper and lower coincide."""
        R = np.clip(self.corr, -1.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = -np.sqrt((self.nu + 1.0) * (1.0 - R) / (1.0 + R))
        out = 2.0 * stdtr(self.nu + 1.0, arg)
        np.fill_diagonal(out, 1.0)
        return out

    @classmethod
    def fit(
        cls,
        data: Any,
        nu: float | None = None,
        pseudo_obs: bool = True,
        nu_bounds: tuple[float, float] = (0.5, 200.0),
    ) -> StudentTND:
        r"""Estimate :math:`(R,\nu)` from data.

        :math:`R` by Kendall's tau inversion
        :math:`\hat R_{ij}=\sin(\pi\hat\tau_{ij}/2)` (valid for all elliptical
        copulas, Lindskog, McNeil & Schmock, 2003); unless ``nu`` is given,
        :math:`\nu` maximizes the pseudo-log-likelihood for fixed
        :math:`\hat R` (Mashal & Zeevi, 2002; Demarta & McNeil, 2005).

        Parameters
        ----------
        data : array_like of shape (n, d)
        nu : float, optional
            Fixed degrees of freedom.
        pseudo_obs : bool
            Rank-transform the data (default).
        nu_bounds : (float, float)
            Search interval for :math:`\nu`.
        """
        U = _prepare_uniform(data, pseudo_obs)
        R = nearest_correlation(np.sin(np.pi * _pairwise_tau(U) / 2.0))
        if nu is not None:
            return cls(R, nu)

        def nll(log_nu):
            return -float(np.sum(cls(R, np.exp(log_nu))._logpdf(U)))

        res = optimize.minimize_scalar(
            nll, bounds=(np.log(nu_bounds[0]), np.log(nu_bounds[1])), method="bounded"
        )
        return cls(R, float(np.exp(res.x)))
