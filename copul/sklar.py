r"""
Sklar's theorem: joint distributions from copulas and margins, and back.

**Sklar's theorem** (Sklar, 1959; Nelsen, 2006, Thm. 2.3.3 and 2.10.9).
For every :math:`d`-dimensional distribution function :math:`H` with
univariate margins :math:`F_1,\dots,F_d` there is a :math:`d`-copula
:math:`C` with

.. math::

   H(x_1,\dots,x_d) = C\bigl(F_1(x_1),\dots,F_d(x_d)\bigr)
   \qquad\text{for all } x\in\overline{\mathbb R}^d,

and :math:`C` is unique on :math:`\operatorname{Ran}F_1\times\dots\times
\operatorname{Ran}F_d`; in particular it is unique if all margins are
continuous, and then

.. math::

   C(u) = H\bigl(F_1^{-1}(u_1),\dots,F_d^{-1}(u_d)\bigr)

(Nelsen, 2006, Cor. 2.3.7 and §2.10).  Conversely, for any copula
:math:`C` and univariate distribution functions :math:`F_i` the right-hand
side defines a joint distribution function with margins :math:`F_i`.

This module provides

* :class:`JointDistribution` -- the distribution :math:`H=C(F_1,\dots,F_d)`
  for a copul copula (bivariate families, constructions, or a
  :math:`d`-dimensional :class:`~copul.multivariate.CopulaND`) and SciPy
  frozen marginals: ``cdf``, ``pdf``/``logpdf`` (continuous margins:
  :math:`h(x)=c(F_1(x_1),\dots,F_d(x_d))\prod_i f_i(x_i)`), ``pmf``
  (discrete margins), survival function, sampling, rectangle probabilities,
  conditional distributions and quantiles, regression curves, covariance
  and correlation (Hoeffding's formula) and fitting (IFM and canonical
  maximum likelihood);
* :func:`copula_from_joint` -- the copula
  :math:`C(u,v)=H(F^{-1}(u),G^{-1}(v))` of a continuous bivariate joint
  distribution as a fully functional copul copula object
  (:class:`SklarCopula`);
* :class:`EmpiricalMarginal` -- the empirical distribution of a sample as a
  SciPy-like marginal.

**Discrete margins.**  If some :math:`F_i` is not continuous, the copula is
determined only on :math:`\prod_i\operatorname{Ran}F_i` and many copulas
yield the same :math:`H`; dependence properties of :math:`C` then need not
transfer to :math:`H` (Genest & Nešlehová, 2007).  :class:`JointDistribution`
supports ``cdf``, ``sf``, ``pmf`` (all margins discrete), ``rvs`` and
rectangle probabilities for such margins -- quantities that only involve
:math:`C` on the range of the margins -- and raises for densities,
conditional distributions and regression curves.

References
----------
* Bouyé, E. and Salmon, M. (2009). Dynamic copula quantile regressions and
  tail area dynamic dependence in Forex markets. *Eur. J. Finance* 15,
  721--750.
* Embrechts, P., McNeil, A. and Straumann, D. (2002). Correlation and
  dependence in risk management: properties and pitfalls. In *Risk
  Management: Value at Risk and Beyond*, 176--223. Cambridge Univ. Press.
* Genest, C., Ghoudi, K. and Rivest, L.-P. (1995). A semiparametric
  estimation procedure of dependence parameters in multivariate families of
  distributions. *Biometrika* 82, 543--552.
* Genest, C. and Nešlehová, J. (2007). A primer on copulas for count data.
  *ASTIN Bull.* 37, 475--515.
* Hoeffding, W. (1940). Masstabinvariante Korrelationstheorie. *Schriften
  Math. Inst. Univ. Berlin* 5, 181--233.
* Joe, H. (2005). Asymptotic efficiency of the two-stage estimation method
  for copula-based models. *J. Multivariate Anal.* 94, 401--419.
* Joe, H. and Xu, J. J. (1996). The estimation method of inference functions
  for margins for multivariate models. Tech. Rep. 166, Dept. of Statistics,
  University of British Columbia.
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer,
  §2.3, §2.9, §2.10, §5.1.
* Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs
  marges. *Publ. Inst. Statist. Univ. Paris* 8, 229--231.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri

from copul.family.constructions._base import NumericBivCopula
from copul.multivariate.base import box_volume, finish, parse_points

__all__ = ["EmpiricalMarginal", "JointDistribution", "SklarCopula", "copula_from_joint"]

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_W_EPS = 2.0**-53


# ---------------------------------------------------------------------------
# marginals
# ---------------------------------------------------------------------------


def _is_discrete(m: Any) -> bool:
    flag = getattr(m, "is_discrete", None)
    if flag is not None:
        return bool(flag)
    return isinstance(getattr(m, "dist", None), stats.rv_discrete)


def _check_marginal(m: Any, i: int) -> None:
    for attr in ("cdf", "ppf"):
        if not callable(getattr(m, attr, None)):
            raise TypeError(
                f"marginal {i} ({type(m).__name__}) has no {attr}(); pass SciPy frozen "
                "distributions such as scipy.stats.norm(0, 1)."
            )


class EmpiricalMarginal:
    r"""Empirical distribution of a univariate sample (SciPy-like).

    :math:`F_n(x)=\frac1n\#\{k: x_k\le x\}` with generalized inverse
    :math:`F_n^{-1}(q)=\inf\{x: F_n(x)\ge q\}`.  It is a discrete
    distribution, so a :class:`JointDistribution` with empirical margins
    supports ``cdf``, ``sf``, ``pmf``, ``rvs`` and rectangle probabilities.

    Parameters
    ----------
    data : array_like
        One-dimensional sample.
    """

    is_discrete = True

    def __init__(self, data: Any) -> None:
        x = np.sort(np.asarray(data, dtype=float).ravel())
        if x.size < 1 or not np.all(np.isfinite(x)):
            raise ValueError("EmpiricalMarginal needs a non-empty, finite sample.")
        self.x = x
        self.n = x.size

    def __repr__(self) -> str:
        return f"EmpiricalMarginal(n={self.n})"

    def cdf(self, x: Any) -> np.ndarray:
        return np.searchsorted(self.x, np.asarray(x, dtype=float), side="right") / self.n

    def sf(self, x: Any) -> np.ndarray:
        return 1.0 - self.cdf(x)

    def ppf(self, q: Any) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        k = np.clip(np.ceil(q * self.n).astype(int) - 1, 0, self.n - 1)
        return self.x[k]

    def pmf(self, x: Any) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo = np.searchsorted(self.x, x, side="left")
        hi = np.searchsorted(self.x, x, side="right")
        return (hi - lo) / self.n

    def rvs(self, size: Any = 1, random_state: Any = None) -> np.ndarray:
        from copul.family.constructions._base import as_rng

        return as_rng(random_state).choice(self.x, size=size, replace=True)

    def mean(self) -> float:
        return float(self.x.mean())

    def var(self) -> float:
        return float(self.x.var())

    def std(self) -> float:
        return float(self.x.std())


# ---------------------------------------------------------------------------
# joint distribution
# ---------------------------------------------------------------------------


class JointDistribution:
    r"""Joint distribution :math:`H(x)=C(F_1(x_1),\dots,F_d(x_d))` (Sklar's theorem).

    Parameters
    ----------
    copula : copul copula
        A bivariate copul copula (``cp.Clayton(2)``, ``cp.Gaussian(0.5)``,
        constructions, checkerboards, ...) or a :math:`d`-dimensional
        :class:`~copul.multivariate.CopulaND` (any object accepted by
        :func:`~copul.multivariate.as_copula_nd`).
    marginals : sequence of SciPy frozen distributions, or a single one
        One per component (a single distribution is used for all), e.g.
        ``[scipy.stats.norm(0, 1), scipy.stats.expon(scale=2)]``.  Discrete
        margins (``scipy.stats.poisson(3)``) are allowed for ``cdf``, ``sf``,
        ``pmf``, ``rvs`` and rectangle probabilities (see module docstring).

    Attributes
    ----------
    dim : int
    copula : the copula as given
    marginals : list
    fit_info : dict or None
        Estimation details for objects created by :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import copul as cp
    >>> from copul.sklar import JointDistribution
    >>> H = JointDistribution(cp.Gaussian(0.6), [stats.norm(), stats.norm()])
    >>> bool(np.isclose(H.cdf([0.0, 0.0]), 0.25 + np.arcsin(0.6) / (2 * np.pi)))
    True
    >>> round(float(H.correlation()[0, 1]), 8)     # Pearson correlation = rho
    0.6
    >>> round(float(H.regression(1.0)), 8)        # E[X2 | X1 = 1] = 0.6
    0.6
    """

    def __init__(self, copula: Any, marginals: Any) -> None:
        from copul.multivariate.basic import as_copula_nd

        self.copula = copula
        self._C = as_copula_nd(copula)
        d = self._C.dim
        if not isinstance(marginals, (list, tuple)):
            marginals = [marginals] * d
        marginals = list(marginals)
        if len(marginals) != d:
            raise ValueError(
                f"need {d} marginals for a {d}-dimensional copula, got {len(marginals)}."
            )
        for i, m in enumerate(marginals):
            _check_marginal(m, i)
        self.marginals = marginals
        self.dim = d
        self._discrete = [_is_discrete(m) for m in marginals]
        self.fit_info: dict | None = None

    def __repr__(self) -> str:
        ms = ", ".join(_dist_name(m) for m in self.marginals)
        return f"JointDistribution(copula={self.copula!r}, marginals=[{ms}])"

    # -- basics ---------------------------------------------------------------
    @property
    def is_continuous(self) -> bool:
        """Whether all margins are continuous (then the copula is unique)."""
        return not any(self._discrete)

    def marginal(self, i: int) -> Any:
        """The ``i``-th marginal distribution (0-based)."""
        return self.marginals[i]

    def transform(self, x: Any) -> np.ndarray:
        r"""Probability integral transform :math:`u_i = F_i(x_i)` of points."""
        X = np.atleast_2d(np.asarray(x, dtype=float))
        return np.column_stack([self.marginals[i].cdf(X[:, i]) for i in range(self.dim)])

    def inverse_transform(self, u: Any) -> np.ndarray:
        r"""Quantile transform :math:`x_i = F_i^{-1}(u_i)`."""
        U = np.atleast_2d(np.asarray(u, dtype=float))
        return np.column_stack([self.marginals[i].ppf(U[:, i]) for i in range(self.dim)])

    def _require_continuous(self, idx: Sequence[int], what: str) -> None:
        bad = [i for i in idx if self._discrete[i]]
        if bad:
            raise ValueError(
                f"{what} needs continuous margins, but margin(s) {bad} are discrete: the copula "
                "is then not unique and the joint distribution has no density "
                "(Genest & Nešlehová, 2007)."
            )

    # -- distribution functions ---------------------------------------------------
    def cdf(self, *args) -> float | np.ndarray:
        r""":math:`H(x)=C(F_1(x_1),\dots,F_d(x_d))` (``(N, d)`` array, point or coordinates)."""
        X, shape, scalar = parse_points(args, self.dim, "cdf")
        return finish(self._C._cdf_clean(self.transform(X)), shape, scalar)

    def sf(self, *args) -> float | np.ndarray:
        r"""Joint survival function :math:`P(X_1>x_1,\dots,X_d>x_d)=\bar C(F_1(x_1),\dots)`."""
        X, shape, scalar = parse_points(args, self.dim, "sf")
        U = np.clip(self.transform(X), 0.0, 1.0)
        return finish(np.clip(self._C._survival(U), 0.0, 1.0), shape, scalar)

    survival_function = sf

    def logpdf(self, *args) -> float | np.ndarray:
        r"""Log-density :math:`\log c(F(x)) + \sum_i\log f_i(x_i)` (continuous margins)."""
        self._require_continuous(range(self.dim), "logpdf")
        X, shape, scalar = parse_points(args, self.dim, "logpdf")
        with np.errstate(all="ignore"):
            lm = np.column_stack([self.marginals[i].logpdf(X[:, i]) for i in range(self.dim)])
            out = self._C._logpdf_clean(self.transform(X)) + lm.sum(axis=1)
        out = np.where(np.isnan(out), -np.inf, out)
        return finish(out, shape, scalar)

    def pdf(self, *args) -> float | np.ndarray:
        r"""Density :math:`h(x)=c(F_1(x_1),\dots,F_d(x_d))\prod_i f_i(x_i)`."""
        if not self._C.is_absolutely_continuous:
            from copul.exceptions import PropertyUnavailableException

            raise PropertyUnavailableException("the copula has no density.")
        val = self.logpdf(*args)
        with np.errstate(over="ignore"):
            return float(np.exp(val)) if np.ndim(val) == 0 else np.exp(val)

    def pmf(self, *args) -> float | np.ndarray:
        r"""Probability mass :math:`P(X=x)` for discrete margins.

        :math:`P(X=x) = V_C\bigl([F(x^-), F(x)]\bigr)`, the :math:`C`-volume of
        the box between the left limits :math:`F_i(x_i^-)=F_i(x_i)-P(X_i=x_i)`
        and :math:`F_i(x_i)` (Genest & Nešlehová, 2007).
        """
        if not all(self._discrete):
            raise ValueError("pmf needs discrete margins (use pdf for continuous ones).")
        X, shape, scalar = parse_points(args, self.dim, "pmf")
        hi = self.transform(X)
        mass = np.column_stack(
            [np.asarray(self.marginals[i].pmf(X[:, i]), float) for i in range(self.dim)]
        )
        lo = np.clip(hi - mass, 0.0, 1.0)
        out = box_volume(self._C._cdf_clean, lo, np.clip(hi, 0.0, 1.0))
        return finish(np.clip(out, 0.0, 1.0), shape, scalar)

    def rectangle_probability(self, a: Any, b: Any) -> float | np.ndarray:
        r""":math:`P(a<X\le b)=V_H((a,b])=V_C\bigl([F(a),F(b)]\bigr)`.

        The :math:`H`-volume of the box (Nelsen, 2006, §2.3 and §2.10);
        ``a`` and ``b`` are points of length ``d`` or ``(N, d)`` arrays and may
        contain :math:`\pm\infty`.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        scalar = a.ndim == 1 and b.ndim == 1
        a2, b2 = np.broadcast_arrays(np.atleast_2d(a), np.atleast_2d(b))
        if a2.shape[-1] != self.dim:
            raise ValueError(f"corners must have length {self.dim}.")
        if np.any(a2 > b2):
            raise ValueError("need a <= b componentwise.")
        lo = np.clip(self.transform(a2), 0.0, 1.0)
        hi = np.clip(self.transform(b2), 0.0, 1.0)
        out = np.clip(box_volume(self._C._cdf_clean, lo, hi), 0.0, 1.0)
        return float(out[0]) if scalar else out

    h_volume = rectangle_probability

    def rvs(self, n: int = 1, random_state: Any = None) -> np.ndarray:
        r"""``n`` samples: :math:`U\sim C`, then :math:`X_i=F_i^{-1}(U_i)`
        (Nelsen, 2006, §2.9)."""
        U = self._C.rvs(n, random_state=random_state)
        return self.inverse_transform(U) if n > 0 else np.empty((0, self.dim))

    # -- conditional distributions (pairs) ------------------------------------------
    def _pair(self, i: int, j: int):
        if i == j:
            raise ValueError("i and j must differ.")
        return self._C.margin(i, j)

    def conditional_cdf(self, y: Any, x: Any, i: int = 0, j: int = 1):
        r""":math:`P(X_j\le y\mid X_i=x) = \partial_1C_{ij}\bigl(F_i(x),F_j(y)\bigr)`.

        :math:`C_{ij}` is the copula of :math:`(X_i, X_j)` (Nelsen, 2006,
        §2.2 and §2.9).
        """
        self._require_continuous((i, j), "conditional_cdf")
        C = self._pair(i, j)
        u = np.asarray(self.marginals[i].cdf(np.asarray(x, float)), float)
        v = np.asarray(self.marginals[j].cdf(np.asarray(y, float)), float)
        return C.cond_distr_1(u, v)

    def conditional_ppf(self, q: Any, x: Any, i: int = 0, j: int = 1):
        r"""Conditional quantile :math:`F_j^{-1}\bigl(\partial_1C_{ij}^{-1}(F_i(x), q)\bigr)`
        of :math:`X_j` given :math:`X_i=x` (copula quantile regression curve,
        Bouyé & Salmon, 2009)."""
        self._require_continuous((i, j), "conditional_ppf")
        C = self._pair(i, j)
        u = np.asarray(self.marginals[i].cdf(np.asarray(x, float)), float)
        w = C.cond_distr_1_inv(u, np.asarray(q, float))
        return self.marginals[j].ppf(w)

    def conditional_pdf(self, y: Any, x: Any, i: int = 0, j: int = 1):
        r"""Conditional density :math:`c_{ij}(F_i(x),F_j(y))\,f_j(y)` of :math:`X_j\mid X_i=x`."""
        self._require_continuous((i, j), "conditional_pdf")
        C = self._pair(i, j)
        y = np.asarray(y, float)
        u = np.asarray(self.marginals[i].cdf(np.asarray(x, float)), float)
        v = np.asarray(self.marginals[j].cdf(y), float)
        return np.asarray(C.pdf(u, v), float) * self.marginals[j].pdf(y)

    def regression(
        self, x: Any, kind: str | float = "mean", i: int = 0, j: int = 1, n_nodes: int = 64
    ):
        r"""Regression curve of :math:`X_j` on :math:`X_i`.

        Parameters
        ----------
        x : float or array_like
            Values of :math:`X_i`.
        kind : "mean", "median" or float in (0, 1)
            * ``"mean"``: :math:`E[X_j\mid X_i=x]=\int_0^1F_j^{-1}\bigl(
              \partial_1C_{ij}^{-1}(F_i(x),w)\bigr)\,dw`, computed with
              ``n_nodes``-point Gauss--Hermite quadrature after the substitution
              :math:`w=\Phi(z)` (exact if the integrand is linear in :math:`z`,
              e.g. Gaussian copula with normal margins);
            * ``"median"`` or a level :math:`q`: the conditional quantile
              curve (median regression, Nelsen, 2006, §2.9; quantile
              regression, Bouyé & Salmon, 2009).
        i, j : int
            Regressor and response components (0-based).
        n_nodes : int
            Quadrature nodes for ``"mean"``.
        """
        if isinstance(kind, str) and kind.lower() == "median":
            kind = 0.5
        if not isinstance(kind, str):
            q = float(kind)
            if not 0.0 < q < 1.0:
                raise ValueError("a quantile level must lie in (0, 1).")
            return self.conditional_ppf(q, x, i=i, j=j)
        if kind.lower() != "mean":
            raise ValueError("kind must be 'mean', 'median' or a level in (0, 1).")
        self._require_continuous((i, j), "regression")
        C = self._pair(i, j)
        x = np.asarray(x, float)
        u = np.asarray(self.marginals[i].cdf(x), float).reshape(-1)
        z, w = np.polynomial.hermite_e.hermegauss(int(n_nodes))
        w = w / np.sqrt(2.0 * np.pi)
        lev = np.clip(ndtr(z), _W_EPS, 1.0 - _W_EPS)
        U, W = np.meshgrid(u, lev, indexing="ij")
        vq = np.asarray(C.cond_distr_1_inv(U.ravel(), W.ravel()), float)
        # keep the extreme nodes (weights below 1e-20) away from infinite quantiles
        vq = np.clip(vq, 1e-300, 1.0 - _W_EPS)
        vals = np.asarray(self.marginals[j].ppf(vq), float).reshape(U.shape)
        out = vals @ w
        return float(out[0]) if x.ndim == 0 else out.reshape(x.shape)

    # -- moments -----------------------------------------------------------------------
    def mean(self) -> np.ndarray:
        """Marginal means."""
        return np.array([float(m.mean()) for m in self.marginals])

    def var(self) -> np.ndarray:
        """Marginal variances."""
        return np.array([float(m.var()) for m in self.marginals])

    def _scale(self, i: int, s: np.ndarray):
        r"""``x(s) = F_i^{-1}(\Phi(s))`` and ``dx/ds = \varphi(s)/f_i(x)``."""
        m = self.marginals[i]
        lower = s <= 0
        x = np.where(
            lower,
            m.ppf(ndtr(np.where(lower, s, 0.0))),
            m.isf(ndtr(-np.where(lower, 0.0, s))) if hasattr(m, "isf") else m.ppf(ndtr(s)),
        )
        with np.errstate(all="ignore"):
            logjac = -0.5 * s * s - _LOG_SQRT_2PI - m.logpdf(x)
        return x, np.exp(logjac)

    def _hoeffding_cov(self, i: int, j: int, n_nodes: int, half_width: float) -> float:
        C = self._pair(i, j)
        g, w = np.polynomial.legendre.leggauss(int(n_nodes))
        s = half_width * g
        w = half_width * w
        _, ji = self._scale(i, s)
        _, jj = self._scale(j, s)
        u = ndtr(s)
        Uu, Vv = np.meshgrid(u, u, indexing="ij")
        D = np.asarray(C.cdf(Uu.ravel(), Vv.ravel()), float).reshape(Uu.shape) - Uu * Vv
        integrand = D * (ji * w)[:, None] * (jj * w)[None, :]
        return float(np.nansum(integrand))

    def covariance(
        self,
        method: str = "hoeffding",
        n_nodes: int = 200,
        half_width: float = 8.5,
        n_samples: int = 1_000_000,
        random_state: Any = 0,
    ) -> np.ndarray:
        r"""Covariance matrix of :math:`X`.

        ``method="hoeffding"`` (default, continuous margins) uses Hoeffding's
        (1940) covariance formula

        .. math::

           \operatorname{Cov}(X_i,X_j) = \int\!\!\int_{\mathbb R^2}
           \bigl[H_{ij}(x,y)-F_i(x)F_j(y)\bigr]\,dx\,dy
           = \int\!\!\int\bigl[C_{ij}(\Phi(s),\Phi(t))-\Phi(s)\Phi(t)\bigr]
           \frac{dx}{ds}\frac{dy}{dt}\,ds\,dt

        (Nelsen, 2006, §5.1), with :math:`x=F_i^{-1}(\Phi(s))`, evaluated by
        tensor Gauss--Legendre quadrature with ``n_nodes`` nodes per axis on
        :math:`[-\text{half\_width}, \text{half\_width}]^2`.  ``method="mc"``
        uses ``n_samples`` simulated observations (any margins).  The
        variances are the marginal variances.
        """
        d = self.dim
        method = method.lower()
        if method == "mc":
            X = self.rvs(int(n_samples), random_state=random_state)
            return np.cov(X, rowvar=False)
        if method != "hoeffding":
            raise ValueError("method must be 'hoeffding' or 'mc'.")
        self._require_continuous(range(d), "covariance(method='hoeffding')")
        out = np.diag(self.var()).astype(float)
        for a in range(d):
            for b in range(a + 1, d):
                out[a, b] = out[b, a] = self._hoeffding_cov(a, b, n_nodes, half_width)
        return out

    def correlation(self, method: str = "pearson", **kwargs) -> np.ndarray:
        r"""Correlation matrix.

        Parameters
        ----------
        method : {"pearson", "spearman", "kendall"}
            * ``"pearson"``: :math:`\operatorname{Cov}(X_i,X_j)/(\sigma_i\sigma_j)`
              from :meth:`covariance` (depends on the margins, Embrechts,
              McNeil & Straumann, 2002);
            * ``"spearman"`` / ``"kendall"``: the copula's pairwise Spearman's
              rho / Kendall's tau (margin-free for continuous margins).
        **kwargs
            Passed to :meth:`covariance`.
        """
        method = method.lower()
        if method in ("spearman", "rho"):
            return self._C.spearmans_rho_matrix()
        if method in ("kendall", "tau"):
            return self._C.kendalls_tau_matrix()
        if method != "pearson":
            raise ValueError("method must be 'pearson', 'spearman' or 'kendall'.")
        S = self.covariance(**kwargs)
        sd = np.sqrt(np.diag(S))
        out = S / np.outer(sd, sd)
        np.fill_diagonal(out, 1.0)
        return out

    # -- estimation -----------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        data: Any,
        copula_family: Any,
        marginals: Sequence[Any] | Any = None,
        method: str = "ifm",
        copula_method: str = "mle",
        **copula_kwargs: Any,
    ) -> JointDistribution:
        r"""Fit a copula model :math:`H=C_\theta(F_1,\dots,F_d)` to data.

        Parameters
        ----------
        data : array_like of shape (n, d)
        copula_family :
            * bivariate data: any family accepted by :func:`copul.stats.fit`
              (``cp.Clayton``, ``"GumbelHougaard"``, ``cp.StudentT(nu=4)``, ...);
            * :math:`d`-dimensional: :class:`~copul.multivariate.GaussianND`,
              :class:`~copul.multivariate.StudentTND` or an Archimedean family
              name (``"clayton"``, ``"gumbel"``, ``"frank"``, ``"joe"``,
              ``"amh"``) -- fitted with their ``fit`` methods.
        marginals : sequence, optional
            Per column: an unfitted continuous SciPy distribution
            (``scipy.stats.norm``; fitted by maximum likelihood, ``dist.fit``),
            a frozen distribution
            (kept fixed) or ``"empirical"`` (:class:`EmpiricalMarginal`).  A
            single entry is used for all columns; default ``"empirical"``.
        method : {"ifm", "cml"}
            * ``"ifm"`` -- inference functions for margins (Joe & Xu, 1996;
              Joe, 2005): the copula is fitted to
              :math:`\hat U_{kj}=\hat F_j(x_{kj})` with the fitted parametric
              margins (empirical columns use ranks);
            * ``"cml"`` -- canonical maximum likelihood (Genest, Ghoudi &
              Rivest, 1995): the copula is fitted to the rank
              pseudo-observations :math:`R_{kj}/(n+1)`, whatever the margins.
        copula_method : str
            ``method`` of :func:`copul.stats.fit` (``"mle"``, ``"itau"``, ...)
            for bivariate families, or of the :math:`d`-dimensional ``fit``.
        **copula_kwargs
            Further arguments of the copula fit.

        Returns
        -------
        JointDistribution
            With ``fit_info`` holding the copula fit result, the marginal
            parameters, the method and the log-likelihood (continuous margins;
            the probability transforms are clipped to
            :math:`[10^{-12}, 1-10^{-12}]`, since maximum likelihood estimates
            of location/threshold parameters can put data on the boundary).
        """
        from copul.stats._utils import as_data
        from copul.stats.pseudo_obs import pseudo_obs

        X = as_data(data, min_dim=2)
        n, d = X.shape
        if marginals is None:
            marginals = "empirical"
        if isinstance(marginals, (str,)) or not isinstance(marginals, (list, tuple)):
            marginals = [marginals] * d
        if len(marginals) != d:
            raise ValueError(f"need {d} marginal specifications, got {len(marginals)}.")
        fitted, mparams = [], []
        for j, spec in enumerate(marginals):
            col = X[:, j]
            if isinstance(spec, str):
                if spec.lower() != "empirical":
                    raise ValueError(f"unknown marginal specification {spec!r}.")
                fitted.append(EmpiricalMarginal(col))
                mparams.append(None)
            elif isinstance(spec, stats.rv_continuous):
                params = spec.fit(col)
                fitted.append(spec(*params))
                mparams.append(tuple(float(p) for p in params))
            else:
                _check_marginal(spec, j)
                fitted.append(spec)
                mparams.append(getattr(spec, "args", None))
        method = method.lower()
        R = pseudo_obs(X)
        if method == "cml":
            U = R
        elif method == "ifm":
            U = R.copy()
            for j, m in enumerate(fitted):
                if not isinstance(m, EmpiricalMarginal):
                    U[:, j] = m.cdf(X[:, j])
            U = np.clip(U, 1e-12, 1.0 - 1e-12)
        else:
            raise ValueError("method must be 'ifm' or 'cml'.")
        copula, fit_result = _fit_copula(copula_family, U, d, copula_method, copula_kwargs)
        H = cls(copula, fitted)
        loglik = np.nan
        if H.is_continuous and H._C.is_absolutely_continuous:
            # copula part at the (clipped) fitted probability transforms
            Uf = np.clip(H.transform(X), 1e-12, 1.0 - 1e-12)
            with np.errstate(all="ignore"):
                lm = sum(np.asarray(m.logpdf(X[:, j]), float) for j, m in enumerate(fitted))
                loglik = float(np.sum(H._C._logpdf_clean(Uf) + lm))
        H.fit_info = {
            "method": method,
            "copula_fit": fit_result,
            "marginal_params": mparams,
            "loglik": loglik,
            "n": n,
        }
        return H


def _dist_name(m: Any) -> str:
    dist = getattr(m, "dist", None)
    if dist is not None:
        args = ", ".join(f"{a:g}" for a in getattr(m, "args", ()))
        kw = ", ".join(f"{k}={v:g}" for k, v in getattr(m, "kwds", {}).items())
        return f"{dist.name}({', '.join(p for p in (args, kw) if p)})"
    return repr(m)


def _fit_copula(family: Any, U: np.ndarray, d: int, method: str, kwargs: dict):
    from copul.multivariate.archimedean import _FAMILIES, ArchimedeanCopulaND
    from copul.multivariate.elliptical import GaussianND, StudentTND

    if (
        isinstance(family, str)
        and family.lower().replace("-", "_") in _FAMILIES
        and (d > 2 or kwargs.pop("nd", False))
    ):
        m = method if method in ("mle", "itau") else "mle"
        C = ArchimedeanCopulaND.fit(U, family, method=m, pseudo_obs=False, **kwargs)
        return C, C
    if family in (GaussianND, "GaussianND"):
        m = method if method in ("itau", "irho", "normal_scores") else "itau"
        C = GaussianND.fit(U, method=m, pseudo_obs=False)
        return C, C
    if family in (StudentTND, "StudentTND"):
        C = StudentTND.fit(U, pseudo_obs=False, **kwargs)
        return C, C
    if d != 2:
        raise ValueError(
            "for d > 2 use GaussianND, StudentTND or an Archimedean family name "
            "('clayton', 'gumbel', 'frank', 'joe', 'amh')."
        )
    from copul.stats import fit as stats_fit

    res = stats_fit(family, U, method=method, pseudo_obs=False, **kwargs)
    return res.copula, res


# ---------------------------------------------------------------------------
# copula of a joint distribution
# ---------------------------------------------------------------------------


def _accepts(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _as_points_fn(fn: Callable, signature: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``f(x, y)`` (vectorized) from a callable with the SciPy ``(N, 2)`` convention.

    SciPy's quasi-Monte Carlo cdfs (``random_state=``/``rng=`` keyword) are
    called with a fixed seed so that the copula is deterministic.
    """
    if signature == "xy":
        return lambda x, y: np.asarray(fn(x, y), dtype=float)
    if signature != "points":
        raise ValueError("signature must be 'points' or 'xy'.")
    has_rs = _accepts(fn, "random_state")
    has_rng = _accepts(fn, "rng")

    def f(x, y):
        pts = np.column_stack([np.ravel(x), np.ravel(y)])
        if has_rs:
            out = fn(pts, random_state=0)
        elif has_rng:
            out = fn(pts, rng=np.random.default_rng(0))
        else:
            out = fn(pts)
        return np.asarray(out, dtype=float).reshape(np.shape(x))

    return f


class SklarCopula(NumericBivCopula):
    r"""Copula :math:`C(u,v)=H\bigl(F^{-1}(u),G^{-1}(v)\bigr)` of a continuous
    bivariate distribution (Nelsen, 2006, Cor. 2.3.7); see
    :func:`copula_from_joint`.

    ``cond_distr_1``/``cond_distr_2`` integrate the density
    :math:`c(u,v)=h(x,y)/(f(x)g(y))` (96-point Gauss--Legendre quadrature in
    normal scores) when a joint density is available, otherwise they are
    finite differences of the cdf; sampling transforms joint samples
    :math:`(X,Y)` to :math:`(F(X),G(Y))` when a joint sampler is available,
    otherwise it inverts the conditional distribution.
    """

    def __init__(
        self,
        joint_cdf: Callable[[np.ndarray, np.ndarray], np.ndarray],
        ppfs: Sequence[Callable],
        cdfs: Sequence[Callable] | None = None,
        joint_pdf: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        pdfs: Sequence[Callable] | None = None,
        joint_rvs: Callable[[int, np.random.Generator], np.ndarray] | None = None,
        name: str = "SklarCopula",
    ) -> None:
        self._H = joint_cdf
        self._ppfs = tuple(ppfs)
        self._cdfs = None if cdfs is None else tuple(cdfs)
        self._h = joint_pdf
        self._pdfs = None if pdfs is None else tuple(pdfs)
        self._joint_rvs = joint_rvs
        self._name = name
        super().__init__()

    def __repr__(self) -> str:
        return f"{self._name}()"

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._h is not None and self._pdfs is not None

    def _xy(self, u, v):
        return np.asarray(self._ppfs[0](u), float), np.asarray(self._ppfs[1](v), float)

    def _cdf(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        out = np.where(u >= 1.0, v, np.where(v >= 1.0, u, 0.0)).astype(float)
        inner = (u > 0) & (v > 0) & (u < 1) & (v < 1)
        if np.any(inner):
            x, y = self._xy(u[inner], v[inner])
            out[inner] = self._H(x, y)
        return out

    def _pdf(self, u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        x, y = self._xy(u, v)
        with np.errstate(all="ignore"):
            return self._h(x, y) / (self._pdfs[0](x) * self._pdfs[1](y))

    def _h_quad(self, a, b, first: bool):
        r""":math:`\int_0^b c(a, t)\,dt` (``first``) or :math:`\int_0^b c(t, a)\,dt`."""
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        shape = a.shape
        a, b = a.ravel(), b.ravel()
        g, w = np.polynomial.legendre.leggauss(96)
        lo = -9.0
        hi = ndtri(np.clip(b, 1e-300, 1.0 - 1e-16))
        half = 0.5 * (hi - lo)
        z = lo + half[:, None] * (g[None, :] + 1.0)
        t = ndtr(z)
        A = np.broadcast_to(a[:, None], t.shape)
        with np.errstate(all="ignore"):
            c = self._pdf(A, t) if first else self._pdf(t, A)
            dens = np.where(np.isfinite(c), c, 0.0) * np.exp(-0.5 * z * z - _LOG_SQRT_2PI)
        out = half * (dens @ w)
        out = np.where(b <= 0, 0.0, np.where(b >= 1, 1.0, out))
        return np.clip(out, 0.0, 1.0).reshape(shape)

    def _h1(self, u, v):
        if self.is_absolutely_continuous:
            return self._h_quad(u, v, True)
        return super()._h1(u, v)

    def _h2(self, u, v):
        if self.is_absolutely_continuous:
            return self._h_quad(v, u, False)
        return super()._h2(u, v)

    def kendalls_tau(self, *args, **kwargs):
        r"""Kendall's :math:`\tau = 4\int\!\!\int C\,c\,du\,dv - 1` (Nelsen, 2006,
        Thm. 5.1.3), by 160-point tensor Gauss--Legendre quadrature in normal
        scores when the density is available (otherwise the numerical
        engine is used)."""
        if not self.is_absolutely_continuous:
            raise NotImplementedError("no density: use the numerical engine")
        g, w = np.polynomial.legendre.leggauss(160)
        s = 8.5 * g
        w = 8.5 * w * np.exp(-0.5 * s * s - _LOG_SQRT_2PI)
        u = ndtr(s)
        U, V = np.meshgrid(u, u, indexing="ij")
        with np.errstate(all="ignore"):
            f = self.cdf_vectorized(U, V) * self._pdf(U, V)
        return float(4.0 * (w @ np.nan_to_num(f) @ w) - 1.0)

    def _rvs(self, n, rng):
        if self._joint_rvs is not None and self._cdfs is not None:
            XY = np.asarray(self._joint_rvs(n, rng), float).reshape(n, 2)
            return np.column_stack([self._cdfs[0](XY[:, 0]), self._cdfs[1](XY[:, 1])])
        from copul.measures.backend import numeric_backend

        be = numeric_backend(self)
        u = rng.random(n)
        return np.column_stack([u, be.h1_inv(u, rng.random(n))])


def copula_from_joint(
    joint_cdf: Any,
    marginal_cdfs: Sequence[Callable] | None = None,
    marginal_ppfs: Sequence[Callable] | None = None,
    *,
    marginals: Sequence[Any] | None = None,
    joint_pdf: Callable | None = None,
    marginal_pdfs: Sequence[Callable] | None = None,
    joint_rvs: Callable | None = None,
    signature: str = "points",
) -> SklarCopula:
    r"""Copula of a continuous bivariate distribution (Sklar's theorem).

    .. math::

       C(u,v) = H\bigl(F^{-1}(u), G^{-1}(v)\bigr)

    (Nelsen, 2006, Cor. 2.3.7).  The result is a copul bivariate copula
    (:class:`SklarCopula`, a
    :class:`~copul.family.constructions.NumericBivCopula`) with the full
    numerical API, sampling and all dependence measures.

    Parameters
    ----------
    joint_cdf : callable or SciPy frozen multivariate distribution
        :math:`H`.  A callable takes an ``(N, 2)`` array of points (SciPy
        convention, e.g. ``scipy.stats.multivariate_normal(...).cdf``) or,
        with ``signature="xy"``, two arrays ``(x, y)``.  A distribution object
        (with ``cdf`` and optionally ``pdf``/``rvs``) supplies the joint cdf,
        density and sampler at once.
    marginal_cdfs, marginal_ppfs : pair of callables
        :math:`F, G` and their quantile functions :math:`F^{-1}, G^{-1}`
        (the cdfs are only needed for sampling from a joint sampler).
    marginals : pair of SciPy frozen distributions, optional
        Alternative to the callables (supplies cdf, ppf and pdf).
    joint_pdf : callable, optional
        Joint density :math:`h` (same convention as ``joint_cdf``); with the
        marginal densities it gives the copula density
        :math:`c(u,v)=h(x,y)/(f(x)g(y))` and accurate conditional
        distributions.
    marginal_pdfs : pair of callables, optional
    joint_rvs : callable, optional
        ``joint_rvs(n, rng)`` returning an ``(n, 2)`` sample of :math:`H`.
    signature : {"points", "xy"}
        Calling convention of ``joint_cdf``/``joint_pdf``.

    Returns
    -------
    SklarCopula

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import copul as cp
    >>> from copul.sklar import copula_from_joint
    >>> H = stats.multivariate_normal([1.0, -2.0], [[4.0, 1.2], [1.2, 1.0]])
    >>> C = copula_from_joint(H, marginals=[stats.norm(1, 2), stats.norm(-2, 1)])
    >>> bool(np.isclose(C.cdf(0.3, 0.6), cp.Gaussian(0.6).cdf(0.3, 0.6)))
    True
    """
    if marginals is not None:
        if len(marginals) != 2:
            raise ValueError("copula_from_joint needs exactly two marginals.")
        marginal_cdfs = marginal_cdfs or [m.cdf for m in marginals]
        marginal_ppfs = marginal_ppfs or [m.ppf for m in marginals]
        if marginal_pdfs is None and all(hasattr(m, "pdf") for m in marginals):
            marginal_pdfs = [m.pdf for m in marginals]
    if marginal_ppfs is None or len(marginal_ppfs) != 2:
        raise ValueError("two marginal quantile functions (marginal_ppfs or marginals) are needed.")
    name = "SklarCopula"
    if hasattr(joint_cdf, "cdf"):  # a (SciPy) distribution object
        dist = joint_cdf
        name = f"SklarCopula({type(dist).__name__})"
        if joint_pdf is None and hasattr(dist, "pdf"):
            joint_pdf = dist.pdf
        if joint_rvs is None and hasattr(dist, "rvs"):

            def joint_rvs(n, rng, _d=dist):
                return _d.rvs(size=n, random_state=rng)

        joint_cdf = dist.cdf
        signature = "points"
    H = _as_points_fn(joint_cdf, signature)
    h = None if joint_pdf is None else _as_points_fn(joint_pdf, signature)
    return SklarCopula(
        H,
        marginal_ppfs,
        cdfs=marginal_cdfs,
        joint_pdf=h,
        pdfs=marginal_pdfs,
        joint_rvs=joint_rvs,
        name=name,
    )
