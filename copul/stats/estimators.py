r"""
Rank-based sample estimators of the dependence measures of :mod:`copul.measures`.

Every function takes two samples ``x, y`` of equal length (raw data or
pseudo-observations -- all estimators are rank based and hence invariant
under strictly increasing marginal transformations) and returns a ``float``.
:data:`SAMPLE_ESTIMATORS` maps the canonical measure keys of
:mod:`copul.measures.registry` to these functions; use
:func:`sample_measure` or :class:`copul.stats.EmpiricalCopula` for keyed
access.

Notation: :math:`R_i, S_i` are the (mid-)ranks of :math:`x_i, y_i`,
:math:`\hat U_i = R_i/n`, :math:`\hat V_i = S_i/n`, and
:math:`C_n(u,v) = \frac1n\sum_i 1\{\hat U_i\le u, \hat V_i\le v\}` is the
empirical copula (Deheuvels, 1979).  All estimators are consistent for the
population measure of the copula of :math:`(X, Y)` (for continuous margins).

References
----------
* Blest, D. C. (2000). Rank correlation -- an alternative measure.
  *Aust. N. Z. J. Stat.* 42, 101--111.
* Blomqvist, N. (1950). On a measure of dependence between two random
  variables. *Ann. Math. Statist.* 21, 593--600.
* Blum, J. R., Kiefer, J. and Rosenblatt, M. (1961). Distribution free tests
  of independence based on the sample distribution function.
  *Ann. Math. Statist.* 32, 485--498.
* Capéraà, P., Fougères, A.-L. and Genest, C. (1997). A nonparametric
  estimation procedure for bivariate extreme value copulas. *Biometrika*
  84, 567--577.
* Chatterjee, S. (2021). A new coefficient of correlation. *JASA* 116,
  2009--2022.
* Frahm, G., Junker, M. and Schmidt, R. (2005). Estimating the
  tail-dependence coefficient: properties and pitfalls. *Insurance Math.
  Econom.* 37, 80--100.
* Gaißer, S., Ruppert, M. and Schmid, F. (2010). A multivariate version of
  Hoeffding's Phi-Square. *J. Multivariate Anal.* 101, 2571--2586.
* Genest, C. and Plante, J.-F. (2003). On Blest's measure of rank
  correlation. *Canad. J. Statist.* 31, 35--52.
* Genest, C., Nešlehová, J. and Ben Ghorbal, N. (2010). Spearman's footrule
  and Gini's gamma: a review with complements. *J. Nonparametr. Stat.* 22,
  937--954.
* Hoeffding, W. (1948). A non-parametric test of independence.
  *Ann. Math. Statist.* 19, 546--557.
* Knight, W. R. (1966). A computer method for calculating Kendall's tau with
  ungrouped data. *JASA* 61, 436--439.
* Kraskov, A., Stögbauer, H. and Grassberger, P. (2004). Estimating mutual
  information. *Phys. Rev. E* 69, 066138.
* Schmidt, R. and Stadtmüller, U. (2006). Non-parametric estimation of tail
  dependence. *Scand. J. Statist.* 33, 307--335.
* Schweizer, B. and Wolff, E. F. (1981). On nonparametric measures of
  dependence for random variables. *Ann. Statist.* 9, 879--885.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats
from scipy.special import digamma

from copul.chatterjee import xi_ncalculate
from copul.measures.numeric import lp_constant
from copul.measures.registry import resolve_key
from copul.stats._utils import RandomLike, chunk_size, dominance_counts, grid_counts_blocks

__all__ = [
    "SAMPLE_ESTIMATORS",
    "cramer_von_mises_independence",
    "sample_beta",
    "sample_bkr",
    "sample_footrule",
    "sample_gamma",
    "sample_hoeffdings_d",
    "sample_kappa",
    "sample_lambda_l",
    "sample_lambda_u",
    "sample_lp",
    "sample_measure",
    "sample_mutual_information",
    "sample_nu",
    "sample_rho",
    "sample_sigma",
    "sample_tau",
    "sample_xi",
    "sample_xi_2",
]


def _pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length, got {x.size} and {y.size}.")
    if x.size < 2:
        raise ValueError("at least two observations are required.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must be finite.")
    return x, y


def _midranks(x, y) -> tuple[np.ndarray, np.ndarray]:
    return stats.rankdata(x), stats.rankdata(y)


# ---------------------------------------------------------------------------
# concordance measures
# ---------------------------------------------------------------------------


def sample_tau(x, y, variant: str = "b") -> float:
    r"""Kendall's :math:`\tau_n` in :math:`O(n\log n)` (Knight, 1966).

    .. math::

       \tau_n = \frac{2}{n(n-1)}\sum_{i<j}
       \operatorname{sgn}(x_i - x_j)\operatorname{sgn}(y_i - y_j)

    (``variant="a"``); the default ``variant="b"`` is Kendall's tie-corrected
    :math:`\tau_b` (identical to :math:`\tau_a` without ties).
    """
    x, y = _pair(x, y)
    if variant not in ("a", "b"):
        raise ValueError("variant must be 'a' or 'b'")
    res = stats.kendalltau(x, y, variant="b")
    tb = float(res.statistic if hasattr(res, "statistic") else res[0])
    if variant == "b" or not np.isfinite(tb):
        return tb
    n = x.size
    n0 = n * (n - 1) / 2.0

    def tie_pairs(a):
        _, c = np.unique(a, return_counts=True)
        return float(np.sum(c * (c - 1) / 2.0))

    return float(tb * np.sqrt((n0 - tie_pairs(x)) * (n0 - tie_pairs(y))) / n0)


def sample_rho(x, y) -> float:
    r"""Spearman's :math:`\rho_n`: Pearson correlation of the mid-ranks.

    Without ties :math:`\rho_n = 1 - \frac{6\sum_i (R_i - S_i)^2}{n(n^2-1)}`.
    """
    x, y = _pair(x, y)
    r, s = _midranks(x, y)
    r = r - r.mean()
    s = s - s.mean()
    den = np.sqrt(np.dot(r, r) * np.dot(s, s))
    return float(np.dot(r, s) / den) if den > 0 else float("nan")


def sample_footrule(x, y) -> float:
    r"""Spearman's footrule :math:`\psi_n = 1 - \frac{3}{n^2-1}\sum_i |R_i - S_i|`.

    Exactly 1 for comonotone samples (Genest, Nešlehová & Ben Ghorbal, 2010).
    """
    x, y = _pair(x, y)
    r, s = _midranks(x, y)
    n = x.size
    return float(1.0 - 3.0 * np.sum(np.abs(r - s)) / (n * n - 1.0))


def sample_gamma(x, y) -> float:
    r"""Gini's :math:`\gamma_n = \frac{1}{\lfloor n^2/2\rfloor}\sum_i
    \bigl(|R_i + S_i - n - 1| - |R_i - S_i|\bigr)`.

    Equals :math:`\pm1` for co-/countermonotone samples (Genest, Nešlehová
    & Ben Ghorbal, 2010).
    """
    x, y = _pair(x, y)
    r, s = _midranks(x, y)
    n = x.size
    return float(np.sum(np.abs(r + s - n - 1.0) - np.abs(r - s)) / np.floor(n * n / 2.0))


def sample_beta(x, y) -> float:
    r"""Blomqvist's :math:`\beta_n = 4\,C_n(\tfrac12, \tfrac12) - 1`.

    Uses the pseudo-observations :math:`R_i/(n+1)`, i.e. counts the points
    with both coordinates at most the sample medians (Blomqvist, 1950;
    Schmid & Schmidt, 2007).
    """
    x, y = _pair(x, y)
    r, s = _midranks(x, y)
    n = x.size
    half = (n + 1) / 2.0
    c = np.count_nonzero((r <= half) & (s <= half)) / n
    return float(4.0 * c - 1.0)


def sample_nu(x, y) -> float:
    r"""Blest's :math:`\nu_n` (Genest & Plante, 2003).

    .. math::

       \nu_n = \frac{2n+1}{n-1} - \frac{12}{n^2-n}
       \sum_i \Bigl(1 - \frac{R_i}{n+1}\Bigr)^2 S_i ,

    consistent for :math:`\nu(C) = 24\iint (1-u)\,C(u,v)\,du\,dv - 2`
    (``x`` plays the role of the first copula argument :math:`u`) and exactly
    1 for comonotone samples.
    """
    x, y = _pair(x, y)
    r, s = _midranks(x, y)
    n = x.size
    return float(
        (2.0 * n + 1.0) / (n - 1.0) - 12.0 / (n * n - n) * np.sum((1.0 - r / (n + 1.0)) ** 2 * s)
    )


# ---------------------------------------------------------------------------
# Chatterjee's xi
# ---------------------------------------------------------------------------


def sample_xi(x, y, random_state: RandomLike = None) -> float:
    r"""Chatterjee's :math:`\xi_n(X, Y)` estimating
    :math:`\xi(C) = 6\iint(\partial_1 C)^2 - 2` (``Y`` explained by ``X``).

    Thin wrapper of :func:`copul.chatterjee.xi_ncalculate` (Chatterjee, 2021);
    ties in ``x`` are broken at random with ``random_state``.
    """
    x, y = _pair(x, y)
    return float(xi_ncalculate(x, y, random_state=random_state))


def sample_xi_2(x, y, random_state: RandomLike = None) -> float:
    r"""Chatterjee's :math:`\xi_n(Y, X)` estimating :math:`\xi_2(C) = \xi(C^\top)`."""
    x, y = _pair(x, y)
    return float(xi_ncalculate(y, x, random_state=random_state))


# ---------------------------------------------------------------------------
# distances to independence (empirical copula integrals)
# ---------------------------------------------------------------------------


def _max_ranks(x, y) -> tuple[np.ndarray, np.ndarray]:
    return (
        stats.rankdata(x, method="max").astype(np.int64),
        stats.rankdata(y, method="max").astype(np.int64),
    )


def _count_blocks(x, y):
    """Blocks of :math:`N_{ij} = n C_n(i/n, j/n)` (ties handled by max-ranks)."""
    r, s = _max_ranks(x, y)
    n = r.size
    order = np.argsort(r, kind="stable")
    r_sorted, s_sorted = r[order], s[order]
    step = chunk_size(n)
    carry = np.zeros(n, dtype=np.int64)
    if np.unique(r).size == n and np.unique(s).size == n:
        yield from grid_counts_blocks(r, s)
        return
    for i0 in range(0, n, step):
        i1 = min(i0 + step, n)
        inc = np.zeros((i1 - i0, n), dtype=np.int64)
        lo, hi = np.searchsorted(r_sorted, [i0 + 1, i1 + 1], side="left")
        np.add.at(inc, (r_sorted[lo:hi] - 1 - i0, s_sorted[lo:hi] - 1), 1)
        np.cumsum(inc, axis=1, out=inc)
        np.cumsum(inc, axis=0, out=inc)
        inc += carry[None, :]
        carry = inc[-1].copy()
        yield i0, inc


def _phi2_sums(u: np.ndarray, v: np.ndarray) -> float:
    r""":math:`\iint (C_n(a,b) - ab)^2\,da\,db` for the empirical copula of
    points ``(u_i, v_i)`` (closed form of Gaißer, Ruppert & Schmid, 2010)."""
    n = u.size
    step = chunk_size(n)
    s2 = 0.0
    for s in range(0, n, step):
        a = 1.0 - np.maximum(u[s : s + step, None], u[None, :])
        b = 1.0 - np.maximum(v[s : s + step, None], v[None, :])
        s2 += float(np.einsum("ij,ij->", a, b))
    s2 /= n * n
    s1 = float(np.mean((1.0 - u * u) * (1.0 - v * v))) / 4.0
    return s2 - 2.0 * s1 + 1.0 / 9.0


def sample_hoeffdings_d(x, y) -> float:
    r"""Hoeffding's :math:`\Phi^2_n = 90\iint (C_n(u,v) - uv)^2\,du\,dv`.

    With :math:`\hat U_i = R_i/n` the integral has the closed form
    (Gaißer, Ruppert & Schmid, 2010)

    .. math::

       \Phi^2_n = 90\Bigl[\frac1{n^2}\sum_{i,j}(1-\hat U_i\vee\hat U_j)
       (1-\hat V_i\vee\hat V_j) - \frac{1}{2n}\sum_i(1-\hat U_i^2)(1-\hat V_i^2)
       + \frac19\Bigr],

    evaluated in :math:`O(n^2)` time by chunks (no :math:`n\times n`
    storage).  Registry key ``"hoeffdings_d"`` (the dependence index
    :math:`\Phi^2`; Hoeffding's :math:`D` statistic is :func:`sample_bkr`).
    """
    x, y = _pair(x, y)
    n = x.size
    u = stats.rankdata(x, method="max") / n
    v = stats.rankdata(y, method="max") / n
    return float(90.0 * _phi2_sums(u, v))


def cramer_von_mises_independence(x, y) -> float:
    r"""Deheuvels' Cramér--von Mises statistic
    :math:`I_n = n\iint (C_n(u,v) - uv)^2\,du\,dv = n\,\Phi^2_n / 90`
    (Deheuvels, 1981; Genest & Rémillard, 2004)."""
    x, y = _pair(x, y)
    return x.size * sample_hoeffdings_d(x, y) / 90.0


def _grid_abs_sum(x, y, p: float) -> float:
    r""":math:`\sum_{i,j=1}^n |C_n(i/n,j/n) - ij/n^2|^p`."""
    n = x.size
    j = np.arange(1, n + 1, dtype=float)
    tot = 0.0
    for i0, blk in _count_blocks(x, y):
        i = np.arange(i0 + 1, i0 + 1 + blk.shape[0], dtype=float)
        d = np.abs(blk / n - np.outer(i, j) / (n * n))
        tot += float(np.sum(d if p == 1 else d**p))
    return tot


def sample_sigma(x, y) -> float:
    r"""Schweizer--Wolff :math:`\sigma_n` (Schweizer & Wolff, 1981).

    .. math::

       \sigma_n = \frac{12}{n^2-1}\sum_{i=1}^n\sum_{j=1}^n
       \Bigl|C_n\bigl(\tfrac in, \tfrac jn\bigr) - \frac{ij}{n^2}\Bigr| ,

    which is exactly 1 for co- and countermonotone samples.  :math:`O(n^2)`
    time, chunked memory.
    """
    x, y = _pair(x, y)
    n = x.size
    return float(12.0 / (n * n - 1.0) * _grid_abs_sum(x, y, 1.0))


def sample_lp(x, y, p: float = 2.0) -> float:
    r""":math:`L^p` distance :math:`\delta_{p,n} = \frac{k(p)}{n^2}\sum_{i,j}
    |C_n(i/n, j/n) - ij/n^2|^p` (Riemann sum on the rank grid; ``p=1`` is
    :math:`\sigma` up to the finite-sample factor :math:`n^2/(n^2-1)`)."""
    x, y = _pair(x, y)
    n = x.size
    return float(lp_constant(p) / (n * n) * _grid_abs_sum(x, y, float(p)))


def sample_kappa(x, y) -> float:
    r"""Uniform distance :math:`\kappa_n = 4\sup_{u,v}|C_n(u,v) - uv|`.

    The supremum is computed exactly: :math:`C_n` is constant on the cells
    :math:`[\frac in,\frac{i+1}n)\times[\frac jn,\frac{j+1}n)`, where
    :math:`uv` ranges over :math:`[\frac{ij}{n^2}, \frac{(i+1)(j+1)}{n^2})`.
    This is the Kolmogorov--Smirnov type statistic of Blum, Kiefer &
    Rosenblatt (1961) (Deheuvels, 1981).
    """
    x, y = _pair(x, y)
    n = x.size
    nn = float(n * n)
    j = np.arange(1, n + 1, dtype=float)
    best = 1.0 / n  # cells with i = 0 or j = 0 (C_n = 0 there)
    for i0, blk in _count_blocks(x, y):
        i = np.arange(i0 + 1, i0 + 1 + blk.shape[0], dtype=float)
        c = blk / n
        interior_i = i < n
        lo = np.outer(i, j) / nn
        hi = np.outer(np.where(interior_i, i + 1, i), np.where(j < n, j + 1, j)) / nn
        best = max(best, float(np.max(np.abs(c - lo))), float(np.max(np.abs(c - hi))))
    return float(min(1.0, 4.0 * best))


def sample_bkr(x, y) -> float:
    r"""Hoeffding's :math:`D_n` statistic, an unbiased estimator of the
    Blum--Kiefer--Rosenblatt coefficient :math:`B = 30\int (C - uv)^2\,dC`.

    With mid-ranks :math:`R_i, S_i` and bivariate ranks :math:`Q_i` (number of
    points dominated by :math:`(x_i, y_i)`, plus one, tie corrected)

    .. math::

       D_n = 30\,\frac{(n-2)(n-3)D_1 + D_2 - 2(n-2)D_3}{n(n-1)(n-2)(n-3)(n-4)},

    :math:`D_1 = \sum (Q_i-1)(Q_i-2)`,
    :math:`D_2 = \sum (R_i-1)(R_i-2)(S_i-1)(S_i-2)`,
    :math:`D_3 = \sum (R_i-2)(S_i-2)(Q_i-1)` (Hoeffding, 1948).  Being
    unbiased it may be slightly negative under independence.  Requires
    :math:`n\ge5`; :math:`O(n^2)` time.
    """
    x, y = _pair(x, y)
    n = x.size
    if n < 5:
        raise ValueError("Hoeffding's D requires at least 5 observations.")
    r, s = _midranks(x, y)
    q = 1.0 + dominance_counts(x, y, ties="hoeffding")
    d1 = np.sum((q - 1.0) * (q - 2.0))
    d2 = np.sum((r - 1.0) * (r - 2.0) * (s - 1.0) * (s - 2.0))
    d3 = np.sum((r - 2.0) * (s - 2.0) * (q - 1.0))
    num = (n - 2.0) * (n - 3.0) * d1 + d2 - 2.0 * (n - 2.0) * d3
    den = n * (n - 1.0) * (n - 2.0) * (n - 3.0) * (n - 4.0)
    return float(30.0 * num / den)


# ---------------------------------------------------------------------------
# tail dependence
# ---------------------------------------------------------------------------


def _default_k(n: int) -> int:
    return max(1, int(np.floor(np.sqrt(n))))


def _cfg_upper(u: np.ndarray, v: np.ndarray) -> float:
    lu, lv = np.log(1.0 / u), np.log(1.0 / v)
    lm = np.log(1.0 / np.maximum(u, v) ** 2)
    return float(2.0 - 2.0 * np.exp(np.mean(np.log(np.sqrt(lu * lv) / lm))))


def _tail(x, y, lower: bool, method: str, k: int | None) -> float:
    x, y = _pair(x, y)
    n = x.size
    r, s = _midranks(x, y)
    if method in ("ss", "schmidt_stadtmueller", "schmidt-stadtmueller", "empirical"):
        k = _default_k(n) if k is None else int(k)
        if not 1 <= k <= n:
            raise ValueError(f"k must be in 1..{n}, got {k}.")
        if lower:
            cnt = np.count_nonzero((r <= k) & (s <= k))
        else:
            cnt = np.count_nonzero((r > n - k) & (s > n - k))
        return float(cnt / k)
    if method == "cfg":
        u, v = r / (n + 1.0), s / (n + 1.0)
        if lower:
            u, v = 1.0 - u, 1.0 - v
        return float(np.clip(_cfg_upper(u, v), 0.0, 1.0))
    raise ValueError(f"unknown tail estimator {method!r}; use 'ss' or 'cfg'.")


def sample_lambda_l(x, y, method: str = "ss", k: int | None = None) -> float:
    r"""Lower tail-dependence coefficient :math:`\lambda_L`.

    * ``method="ss"`` (default): Schmidt & Stadtmüller (2006),
      :math:`\hat\lambda_L = \frac nk\,C_n(\tfrac kn, \tfrac kn) =
      \frac1k\#\{i : R_i\le k,\ S_i\le k\}` with ``k`` (default
      :math:`\lfloor\sqrt n\rfloor`) the number of tail observations;
    * ``method="cfg"``: the estimator based on the Capéraà--Fougères--Genest
      Pickands estimator applied to :math:`(1-\hat U, 1-\hat V)`, valid for
      (survival) extreme-value copulas (Frahm, Junker & Schmidt, 2005).
    """
    return _tail(x, y, True, method, k)


def sample_lambda_u(x, y, method: str = "ss", k: int | None = None) -> float:
    r"""Upper tail-dependence coefficient :math:`\lambda_U`.

    * ``method="ss"`` (default): Schmidt & Stadtmüller (2006),
      :math:`\hat\lambda_U = \frac1k\#\{i : R_i > n-k,\ S_i > n-k\}`;
    * ``method="cfg"``: for extreme-value copulas (Frahm, Junker & Schmidt,
      2005, based on Capéraà, Fougères & Genest, 1997),

      .. math::

         \hat\lambda_U = 2 - 2\exp\Bigl\{\frac1n\sum_i\log
         \frac{\sqrt{\log(1/\hat U_i)\log(1/\hat V_i)}}
         {\log\bigl(1/\max(\hat U_i,\hat V_i)^2\bigr)}\Bigr\},
         \quad \hat U_i = \frac{R_i}{n+1}.
    """
    return _tail(x, y, False, method, k)


# ---------------------------------------------------------------------------
# mutual information
# ---------------------------------------------------------------------------


def sample_mutual_information(x, y, k: int = 3, random_state: RandomLike = None) -> float:
    r"""Kraskov--Stögbauer--Grassberger (2004) :math:`k`-nearest-neighbour
    estimator (algorithm 1) of the mutual information
    :math:`I = \iint c\log c` (= minus the copula entropy), applied to the
    pseudo-observations:

    .. math::

       \hat I = \psi(k) + \psi(n) - \frac1n\sum_i
       \bigl[\psi(n_{x,i}+1) + \psi(n_{y,i}+1)\bigr],

    with :math:`\varepsilon_i` the max-norm distance to the :math:`k`-th
    neighbour and :math:`n_{x,i}` the number of points with
    :math:`|\hat U_j - \hat U_i| < \varepsilon_i`.  Ties are broken at random.
    """
    from scipy.spatial import cKDTree

    from copul.stats.pseudo_obs import pseudo_obs

    x, y = _pair(x, y)
    n = x.size
    k = int(k)
    if not 1 <= k < n:
        raise ValueError(f"k must be in 1..{n - 1}")
    u = pseudo_obs(np.column_stack([x, y]), ties="random", random_state=random_state)
    tree = cKDTree(u)
    dist, _ = tree.query(u, k=k + 1, p=np.inf)
    eps = dist[:, -1]
    su, sv = np.sort(u[:, 0]), np.sort(u[:, 1])
    nx = np.searchsorted(su, u[:, 0] + eps, side="left") - np.searchsorted(
        su, u[:, 0] - eps, side="right"
    )
    ny = np.searchsorted(sv, u[:, 1] + eps, side="left") - np.searchsorted(
        sv, u[:, 1] - eps, side="right"
    )
    nx = np.maximum(nx - 1, 0)
    ny = np.maximum(ny - 1, 0)
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

#: canonical measure key -> sample estimator ``f(x, y, **options)``
SAMPLE_ESTIMATORS: dict[str, Callable[..., float]] = {
    "xi": sample_xi,
    "xi_2": sample_xi_2,
    "rho": sample_rho,
    "tau": sample_tau,
    "footrule": sample_footrule,
    "gamma": sample_gamma,
    "beta": sample_beta,
    "nu": sample_nu,
    "hoeffdings_d": sample_hoeffdings_d,
    "sigma": sample_sigma,
    "kappa": sample_kappa,
    "lp": sample_lp,
    "bkr": sample_bkr,
    "mutual_information": sample_mutual_information,
    "lambda_l": sample_lambda_l,
    "lambda_u": sample_lambda_u,
}


def _filter_kwargs(f: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    names = set(inspect.signature(f).parameters)
    return {k: v for k, v in kwargs.items() if k in names}


def sample_measure(x, y, key: str, **options: Any) -> float:
    """Sample estimate of the measure ``key`` (canonical key or alias).

    Options not accepted by the respective estimator (e.g. ``k`` for
    ``"rho"``) are ignored, so a common option dict can be passed for several
    measures.
    """
    k = resolve_key(key)
    f = SAMPLE_ESTIMATORS.get(k)
    if f is None:  # pragma: no cover - all registry keys are covered
        raise KeyError(f"No sample estimator for {key!r}.")
    return f(x, y, **_filter_kwargs(f, options))
