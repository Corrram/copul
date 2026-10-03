r"""
Exchangeability and radial symmetry of bivariate copulas.

A copula :math:`C` is **exchangeable** (symmetric) if
:math:`C(u,v)=C^\top(u,v):=C(v,u)`, i.e. :math:`(U,V)\overset{d}{=}(V,U)`,
and **radially symmetric** if :math:`C=\hat C` with the survival copula
:math:`\hat C(u,v)=u+v-1+C(1-u,1-v)`, i.e.
:math:`(U,V)\overset{d}{=}(1-U,1-V)` (Nelsen 2006, Sect. 2.7).

Non-exchangeability
-------------------
For every copula and :math:`u\le v`,

.. math::

   |C(u,v)-C(v,u)| \le \min(u,\,1-v,\,v-u) \le \tfrac13

(Klement & Mesiar 2006; Nelsen 2007): the first two bounds are
:math:`C\le M` and :math:`C(u,v)-C(v,u) = P(U>v,V\le u)-P(U\le u,V>v)`,
the third is the Lipschitz property.  The bound :math:`\tfrac13` is
attained at :math:`(\tfrac13,\tfrac23)` by the shuffle of :math:`M`
returned by :func:`maximally_nonexchangeable_copula`.  Hence

.. math::

   \mu_\infty(C) = 3\sup_{(u,v)\in[0,1]^2}|C(u,v)-C(v,u)| \in [0,1]

is a normalized measure of non-exchangeability in the sense of Durante,
Klement, Sempi & Úbeda-Flores (2010).  For finite :math:`p` the
:math:`L^p` distance :math:`\|C-C^\top\|_p` is reported *without*
normalization (it is at most :math:`\tfrac13`, but this is not sharp).

Radial asymmetry
----------------
Dehgani, Dolati & Úbeda-Flores (2013) introduce axioms for measures of
radial asymmetry.  :func:`radial_asymmetry` returns the plain distance
:math:`\sup|C-\hat C|` (or :math:`\|C-\hat C\|_p`) without a normalizing
constant.

Tests from data
---------------
:func:`exchangeability_test` and :func:`radial_symmetry_test` compare the
empirical copula :math:`C_n` of the pseudo-observations with the empirical
copula of the transformed sample, :math:`(V_i,U_i)` resp.
:math:`(1-U_i,1-V_i)`, through the Cramér–von Mises statistic

.. math::

   S_n = \sum_{i=1}^n \bigl\{C_n(\hat U_i,\hat V_i) - C_n^{T}(\hat U_i,\hat V_i)\bigr\}^2
       = n\int\!\!\int (C_n - C_n^{T})^2\,dC_n

of Genest, Nešlehová & Quessy (2012) (exchangeability) and Genest &
Nešlehová (2014) (radial symmetry).  p-values come from the multiplier
bootstrap of the empirical copula process (Rémillard & Scaillet 2009),
applied jointly to both empirical copulas with the same multipliers, with
partial derivatives estimated by finite differences of bandwidth
:math:`n^{-1/2}`.  Continuous data (no ties) are assumed.

References
----------
Dehgani, A., Dolati, A. & Úbeda-Flores, M. (2013). Measures of radial
asymmetry for bivariate random vectors. *Statistical Papers* 54, 271–286.

Durante, F., Klement, E. P., Sempi, C. & Úbeda-Flores, M. (2010). Measures
of non-exchangeability for bivariate random vectors. *Statistical Papers*
51, 687–699.

Genest, C. & Nešlehová, J. G. (2014). On tests of radial symmetry for
bivariate copulas. *Statistical Papers* 55, 1107–1119.

Genest, C., Nešlehová, J. & Quessy, J.-F. (2012). Tests of symmetry for
bivariate copulas. *Annals of the Institute of Statistical Mathematics*
64, 811–834.

Klement, E. P. & Mesiar, R. (2006). How non-symmetric can a copula be?
*Commentationes Mathematicae Universitatis Carolinae* 47, 141–148.

Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Sect. 2.7.

Nelsen, R. B. (2007). Extremes of nonexchangeability. *Statistical Papers*
48, 329–336.

Rémillard, B. & Scaillet, O. (2009). Testing for equality between two
copulas. *Journal of Multivariate Analysis* 100, 377–386.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from copul.measures.quadrature import integrate_2d
from copul.theory.bounds import ShuffleOfM
from copul.theory.quasi import cdf_function

__all__ = [
    "SymmetryTestResult",
    "exchangeability_test",
    "is_exchangeable",
    "is_radially_symmetric",
    "maximally_nonexchangeable_copula",
    "nonexchangeability",
    "radial_asymmetry",
    "radial_symmetrize",
    "radial_symmetry_test",
    "symmetrize",
]


# ---------------------------------------------------------------------------
# sup / L^p norms of differences of copulas
# ---------------------------------------------------------------------------


def _grid_points(C: Any, m: int) -> np.ndarray:
    """Grid ``{k/m}`` merged with the grid of a checkerboard copula (if any)."""
    pts = [np.linspace(0.0, 1.0, int(m) + 1)]
    matr = getattr(C, "matr", None)
    if matr is not None and np.ndim(matr) == 2:
        for k in np.shape(matr):
            if k <= 2000:
                pts.append(np.linspace(0.0, 1.0, int(k) + 1))
    return np.unique(np.concatenate(pts))


def _sup_abs(
    diff: Callable, g: np.ndarray, refine: bool, n_starts: int = 6
) -> tuple[float, tuple[float, float]]:
    """``sup |diff(u, v)|`` over the unit square: grid search + local refinement."""
    U, V = np.meshgrid(g, g, indexing="ij")
    Z = np.abs(diff(U, V))
    flat = Z.ravel()
    k = int(np.argmax(flat))
    best, loc = float(flat[k]), (float(U.ravel()[k]), float(V.ravel()[k]))
    if not refine or best == 0.0:
        return best, loc
    from scipy.optimize import minimize

    def obj(x):
        x = np.clip(x, 0.0, 1.0)
        return -float(np.abs(diff(np.array([x[0]]), np.array([x[1]])))[0])

    order = np.argsort(flat)[::-1][: int(n_starts)]
    step = float(np.max(np.diff(g)))
    for idx in order:
        x0 = np.array([U.ravel()[idx], V.ravel()[idx]])
        res = minimize(
            obj,
            x0,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-14, "initial_simplex": _simplex(x0, step)},
        )
        val = -float(res.fun)
        if val > best:
            best, loc = val, (float(np.clip(res.x[0], 0, 1)), float(np.clip(res.x[1], 0, 1)))
    return best, loc


def _simplex(x0: np.ndarray, step: float) -> np.ndarray:
    s = 0.5 * step
    su = s if x0[0] + s <= 1.0 else -s
    sv = s if x0[1] + s <= 1.0 else -s
    pts = np.array([x0, x0 + np.array([su, 0.0]), x0 + np.array([0.0, sv])])
    return np.clip(pts, 0.0, 1.0)


def _lp_norm(diff: Callable, p: float) -> float:
    p = float(p)
    if p < 1.0:
        raise ValueError("p must be >= 1 (or np.inf)")
    val, _ = integrate_2d(lambda u, v: np.abs(diff(u, v)) ** p, atol=1e-13, rtol=1e-9)
    return float(max(val, 0.0) ** (1.0 / p))


def _exact_on_grid(C: Any) -> bool:
    r"""Whether the supremum of :math:`|C-C^\top|` and :math:`|C-\hat C|` is
    attained on the checkerboard grid.

    True for :class:`~copul.BivCheckPi` (the differences are bilinear on the
    cells of the merged grid) and for square checkerboards whose cells all
    carry the same :math:`M` (or all the same :math:`W`) kernel (piecewise
    linear on the two triangles of each cell, which transposition and
    reflection map onto triangles of the same orientation).
    """
    matr = getattr(C, "matr", None)
    if matr is None or np.ndim(matr) != 2 or not hasattr(C, "_kernel_signs"):
        return False
    try:
        signs = C._kernel_signs()
    except Exception:
        return False
    if signs is None or not np.any(signs):
        return True
    signs = np.asarray(signs)
    square = np.shape(matr)[0] == np.shape(matr)[1]
    return bool(square and np.all(signs == signs.flat[0]))


def _measure(diff, C, p, m, refine, return_location, scale):
    if np.isinf(p):
        best, loc = _sup_abs(diff, _grid_points(C, m), refine and not _exact_on_grid(C))
        val = min(scale * best, 1.0)
        return (val, loc) if return_location else val
    val = _lp_norm(diff, p)
    return (val, None) if return_location else val


def nonexchangeability(
    C: Any,
    p: float = np.inf,
    m: int = 240,
    refine: bool = True,
    return_location: bool = False,
):
    r"""Degree of non-exchangeability of a copula.

    For ``p = np.inf`` (default) the normalized measure

    .. math::

       \mu_\infty(C) = 3\sup_{(u,v)\in[0,1]^2}|C(u,v)-C(v,u)|\in[0,1]

    (Klement & Mesiar 2006; Nelsen 2007; Durante et al. 2010); for finite
    :math:`p\ge1` the unnormalized :math:`L^p` distance

    .. math::

       \|C-C^\top\|_p = \Bigl(\int_0^1\!\!\int_0^1|C(u,v)-C(v,u)|^p\,du\,dv\Bigr)^{1/p}
       \in[0,\tfrac13].

    Parameters
    ----------
    C : copula, NumericQuasiCopula or callable
    p : float
        ``np.inf`` or :math:`p\ge1`.
    m : int
        Grid for the supremum (merged with the grid of a checkerboard
        copula, on which the supremum is attained for square checkerboards).
    refine : bool
        Refine the grid maximum by local optimization (Nelder–Mead).
    return_location : bool
        Also return a maximizer :math:`(u,v)` (``None`` for finite ``p``).

    Returns
    -------
    float or (float, tuple)

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.symmetry import nonexchangeability, maximally_nonexchangeable_copula
    >>> nonexchangeability(cp.Clayton(2))
    0.0
    >>> round(nonexchangeability(maximally_nonexchangeable_copula()), 12)
    1.0
    """
    f = cdf_function(C)
    return _measure(lambda u, v: f(u, v) - f(v, u), C, p, m, refine, return_location, 3.0)


def radial_asymmetry(
    C: Any,
    p: float = np.inf,
    m: int = 240,
    refine: bool = True,
    return_location: bool = False,
):
    r"""Degree of radial asymmetry of a copula.

    For ``p = np.inf`` (default) the distance

    .. math::

       \sup_{(u,v)}|C(u,v)-\hat C(u,v)|,
       \qquad \hat C(u,v)=u+v-1+C(1-u,1-v),

    and for finite :math:`p\ge1` the :math:`L^p` distance
    :math:`\|C-\hat C\|_p`; both vanish iff :math:`C` is radially
    symmetric (see Dehgani, Dolati & Úbeda-Flores 2013 for axioms of
    measures of radial asymmetry).

    Parameters
    ----------
    C : copula, NumericQuasiCopula or callable
    p, m, refine, return_location
        See :func:`nonexchangeability`.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.symmetry import radial_asymmetry
    >>> radial_asymmetry(cp.Frank(4)) < 1e-10
    True
    >>> radial_asymmetry(cp.Clayton(3)) > 0.03
    True
    """
    f = cdf_function(C)

    def diff(u, v):
        return f(u, v) - (u + v - 1.0 + f(1.0 - u, 1.0 - v))

    return _measure(diff, C, p, m, refine, return_location, 1.0)


def is_exchangeable(C: Any, tol: float = 1e-8, m: int = 240) -> bool:
    r"""Whether :math:`\sup|C-C^\top|\le` ``tol`` (grid search with refinement)."""
    return bool(nonexchangeability(C, m=m) / 3.0 <= tol)


def is_radially_symmetric(C: Any, tol: float = 1e-8, m: int = 240) -> bool:
    r"""Whether :math:`\sup|C-\hat C|\le` ``tol`` (grid search with refinement)."""
    return bool(radial_asymmetry(C, m=m) <= tol)


# ---------------------------------------------------------------------------
# extremal examples and symmetrizations
# ---------------------------------------------------------------------------


def maximally_nonexchangeable_copula(transpose: bool = False) -> ShuffleOfM:
    r"""A copula with maximal non-exchangeability :math:`\mu_\infty = 1`.

    The shuffle of :math:`M` with :math:`V = U+\tfrac13` for
    :math:`U<\tfrac23` and :math:`V=U-\tfrac23` otherwise,

    .. math::

       C(u,v) = \max\bigl\{0,\min(u, v-\tfrac13)\bigr\}
              + \max\bigl\{0,\min(u-\tfrac23, v)\bigr\},

    has :math:`C(\tfrac13,\tfrac23)=\tfrac13` and
    :math:`C(\tfrac23,\tfrac13)=0`, so it attains the bound
    :math:`|C(u,v)-C(v,u)|\le\tfrac13` of Klement & Mesiar (2006) and
    Nelsen (2007).  ``transpose=True`` returns :math:`C^\top`.
    """
    third = 1.0 / 3.0
    if transpose:
        return ShuffleOfM([(0.0, 2 * third, third, 1), (third, 0.0, 2 * third, 1)])
    return ShuffleOfM([(0.0, third, 2 * third, 1), (2 * third, 0.0, third, 1)])


def symmetrize(C: Any):
    r"""The exchangeable copula :math:`\tfrac12(C + C^\top)`.

    Returns a :class:`~copul.family.constructions.MixtureCopula` of
    :math:`C` and :math:`C^\top` (equal weights).
    """
    from copul.family.constructions import mixture, transpose

    return mixture([C, transpose(C)], [0.5, 0.5])


def radial_symmetrize(C: Any):
    r"""The radially symmetric copula :math:`\tfrac12(C + \hat C)`.

    Returns a :class:`~copul.family.constructions.MixtureCopula` of
    :math:`C` and its survival copula :math:`\hat C` (equal weights).
    """
    from copul.family.constructions import mixture, survival

    return mixture([C, survival(C)], [0.5, 0.5])


# ---------------------------------------------------------------------------
# tests from data
# ---------------------------------------------------------------------------


@dataclass
class SymmetryTestResult:
    """Result of :func:`exchangeability_test` / :func:`radial_symmetry_test`.

    Attributes
    ----------
    statistic : float
        Observed Cramér–von Mises statistic.
    pvalue : float
        Multiplier-bootstrap p-value ``(1 + #{S* >= S}) / (n_boot + 1)``.
    method : str
    n : int
        Sample size.
    n_boot : int
        Number of multiplier replicates.
    extra : dict
        Bootstrap replicates (``"bootstrap"``) and options.
    """

    statistic: float
    pvalue: float
    method: str
    n: int
    n_boot: int
    extra: dict = field(default_factory=dict, repr=False)

    def reject(self, alpha: float = 0.05) -> bool:
        """Whether the null hypothesis of symmetry is rejected at level ``alpha``."""
        return bool(self.pvalue < alpha)


def _ecdf(Z: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Empirical copula of the sample ``Z`` at the points ``(x, y)``."""
    return np.mean((Z[:, 0, None] <= x[None, :]) & (Z[:, 1, None] <= y[None, :]), axis=0)


def _multiplier_process(Z, x, y, Xi, h):
    """Multiplier replicates of the empirical copula process of ``Z`` at ``(x, y)``.

    Rémillard & Scaillet (2009): with centred multipliers
    :math:`\\xi_i-\\bar\\xi`, the replicate is
    :math:`n^{-1/2}\\sum_i(\\xi_i-\\bar\\xi)[1(Z_i\\le(x,y)) - \\partial_1 C_n\\,1(Z_{i1}\\le x)
    - \\partial_2 C_n\\,1(Z_{i2}\\le y)]`.
    """
    n = Z.shape[0]
    Iu = Z[:, 0, None] <= x[None, :]
    Iv = Z[:, 1, None] <= y[None, :]
    I = (Iu & Iv).astype(float)
    xl, xr = np.clip(x - h, 0.0, 1.0), np.clip(x + h, 0.0, 1.0)
    yl, yr = np.clip(y - h, 0.0, 1.0), np.clip(y + h, 0.0, 1.0)
    d1 = (_ecdf(Z, xr, y) - _ecdf(Z, xl, y)) / (xr - xl)
    d2 = (_ecdf(Z, x, yr) - _ecdf(Z, x, yl)) / (yr - yl)
    d1 = np.clip(d1, 0.0, 1.0)
    d2 = np.clip(d2, 0.0, 1.0)
    s = 1.0 / np.sqrt(n)
    return s * (Xi @ I - d1 * (Xi @ Iu.astype(float)) - d2 * (Xi @ Iv.astype(float)))


def _symmetry_test(data, transform, n_boot, statistic, m, random_state, method):
    from copul.stats._utils import as_data, as_rng
    from copul.stats.pseudo_obs import pseudo_obs

    X = as_data(data, min_dim=2, max_dim=2)
    S = pseudo_obs(X)
    T = transform(S)
    n = S.shape[0]
    rng = as_rng(random_state)
    if statistic in ("Sn", "S", "cvm"):
        px, py = S[:, 0], S[:, 1]
        w = np.full(n, 1.0)  # S_n = sum_i D_n(U_i, V_i)^2 (n * integral w.r.t. C_n)
        wb = np.full(n, 1.0 / n)
    elif statistic in ("Rn", "R"):
        g = (np.arange(int(m)) + 0.5) / int(m)
        px, py = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
        w = np.full(px.size, n / px.size)
        wb = np.full(px.size, 1.0 / px.size)
    else:
        raise ValueError("statistic must be 'Sn' or 'Rn'")
    n_boot = int(n_boot)
    Xi = rng.standard_normal((n_boot, n))
    Xi -= Xi.mean(axis=1, keepdims=True)
    h = 1.0 / np.sqrt(n)
    stat = 0.0
    boot = np.zeros(n_boot)
    chunk = max(1, 2_000_000 // max(n, 1))
    for start in range(0, px.size, chunk):
        sl = slice(start, start + chunk)
        x, y = px[sl], py[sl]
        D = _ecdf(S, x, y) - _ecdf(T, x, y)
        stat += float(np.sum(w[sl] * D**2))
        Dk = _multiplier_process(S, x, y, Xi, h) - _multiplier_process(T, x, y, Xi, h)
        boot += (Dk**2) @ wb[sl]
    pvalue = (1.0 + float(np.sum(boot >= stat))) / (n_boot + 1.0)
    return SymmetryTestResult(
        statistic=stat,
        pvalue=pvalue,
        method=method,
        n=n,
        n_boot=n_boot,
        extra={"bootstrap": boot, "statistic_type": statistic},
    )


def exchangeability_test(
    data: Any,
    n_boot: int = 1000,
    statistic: str = "Sn",
    m: int = 20,
    random_state: Any = None,
) -> SymmetryTestResult:
    r"""Test of exchangeability :math:`H_0: C(u,v)=C(v,u)` from a bivariate sample.

    Statistic (Genest, Nešlehová & Quessy 2012)

    .. math::

       S_n = \sum_{i=1}^n\{C_n(\hat U_i,\hat V_i)-C_n(\hat V_i,\hat U_i)\}^2

    (``statistic="Sn"``) or
    :math:`R_n = n\int\!\!\int\{C_n(u,v)-C_n(v,u)\}^2\,du\,dv` on an
    :math:`m\times m` midpoint grid (``statistic="Rn"``), with the
    multiplier bootstrap of Rémillard & Scaillet (2009) for the p-value.

    Parameters
    ----------
    data : array_like of shape (n, 2)
        Sample (continuous margins; ranks are taken internally).
    n_boot : int
        Number of multiplier replicates.
    statistic : {"Sn", "Rn"}
    m : int
        Grid size for ``"Rn"``.
    random_state : int, Generator or None

    Returns
    -------
    SymmetryTestResult

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.symmetry import exchangeability_test
    >>> X = cp.Clayton(2).rvs(300, random_state=1)
    >>> exchangeability_test(X, n_boot=200, random_state=0).pvalue > 0.05
    True
    """
    return _symmetry_test(
        data,
        lambda S: S[:, ::-1],
        n_boot,
        statistic,
        m,
        random_state,
        "exchangeability (Genest-Neslehova-Quessy 2012, multiplier bootstrap)",
    )


def radial_symmetry_test(
    data: Any,
    n_boot: int = 1000,
    statistic: str = "Sn",
    m: int = 20,
    random_state: Any = None,
) -> SymmetryTestResult:
    r"""Test of radial symmetry :math:`H_0: C=\hat C` from a bivariate sample.

    Compares the empirical copula :math:`C_n` of the pseudo-observations
    with the empirical copula :math:`\hat C_n` of
    :math:`(1-\hat U_i, 1-\hat V_i)` (Genest & Nešlehová 2014):

    .. math::

       S_n = \sum_{i=1}^n\{C_n(\hat U_i,\hat V_i)-\hat C_n(\hat U_i,\hat V_i)\}^2,

    or the grid version ``"Rn"``; p-values from the multiplier bootstrap
    (Rémillard & Scaillet 2009) applied jointly to :math:`C_n` and
    :math:`\hat C_n`.

    Parameters
    ----------
    data, n_boot, statistic, m, random_state
        See :func:`exchangeability_test`.

    Returns
    -------
    SymmetryTestResult
    """
    return _symmetry_test(
        data,
        lambda S: 1.0 - S,
        n_boot,
        statistic,
        m,
        random_state,
        "radial symmetry (Genest-Neslehova 2014, multiplier bootstrap)",
    )
