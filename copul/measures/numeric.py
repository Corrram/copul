r"""
Pure numerical formulas for bivariate dependence measures.

These functions only need *vectorized callables* (or grids) describing a
copula and are therefore usable without any copula class -- e.g. for copulas
that are only known through their h-functions (Markov kernels).

Callable conventions
--------------------
All callables are evaluated elementwise on NumPy arrays of equal shape and
only at interior points of :math:`(0,1)^2`:

* ``cdf(u, v)``  -- the copula :math:`C(u,v)`;
* ``h1(u, v)``   -- :math:`\partial_1 C(u,v) = P(V\le v \mid U=u)`;
* ``h2(u, v)``   -- :math:`\partial_2 C(u,v) = P(U\le u \mid V=v)`;
* ``pdf(u, v)``  -- the density :math:`c(u,v)` (absolutely continuous part).

Grid convention (``measures_from_h`` with an ``ndarray``)
---------------------------------------------------------
``h[i, j] = \partial_1 C(u_i, v_j)`` with cell midpoints
:math:`u_i=(i+\tfrac12)/m`, :math:`v_j=(j+\tfrac12)/n` (rows index
:math:`u`, columns index :math:`v`).  Integrals are then evaluated by the
midpoint rule, :math:`C` by cumulative (trapezoidal) summation of ``h`` in
:math:`u` and :math:`\partial_2 C` by central differences in :math:`v`;
the accuracy is :math:`O(m^{-2}+n^{-2})` for smooth copulas.

Every ``*_from_*`` function returns a ``float`` or, with
``full_output=True``, a tuple ``(value, error_estimate)``.  Tolerances refer
to the measure itself.

Examples
--------
>>> import numpy as np
>>> from copul.measures.numeric import xi_from_h, rho_from_cdf
>>> th = 0.5  # FGM copula
>>> h1 = lambda u, v: v + th * v * (1 - v) * (1 - 2 * u)
>>> round(xi_from_h(h1), 10) == round(th**2 / 15, 10)
True
>>> round(rho_from_cdf(lambda u, v: u * v * (1 + th * (1 - u) * (1 - v))), 10)
0.1666666667
"""

from __future__ import annotations

import contextvars
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from scipy.special import beta as _beta_fn

from copul.measures.quadrature import integrate_1d, integrate_2d
from copul.measures.registry import _iter_keys, get_measure, resolve_key

__all__ = [
    "NumericCopula",
    "beta_from_cdf",
    "bkr_from_h",
    "evaluate",
    "footrule_from_cdf",
    "gamma_from_cdf",
    "hoeffdings_d_from_cdf",
    "kappa_from_cdf",
    "lambda_l_from_cdf",
    "lambda_l_from_h",
    "lambda_u_from_cdf",
    "lambda_u_from_h",
    "lp_constant",
    "lp_from_cdf",
    "measures_from_cdf",
    "measures_from_h",
    "mutual_information_from_pdf",
    "nu_from_cdf",
    "nu_from_h",
    "rho_from_cdf",
    "rho_from_h",
    "sigma_from_cdf",
    "tau_from_h",
    "xi_from_h",
    "zeta1_checkerboard",
    "zeta1_from_h",
]

DEFAULT_RTOL = 1e-8
DEFAULT_ATOL = 1e-10

Array = np.ndarray
Func = Callable[[Array, Array], Array]


def _ret(val, err, full_output):
    val = float(val)
    return (val, float(err)) if full_output else val


def _call(f, u, v):
    with np.errstate(all="ignore"):
        return np.asarray(f(u, v), dtype=float)


# known discontinuity lines (u_breaks, v_breaks) of the copula currently
# being evaluated (set by :func:`evaluate`, e.g. checkerboard grid lines)
_BREAKS: contextvars.ContextVar = contextvars.ContextVar("copul_measure_breaks", default=None)


def _i2(g, rtol, atol, scale):
    """Double integral with tolerances expressed on the measure scale."""
    br = _BREAKS.get()
    ub, vb = (None, None) if br is None else br
    return integrate_2d(g, rtol=rtol, atol=atol / abs(scale), u_breaks=ub, v_breaks=vb)


def _i1(g, rtol, atol, scale, diag=False, anti=False):
    br = _BREAKS.get()
    breaks = None
    if br is not None:
        ub, vb = br
        parts = [np.asarray(ub if ub is not None else [], float)]
        if vb is not None:
            vb = np.asarray(vb, float)
            parts.append(vb)
            if anti:
                parts.append(1.0 - vb)
        breaks = np.unique(np.concatenate(parts))
    return integrate_1d(g, rtol=rtol, atol=atol / abs(scale), breaks=breaks)


# ---------------------------------------------------------------------------
# Concordance-type measures
# ---------------------------------------------------------------------------


def _rho_cdf(cdf, rtol, atol):
    val, err = _i2(lambda u, v: _call(cdf, u, v), rtol, atol, 12)
    return 12 * val - 3, 12 * err


def rho_from_cdf(cdf: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Spearman's :math:`\rho = 12\int\!\!\int C - 3`."""
    return _ret(*_rho_cdf(cdf, rtol, atol), full_output)


def _rho_h(h1, rtol, atol):
    # int_0^1 C(u, v) du = int_0^1 (1 - s) h1(s, v) ds  (Fubini)
    val, err = _i2(lambda u, v: (1 - u) * _call(h1, u, v), rtol, atol, 12)
    return 12 * val - 3, 12 * err


def rho_from_h(h1: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Spearman's :math:`\rho = 12\int\!\!\int (1-u)\,\partial_1C(u,v)\,du\,dv - 3`."""
    return _ret(*_rho_h(h1, rtol, atol), full_output)


def _tau_h(h1, h2, rtol, atol):
    val, err = _i2(lambda u, v: _call(h1, u, v) * _call(h2, u, v), rtol, atol, 4)
    return 1 - 4 * val, 4 * err


def tau_from_h(h1: Func, h2: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Kendall's :math:`\tau = 1 - 4\int\!\!\int \partial_1C\,\partial_2C`.

    Valid for arbitrary copulas, including those with singular components.
    """
    return _ret(*_tau_h(h1, h2, rtol, atol), full_output)


def _xi_h(h, rtol, atol):
    val, err = _i2(lambda u, v: _call(h, u, v) ** 2, rtol, atol, 6)
    return 6 * val - 2, 6 * err


def xi_from_h(h: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Chatterjee's :math:`\xi = 6\int\!\!\int (\partial_1 C)^2 - 2`.

    Pass :math:`\partial_2 C` to obtain the variant conditioning on the
    second variable.
    """
    return _ret(*_xi_h(h, rtol, atol), full_output)


def _footrule_cdf(cdf, rtol, atol):
    val, err = _i1(lambda t: _call(cdf, t, t), rtol, atol, 6)
    return 6 * val - 2, 6 * err


def footrule_from_cdf(cdf: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Spearman's footrule :math:`\psi = 6\int_0^1 C(t,t)\,dt - 2`."""
    return _ret(*_footrule_cdf(cdf, rtol, atol), full_output)


def _gamma_cdf(cdf, rtol, atol):
    val, err = _i1(lambda t: _call(cdf, t, t) + _call(cdf, t, 1 - t), rtol, atol, 4, anti=True)
    return 4 * val - 2, 4 * err


def gamma_from_cdf(cdf: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Gini's :math:`\gamma = 4\int_0^1 [C(t,t)+C(t,1-t)]\,dt - 2`."""
    return _ret(*_gamma_cdf(cdf, rtol, atol), full_output)


def _beta_cdf(cdf, rtol=None, atol=None):
    c = float(_call(cdf, np.array([0.5]), np.array([0.5]))[0])
    return 4 * c - 1, 0.0


def beta_from_cdf(cdf: Func, *, full_output=False, **_):
    r"""Blomqvist's :math:`\beta = 4C(\tfrac12,\tfrac12) - 1`."""
    return _ret(*_beta_cdf(cdf), full_output)


def _nu_cdf(cdf, rtol, atol):
    val, err = _i2(lambda u, v: (1 - u) * _call(cdf, u, v), rtol, atol, 24)
    return 24 * val - 2, 24 * err


def nu_from_cdf(cdf: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Blest's :math:`\nu = 24\int\!\!\int (1-u)\,C(u,v) - 2`."""
    return _ret(*_nu_cdf(cdf, rtol, atol), full_output)


def _nu_h(h1, rtol, atol):
    # int_0^1 (1-u) C(u,v) du = int_0^1 h1(s,v) (1-s)^2 / 2 ds
    val, err = _i2(lambda u, v: (1 - u) ** 2 * _call(h1, u, v), rtol, atol, 12)
    return 12 * val - 2, 12 * err


def nu_from_h(h1: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Blest's :math:`\nu = 12\int\!\!\int (1-u)^2\,\partial_1C(u,v) - 2`."""
    return _ret(*_nu_h(h1, rtol, atol), full_output)


# ---------------------------------------------------------------------------
# Distances to independence
# ---------------------------------------------------------------------------


def lp_constant(p: float) -> float:
    r"""Normalizing constant :math:`k(p)` with :math:`k(p)\int\!\!\int|M-\Pi|^p = 1`.

    :math:`k(p) = (p+1) / (2\,B(p+1, p+2))`; e.g. :math:`k(1)=12`,
    :math:`k(2)=90`, :math:`k(3)=560`.
    """
    p = float(p)
    if p <= 0:
        raise ValueError("p must be positive")
    k = (p + 1) / (2 * _beta_fn(p + 1, p + 2))
    kr = round(k)
    return float(kr) if abs(k - kr) < 1e-9 * k else float(k)


def _lp_cdf(cdf, p, rtol, atol):
    k = lp_constant(p)
    val, err = _i2(lambda u, v: np.abs(_call(cdf, u, v) - u * v) ** p, rtol, atol, k)
    return k * val, k * err


def lp_from_cdf(
    cdf: Func, p: float = 2, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False
):
    r""":math:`L^p` distance :math:`k(p)\int\!\!\int |C-\Pi|^p`, normalized to 1 at :math:`M`."""
    return _ret(*_lp_cdf(cdf, p, rtol, atol), full_output)


def hoeffdings_d_from_cdf(cdf: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Hoeffding's :math:`\Phi^2 = 90\int\!\!\int (C-\Pi)^2`."""
    return _ret(*_lp_cdf(cdf, 2, rtol, atol), full_output)


def sigma_from_cdf(cdf: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Schweizer--Wolff :math:`\sigma = 12\int\!\!\int |C-\Pi|`."""
    return _ret(*_lp_cdf(cdf, 1, rtol, atol), full_output)


def _kappa_cdf(cdf, rtol=None, atol=None, n_grid=129, n_starts=6, n_zoom=14):
    def g(u, v):
        return np.abs(_call(cdf, u, v) - u * v)

    x = (np.arange(n_grid) + 0.5) / n_grid
    uu, vv = np.meshgrid(x, x, indexing="ij")
    vals = g(uu.ravel(), vv.ravel())
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(-vals)[:n_starts]
    best = float(vals[order[0]])
    r0 = 1.0 / n_grid
    for idx in order:
        cu, cv = float(uu.ravel()[idx]), float(vv.ravel()[idx])
        r = r0
        cur = float(vals[idx])
        for _ in range(n_zoom):
            s = np.linspace(-r, r, 11)
            pu = np.clip(cu + s, 1e-12, 1 - 1e-12)
            pv = np.clip(cv + s, 1e-12, 1 - 1e-12)
            PU, PV = np.meshgrid(pu, pv, indexing="ij")
            gv = g(PU.ravel(), PV.ravel())
            gv = np.where(np.isfinite(gv), gv, -np.inf)
            j = int(np.argmax(gv))
            if gv[j] >= cur:
                cur = float(gv[j])
                cu, cv = float(PU.ravel()[j]), float(PV.ravel()[j])
            r /= 4.0
        best = max(best, cur)
    # error: remaining search radius times a Lipschitz bound (|d(C-Pi)| <= 2)
    return 4 * best, 4 * 2 * r0 * 4.0**-n_zoom


def kappa_from_cdf(cdf: Func, *, full_output=False, **_):
    r"""Uniform distance :math:`\kappa = 4\sup|C-\Pi|` (grid search + zooming)."""
    return _ret(*_kappa_cdf(cdf), full_output)


def _bkr_h(cdf, h1, h2, rtol, atol):
    def g(u, v):
        return _call(h1, u, v) * (_call(cdf, u, v) - u * v) * (_call(h2, u, v) - u)

    val, err = _i2(g, rtol, atol, 60)
    return -60 * val, 60 * err


def bkr_from_h(
    cdf: Func, h1: Func, h2: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False
):
    r"""Blum--Kiefer--Rosenblatt :math:`B = 30\int (C-\Pi)^2\,dC`.

    Evaluated as :math:`-60\int\!\!\int \partial_1 C\,(C-\Pi)\,(\partial_2 C-u)`
    (integration by parts in :math:`v`), which is valid also for copulas with
    singular components.
    """
    return _ret(*_bkr_h(cdf, h1, h2, rtol, atol), full_output)


def _mi_pdf(pdf, rtol, atol):
    # a singular component (density mass < 1) makes the information infinite
    mass, mass_err = _i2(lambda u, v: np.maximum(_call(pdf, u, v), 0.0), 1e-7, 1e-9, 1)
    if mass < 1.0 - max(1e-5, 10 * mass_err):
        return math.inf, 0.0

    def g(u, v):
        c = np.maximum(_call(pdf, u, v), 0.0)
        with np.errstate(all="ignore"):
            return np.where(c > 0, c * np.log(np.where(c > 0, c, 1.0)), 0.0)

    val, err = _i2(g, rtol, atol, 1)
    return val, err


def mutual_information_from_pdf(
    pdf: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False
):
    r"""Mutual information :math:`I = \int\!\!\int c\log c` (natural logarithm).

    Returns ``inf`` if the density integrates to less than one, i.e. if the
    copula has a singular component (``pdf`` is then only its absolutely
    continuous part).
    """
    return _ret(*_mi_pdf(pdf, rtol, atol), full_output)


# ---------------------------------------------------------------------------
# Trutschnig's zeta_1 (D_1 distance to independence)
# ---------------------------------------------------------------------------


def _zeta1_h(h1, rtol, atol):
    val, err = _i2(lambda u, v: np.abs(_call(h1, u, v) - v), rtol, atol, 3)
    return 3 * val, 3 * err


def zeta1_from_h(h1: Func, *, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL, full_output=False):
    r"""Trutschnig's :math:`\zeta_1 = 3\int\!\!\int|\partial_1 C(u,v) - v|\,du\,dv`.

    :math:`\zeta_1(C) = 3 D_1(C, \Pi)` with the :math:`\partial`-metric
    :math:`D_1` of Trutschnig (2011); :math:`\zeta_1(C)=0` iff
    :math:`C=\Pi` and :math:`\zeta_1(C)=1` iff :math:`C` is completely
    dependent.

    References
    ----------
    Trutschnig, W. (2011). On a strong metric on the space of copulas and its
    induced dependence measure. *J. Math. Anal. Appl.* 384, 690--705.
    """
    return _ret(*_zeta1_h(h1, rtol, atol), full_output)


def _int_poly_abs(w0, w1, c0, c1):
    r""":math:`\int_0^1 (w_0 + w_1 b)\,|c_0 + c_1 b|\,db` exactly (vectorized)."""
    w0, w1, c0, c1 = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (w0, w1, c0, c1)))

    def prim(b):  # antiderivative of (w0 + w1 b)(c0 + c1 b)
        return w0 * c0 * b + (w0 * c1 + w1 * c0) * b**2 / 2 + w1 * c1 * b**3 / 3

    with np.errstate(all="ignore"):
        r = np.where(c1 != 0, -c0 / np.where(c1 != 0, c1, 1.0), -1.0)
    r = np.where((r > 0) & (r < 1), r, 1.0)
    s0 = np.sign(c0 + c1 * 0.5 * r)  # sign on (0, r)
    s1 = np.sign(c0 + c1 * 0.5 * (1 + r))  # sign on (r, 1)
    zero = np.zeros_like(r)
    return s0 * (prim(r) - prim(zero)) + s1 * (prim(np.ones_like(r)) - prim(r))


def zeta1_checkerboard(P, S=None) -> float:
    r"""Exact :math:`\zeta_1` of a bivariate checkerboard copula.

    ``P`` is the ``m x n`` mass matrix (normalized internally) and ``S`` the
    kernel sign matrix (``0``: :math:`\Pi`, ``1``: :math:`M`, ``-1``:
    :math:`W` cells, see :mod:`copul.checkerboard._biv_engine`).  For
    :math:`u` in row :math:`i` (local coordinate :math:`a`) and :math:`v` in
    column :math:`j` (local :math:`b`, :math:`v=(j+b)/n`)

    .. math::

       \partial_1 C(u,v) = m R_{ij} + m\Delta_{ij}\,k(a,b),\qquad
       R_{ij} = \textstyle\sum_{j'<j}\Delta_{ij'},

    with :math:`k=b` for :math:`\Pi` cells and :math:`k=\mathbf 1\{b>a\}`
    (:math:`M`) resp. :math:`\mathbf 1\{a+b>1\}` (:math:`W`).  For fixed
    :math:`b` the indicator equals one on an :math:`a`-set of measure
    :math:`b` in both singular cases, so with :math:`A = mR_{ij} - j/n` and
    :math:`p = m\Delta_{ij}` the cell contributes

    .. math::

       \frac{1}{mn}\int_0^1 |A + (p - \tfrac1n) b|\,db \;(\Pi),\qquad
       \frac{1}{mn}\int_0^1 \bigl[b\,|A + p - \tfrac bn|
       + (1-b)\,|A - \tfrac bn|\bigr]\,db \;(M, W),

    integrals of piecewise polynomials evaluated exactly.
    """
    P = np.asarray(P, dtype=float)
    P = P / P.sum()
    m, n = P.shape
    if S is None:
        S = np.zeros((m, n), dtype=int)
    S = np.broadcast_to(np.asarray(S, dtype=int), (m, n))
    R = np.zeros((m, n))
    R[:, 1:] = np.cumsum(P, axis=1)[:, :-1]
    j = np.arange(n)[None, :]
    A = m * R - j / n
    p = m * P
    c = 1.0 / n
    pi_part = _int_poly_abs(1.0, 0.0, A, p - c)
    sing_part = _int_poly_abs(0.0, 1.0, A + p, -c) + _int_poly_abs(1.0, -1.0, A, -c)
    cell = np.where(S == 0, pi_part, sing_part)
    return float(3.0 * cell.sum() / (m * n))


# ---------------------------------------------------------------------------
# Tail dependence (extrapolation)
# ---------------------------------------------------------------------------


def _extrapolate(ts, rs):
    """Limit of ``r(t)`` as ``t -> 0`` from values on a geometric grid ``t_k``.

    The sequence is truncated where it stops converging (an increment larger
    than twice the previous one signals floating-point breakdown of the
    copula expression at tiny ``t``); then an Aitken/Richardson step with the
    empirically estimated rate is applied to the last reliable triple.  The
    error estimate is the size of that correction plus the last increment.
    """
    rs = np.asarray(rs, float)
    ok = np.isfinite(rs)
    if not np.any(ok):
        return math.nan, math.inf
    # keep the leading finite part
    stop = np.flatnonzero(~ok)
    rs = rs[: stop[0]] if stop.size else rs
    if rs.size < 3:
        return float(np.clip(rs[-1], 0, 1)), math.inf
    d = np.diff(rs)
    m = rs.size
    for i in range(1, d.size):
        if abs(d[i]) > 2 * abs(d[i - 1]) + 1e-12:
            m = i + 1  # keep r_0..r_i
            break
    rs = rs[:m]
    if rs.size < 3:
        return float(np.clip(rs[-1], 0, 1)), float(abs(d[0]) if d.size else math.inf)
    r0, r1, r2 = rs[-3], rs[-2], rs[-1]
    d1, d2 = r1 - r0, r2 - r1
    lam = r2
    if d1 != 0 and abs(d2) < abs(d1):
        q = d2 / d1  # ~ 2^-s for r(t) = lambda + a t^s
        corr = d2 * q / (1 - q)
        lam = r2 + corr
        err = abs(corr) + abs(d2)
    else:
        err = abs(d2) + abs(d1)
    return float(np.clip(lam, 0.0, 1.0)), float(err)


def lambda_l_from_cdf(cdf: Func, *, full_output=False, **_):
    r"""Lower tail dependence :math:`\lim_{t\downarrow0}C(t,t)/t` (extrapolated).

    Evaluates :math:`C(t,t)/t` on :math:`t=2^{-k}` down to
    :math:`2^{-60}` (as long as the values are finite and positive) and
    extrapolates.  Reliable when :math:`C(t,t)/t` converges at an algebraic
    rate; for slowly (e.g. logarithmically) converging diagonals prefer a
    closed form -- the error estimate then is large.
    """
    t = 2.0 ** -np.arange(8, 61, dtype=float)
    c = _call(cdf, t, t)
    r = c / t
    good = np.isfinite(r) & (c > 0)
    if not np.any(good):
        return _ret(0.0, math.inf, full_output)
    last = np.flatnonzero(good)[-1]
    return _ret(*_extrapolate(t[: last + 1], r[: last + 1]), full_output)


def lambda_u_from_cdf(cdf: Func, *, full_output=False, **_):
    r"""Upper tail dependence :math:`\lim_{t\downarrow0}(2t-1+C(1-t,1-t))/t` (extrapolated).

    Limited by the resolution of :math:`1-t` in double precision
    (:math:`t\ge2^{-30}`), see :func:`lambda_l_from_cdf`.
    """
    t = 2.0 ** -np.arange(8, 31, dtype=float)
    r = (2 * t - 1 + _call(cdf, 1 - t, 1 - t)) / t
    return _ret(*_extrapolate(t, r), full_output)


def _tail_from_h(h1: Func, upper: bool):
    from copul.measures.quadrature import integrate_1d_batch

    k = np.arange(4, 37 if upper else 61, 2, dtype=float)
    t = 2.0**-k

    if upper:

        def g(x, r):
            return 1.0 - _call(h1, 1.0 - t[r] * x, 1.0 - t[r])
    else:

        def g(x, r):
            return _call(h1, t[r] * x, t[r])

    r, _ = integrate_1d_batch(g, np.zeros(t.size), np.ones(t.size), atol=1e-12, rtol=1e-10)
    return _extrapolate(t, r)


def lambda_l_from_h(h1: Func, *, full_output=False, **_):
    r"""Lower tail dependence via :math:`C(t,t)/t = \int_0^1 \partial_1C(tx, t)\,dx`.

    Avoids forming :math:`C(t,t)` for tiny :math:`t`; extrapolated like
    :func:`lambda_l_from_cdf`.
    """
    return _ret(*_tail_from_h(h1, upper=False), full_output)


def lambda_u_from_h(h1: Func, *, full_output=False, **_):
    r"""Upper tail dependence via
    :math:`\hat C(t,t)/t = \int_0^1 \bigl(1-\partial_1C(1-tx, 1-t)\bigr)\,dx`,
    which involves no cancellation (unlike :math:`2t-1+C(1-t,1-t)`).
    """
    return _ret(*_tail_from_h(h1, upper=True), full_output)


# ---------------------------------------------------------------------------
# Generic front-ends
# ---------------------------------------------------------------------------


@dataclass
class NumericCopula:
    """Bundle of vectorized callables describing a bivariate copula.

    Missing ingredients are derived on demand: ``h1``/``h2`` by central
    finite differences of ``cdf``, ``pdf`` by finite differences of ``h1``,
    and ``cdf`` by integrating ``h1`` in :math:`u`.
    """

    cdf: Func | None = None
    h1: Func | None = None
    h2: Func | None = None
    pdf: Func | None = None
    #: optional known discontinuity lines ``(u_breaks, v_breaks)``
    breaks: tuple | None = None

    def get(self, what: str) -> Func:
        f = getattr(self, what)
        if f is not None:
            return f
        if what == "cdf":
            if self.h1 is None:
                raise ValueError("Need at least cdf or h1.")
            f = _cdf_from_h1(self.h1)
        elif what == "h1":
            f = _fd_partial(self.get("cdf"), 0)
        elif what == "h2":
            f = _fd_partial(self.get("cdf"), 1)
        elif what == "pdf":
            f = _fd_partial(self.get("h1"), 1, clip=None)
        else:  # pragma: no cover
            raise KeyError(what)
        setattr(self, what, f)
        return f


def _fd_step(x):
    return np.minimum(np.minimum(6e-6, 0.5 * x), 0.5 * (1.0 - x))


def _fd_partial(f: Func, axis: int, clip=(0.0, 1.0)) -> Func:
    """Central finite difference of ``f`` w.r.t. its ``axis``-th argument."""

    def d(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        x = u if axis == 0 else v
        h = _fd_step(x)
        if axis == 0:
            val = (_call(f, u + h, v) - _call(f, u - h, v)) / (2 * h)
        else:
            val = (_call(f, u, v + h) - _call(f, u, v - h)) / (2 * h)
        if clip is not None:
            val = np.clip(val, *clip)
        return val

    return d


def _cdf_from_h1(h1: Func, rtol=1e-12, atol=1e-14) -> Func:
    r""":math:`C(u,v) = \int_0^u \partial_1 C(s,v)\,ds` by adaptive quadrature."""
    from copul.measures.quadrature import integrate_1d_batch

    def cdf(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        shape = u.shape
        uf, vf = u.ravel(), v.ravel()
        vals, _ = integrate_1d_batch(
            lambda s, r: _call(h1, s, vf[r]), np.zeros_like(uf), uf, atol=atol, rtol=rtol
        )
        return vals.reshape(shape)

    return cdf


def _key_options(key: str, options: dict) -> dict:
    m = get_measure(key)
    return {o: options[o] for o in m.options if o in options}


def evaluate(
    key: str,
    funcs: NumericCopula,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    p: float = 2,
    prefer_h: bool = False,
) -> tuple[float, float]:
    """Evaluate the measure ``key`` from a :class:`NumericCopula`.

    Returns ``(value, error_estimate)``.  With ``prefer_h=True`` the
    h-function formulas are used for :math:`\\rho` and :math:`\\nu` (useful
    when only ``h1`` is known).
    """
    token = _BREAKS.set(getattr(funcs, "breaks", None))
    try:
        return _evaluate(key, funcs, rtol, atol, p, prefer_h)
    finally:
        _BREAKS.reset(token)


def _evaluate(key, funcs, rtol, atol, p, prefer_h):
    key = resolve_key(key)
    have_cdf = funcs.cdf is not None
    if key == "rho":
        if prefer_h or not have_cdf:
            return _rho_h(funcs.get("h1"), rtol, atol)
        return _rho_cdf(funcs.cdf, rtol, atol)
    if key == "tau":
        return _tau_h(funcs.get("h1"), funcs.get("h2"), rtol, atol)
    if key == "xi":
        return _xi_h(funcs.get("h1"), rtol, atol)
    if key == "xi_2":
        return _xi_h(funcs.get("h2"), rtol, atol)
    if key == "footrule":
        return _footrule_cdf(funcs.get("cdf"), rtol, atol)
    if key == "gamma":
        return _gamma_cdf(funcs.get("cdf"), rtol, atol)
    if key == "beta":
        return _beta_cdf(funcs.get("cdf"))
    if key == "nu":
        if prefer_h or not have_cdf:
            return _nu_h(funcs.get("h1"), rtol, atol)
        return _nu_cdf(funcs.cdf, rtol, atol)
    if key == "hoeffdings_d":
        return _lp_cdf(funcs.get("cdf"), 2, rtol, atol)
    if key == "sigma":
        return _lp_cdf(funcs.get("cdf"), 1, rtol, atol)
    if key == "lp":
        return _lp_cdf(funcs.get("cdf"), p, rtol, atol)
    if key == "kappa":
        return _kappa_cdf(funcs.get("cdf"))
    if key == "bkr":
        return _bkr_h(funcs.get("cdf"), funcs.get("h1"), funcs.get("h2"), rtol, atol)
    if key == "mutual_information":
        return _mi_pdf(funcs.get("pdf"), rtol, atol)
    if key == "zeta1":
        return _zeta1_h(funcs.get("h1"), rtol, atol)
    if key in ("lambda_l", "lambda_u"):
        # h-function formulas avoid cancellation; use them unless h1 would
        # only be a finite-difference approximation of the cdf
        native = getattr(funcs, "has_native_h", None)
        use_h = native() if native is not None else funcs.h1 is not None
        if use_h:
            f = lambda_l_from_h if key == "lambda_l" else lambda_u_from_h
            return f(funcs.get("h1"), full_output=True)
        f = lambda_l_from_cdf if key == "lambda_l" else lambda_u_from_cdf
        return f(funcs.get("cdf"), full_output=True)
    raise KeyError(key)  # pragma: no cover


def _collect(keys, fn, full_output):
    out = {}
    for k in _iter_keys(keys):
        val, err = fn(k)
        out[k] = (float(val), float(err)) if full_output else float(val)
    return out


def measures_from_cdf(
    cdf: Func,
    measures: str | Iterable[str] | None = None,
    *,
    h1: Func | None = None,
    h2: Func | None = None,
    pdf: Func | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    p: float = 2,
    full_output: bool = False,
) -> dict[str, float]:
    """Compute several measures from a vectorized copula ``cdf``.

    Missing partial derivatives are obtained by central finite differences of
    ``cdf`` (accuracy about ``1e-9`` for smooth copulas); pass ``h1``/``h2``
    for full accuracy.

    Returns
    -------
    dict
        ``{key: value}`` (or ``{key: (value, error)}`` with
        ``full_output=True``) keyed by canonical measure keys.
    """
    funcs = NumericCopula(cdf=cdf, h1=h1, h2=h2, pdf=pdf)
    if isinstance(measures, str):
        val, err = evaluate(measures, funcs, rtol=rtol, atol=atol, p=p)
        return _ret(val, err, full_output)
    return _collect(
        measures,
        lambda k: evaluate(k, funcs, rtol=rtol, atol=atol, p=p),
        full_output,
    )


def measures_from_h(
    h,
    measures: str | Iterable[str] | None = None,
    *,
    h2: Func | None = None,
    cdf: Func | None = None,
    pdf: Func | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    p: float = 2,
    full_output: bool = False,
):
    r"""Compute measures of a copula given by its h-function :math:`\partial_1 C`.

    Parameters
    ----------
    h : callable or ndarray
        Either a vectorized callable ``h(u, v) = \partial_1 C(u, v)`` or a
        2-D array of its values on the midpoint grid (see module docstring).
    measures : str or iterable of str, optional
        Measure keys (default :data:`~copul.measures.registry.DEFAULT_MEASURES`).
    h2, cdf, pdf : callable, optional
        Further ingredients if known; otherwise :math:`C` is obtained by
        integrating ``h`` in :math:`u` and :math:`\partial_2 C` by finite
        differences of that integral (slower and less accurate -- supply
        ``h2`` or ``cdf`` when available).

    Notes
    -----
    :math:`\xi`, :math:`\rho` and :math:`\nu` are computed directly from
    ``h`` (:math:`\rho = 12\int\!\!\int(1-u)h - 3`,
    :math:`\nu = 12\int\!\!\int(1-u)^2h - 2`).
    """
    if isinstance(h, np.ndarray):
        return _measures_from_h_grid(h, measures, p=p, full_output=full_output)
    funcs = NumericCopula(cdf=cdf, h1=h, h2=h2, pdf=pdf)
    if isinstance(measures, str):
        val, err = evaluate(measures, funcs, rtol=rtol, atol=atol, p=p, prefer_h=True)
        return _ret(val, err, full_output)
    return _collect(
        measures,
        lambda k: evaluate(k, funcs, rtol=rtol, atol=atol, p=p, prefer_h=True),
        full_output,
    )


def _measures_from_h_grid(H: np.ndarray, measures, p=2, full_output=False):
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("h grid must be two-dimensional (rows: u, columns: v)")
    m, n = H.shape
    u = (np.arange(m) + 0.5) / m
    v = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, v, indexing="ij")
    # C at u-edges k/m (k = 0..m), then midpoint values
    Ce = np.vstack([np.zeros((1, n)), np.cumsum(H, axis=0) / m])
    C = 0.5 * (Ce[:-1] + Ce[1:])
    H2 = np.gradient(C, v, axis=1, edge_order=2) if n > 2 else np.gradient(C, v, axis=1)

    # bilinear interpolation of C on the grid augmented by the boundary
    ue = np.concatenate([[0.0], u, [1.0]])
    ve = np.concatenate([[0.0], v, [1.0]])
    Cfull = np.zeros((m + 2, n + 2))
    Cfull[1:-1, 1:-1] = C
    Cfull[-1, :] = ve
    Cfull[:, -1] = ue
    Cfull[1:-1, -1] = u  # C(u, 1) = u
    Cfull[-1, 1:-1] = v  # C(1, v) = v
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((ue, ve), Cfull)

    def cdf(a, b):
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        pts = np.column_stack([a.ravel(), b.ravel()])
        return interp(pts).reshape(a.shape)

    def one(key):
        key = resolve_key(key)
        if key == "xi":
            return 6 * np.mean(H**2) - 2
        if key == "xi_2":
            return 6 * np.mean(H2**2) - 2
        if key == "rho":
            return 12 * np.mean(C) - 3
        if key == "tau":
            return 1 - 4 * np.mean(H * H2)
        if key == "nu":
            return 24 * np.mean((1 - U) * C) - 2
        if key in ("footrule", "gamma", "beta"):
            N = max(m, n) * 4
            t = (np.arange(N) + 0.5) / N
            d = np.mean(cdf(t, t))
            if key == "footrule":
                return 6 * d - 2
            if key == "gamma":
                return 4 * (d + np.mean(cdf(t, 1 - t))) - 2
            return 4 * float(cdf(np.array([0.5]), np.array([0.5]))[0]) - 1
        if key == "hoeffdings_d":
            return 90 * np.mean((C - U * V) ** 2)
        if key == "sigma":
            return 12 * np.mean(np.abs(C - U * V))
        if key == "lp":
            return lp_constant(p) * np.mean(np.abs(C - U * V) ** p)
        if key == "kappa":
            return 4 * np.max(np.abs(C - U * V))
        if key == "bkr":
            return -60 * np.mean(H * (C - U * V) * (H2 - U))
        if key == "zeta1":
            return 3 * np.mean(np.abs(H - V))
        if key == "mutual_information":
            c = np.maximum(np.gradient(H, v, axis=1), 0.0)
            with np.errstate(all="ignore"):
                return np.mean(np.where(c > 0, c * np.log(np.where(c > 0, c, 1)), 0))
        raise ValueError(f"Measure {key!r} cannot be computed from an h-grid.")

    step = 1.0 / min(m, n)
    if isinstance(measures, str):
        return _ret(one(measures), step**2, full_output)
    out = {}
    for k in _iter_keys(measures):
        val = float(one(k))
        out[k] = (val, step**2) if full_output else val
    return out
