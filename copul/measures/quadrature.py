r"""
Vectorized quadrature on :math:`[0,1]` and :math:`[0,1]^2`.

All rules use *interior* nodes only, so integrands are never evaluated at
:math:`0` or :math:`1` (where copula expressions typically produce ``0/0``,
``log(0)`` or overflow).

Two families of rules are provided:

* :func:`integrate_1d` / :func:`integrate_2d` -- *locally adaptive*
  Gauss--Kronrod (7/15) bisection, vectorized over many intervals and many
  independent integrals ("rows") at once.  The 2D rule is iterated: an
  adaptive outer integral over :math:`v` of adaptive inner integrals over
  :math:`u`.  Because refinement is local, kinks and jumps of the integrand
  (singular components, Fréchet bounds, checkerboards) and boundary
  singularities are resolved efficiently.  This is the default of the
  measures engine.

* :func:`gauss_legendre_1d` / :func:`gauss_legendre_2d` -- composite
  Gauss--Legendre rules on uniform panels with *global* refinement: the
  number of panels is doubled until two successive estimates agree to
  ``rtol``/``atol`` (or ``max_level`` is reached).  Very fast for integrands
  that are smooth up to the boundary.

Every routine returns ``(value, error_estimate)``.

Conventions for integrands
--------------------------
* 1D: ``f(x)`` receives a 1-D ``ndarray`` and returns an array of the same
  shape.
* 2D: ``f(u, v)`` receives two 1-D arrays of equal length and returns an
  array of that length (elementwise evaluation).
* batched 1D (:func:`integrate_1d_batch`): ``f(x, rows)`` receives the nodes
  and the index of the integral each node belongs to.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

__all__ = [
    "gauss_legendre_1d",
    "gauss_legendre_2d",
    "gauss_legendre_nodes",
    "integrate_1d",
    "integrate_1d_batch",
    "integrate_2d",
]

# ---------------------------------------------------------------------------
# Gauss–Kronrod 7/15 nodes and weights on [-1, 1] (QUADPACK qk15)
# ---------------------------------------------------------------------------
_XGK = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ]
)
_WGK = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_WG = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)

# full 15-point node / weight vectors ordered from -1 to 1
_X15 = np.concatenate([-_XGK[:-1], _XGK[::-1]])
_W15 = np.concatenate([_WGK[:-1], _WGK[::-1]])
_WG15 = np.zeros(15)
# Gauss nodes are the odd-indexed Kronrod nodes (±x1, ±x3, ±x5, 0)
_g_pos = {1: 0, 3: 1, 5: 2, 7: 3}
for _i, _x in enumerate(_X15):
    _k = 7 - abs(_i - 7)  # index into _XGK (0 = outermost)
    if _k in _g_pos:
        _WG15[_i] = _WG[_g_pos[_k]]

_EPS = np.finfo(float).eps
_CHUNK = 100_000  # max. intervals evaluated per call of the integrand


def _lagrange_at(t, nodes):
    w = np.ones(len(nodes))
    for i, xi in enumerate(nodes):
        for j, xj in enumerate(nodes):
            if i != j:
                w[i] *= (t - xj) / (xi - xj)
    return w


# quadratic extrapolation weights from the three outermost Kronrod nodes to
# the interval endpoints (used to detect jumps hidden between an endpoint
# and the first node, where no Gauss-type rule can see them)
_EXT_LO = _lagrange_at(-1.0, _X15[:3])
_EXT_HI = _lagrange_at(1.0, _X15[-3:])
_GAP = 1.0 - _XGK[0]  # distance endpoint -> outermost node (half-width units)


def _qk15(f, lo, hi, rows, lo_check=None, hi_check=None):
    """Evaluate the 15-point Kronrod rule on many intervals at once.

    ``lo_check`` / ``hi_check`` flag interval endpoints that lie strictly
    inside the integration domain; ``f`` is additionally evaluated there and
    the deviation from a smooth extrapolation of the rule's outermost nodes
    enters the error estimate (a jump in the blind zone between an endpoint
    and the first node is invisible to the rule otherwise).
    """
    c = 0.5 * (lo + hi)
    h = 0.5 * (hi - lo)
    m = lo.size
    x = c[:, None] + h[:, None] * _X15[None, :]
    xs = [x.ravel()]
    rs = [np.repeat(rows, 15)]
    n_lo = n_hi = 0
    if lo_check is not None:
        xs.append(lo[lo_check])
        rs.append(rows[lo_check])
        n_lo = int(np.count_nonzero(lo_check))
    if hi_check is not None:
        xs.append(hi[hi_check])
        rs.append(rows[hi_check])
        n_hi = int(np.count_nonzero(hi_check))
    with np.errstate(all="ignore"):
        out = f(np.concatenate(xs), np.concatenate(rs))
    ye = None
    if isinstance(out, tuple):  # (values, pointwise error estimates)
        out, ye = out
    yy = np.broadcast_to(np.asarray(out, dtype=float), (15 * m + n_lo + n_hi,))
    y = yy[: 15 * m].reshape(x.shape)
    if ye is not None:
        ye = np.broadcast_to(np.asarray(ye, dtype=float), yy.shape)[: 15 * m].reshape(x.shape)
        ierr = np.abs(h) * (np.abs(ye) @ _W15)
    else:
        ierr = np.zeros(m)
    blind = np.zeros(m)
    if n_lo:
        f_lo = yy[15 * m : 15 * m + n_lo]
        pred = y[lo_check, :3] @ _EXT_LO
        blind[lo_check] += np.abs(f_lo - pred) * _GAP * np.abs(h[lo_check])
    if n_hi:
        f_hi = yy[15 * m + n_lo :]
        pred = y[hi_check, -3:] @ _EXT_HI
        blind[hi_check] += np.abs(f_hi - pred) * _GAP * np.abs(h[hi_check])
    blind = np.nan_to_num(blind, nan=0.0, posinf=0.0)
    k = h * (y @ _W15)
    g = h * (y @ _WG15)
    # QUADPACK-style error estimate
    mean = k / np.where(h != 0, 2 * h, 1.0)
    resasc = np.abs(h) * (np.abs(y - mean[:, None]) @ _W15)
    resabs = np.abs(h) * (np.abs(y) @ _W15)
    err = np.abs(k - g)
    with np.errstate(all="ignore"):
        scale = np.where(
            (resasc != 0) & (err != 0),
            np.minimum(1.0, (200.0 * err / np.where(resasc != 0, resasc, 1.0)) ** 1.5),
            1.0,
        )
    err = np.where((resasc != 0) & (err != 0), resasc * scale, err)
    err = np.maximum(err, 50 * _EPS * resabs) + blind
    bad = ~np.isfinite(k) | ~np.isfinite(err)
    if np.any(bad):
        k = np.where(bad, np.nan, k)
        err = np.where(bad, np.inf, err)
    return k, err, resabs, ierr


_GRADED = np.concatenate(
    [[0.0], 4.0 ** -np.arange(7, 0, -1), [0.5], 1.0 - 4.0 ** -np.arange(1, 8), [1.0]]
)


def _initial_breakpoints(init_panels, extra=None) -> np.ndarray:
    if isinstance(init_panels, str):
        if init_panels != "graded":
            raise ValueError(f"Unknown init_panels {init_panels!r}")
        t = _GRADED
    elif np.ndim(init_panels) == 0:
        t = np.linspace(0.0, 1.0, max(1, int(init_panels)) + 1)
    else:
        t = np.unique(np.clip(np.asarray(init_panels, float), 0.0, 1.0))
        t = np.unique(np.concatenate([[0.0], t, [1.0]]))
    if extra is not None and np.size(extra):
        e = np.asarray(extra, float).ravel()
        e = e[(e > 0) & (e < 1)]
        t = np.unique(np.concatenate([t, e]))
        # drop graded points that nearly coincide with an extra breakpoint
        keep = np.concatenate([[True], np.diff(t) > 1e-12])
        t = t[keep]
    return t


def integrate_1d_batch(
    f: Callable,
    a,
    b,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    init_panels="graded",
    max_level: int = 50,
    max_intervals: int = 400_000,
    max_per_row: int = 2_000,
    jump_search: bool = False,
    budget: list | None = None,
    breaks=None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Adaptively integrate many 1D integrals :math:`\int_{a_r}^{b_r} f(x, r)\,dx`.

    Parameters
    ----------
    f : callable
        ``f(x, rows) -> ndarray``; ``rows`` holds the integral index of each
        node.
    a, b : float or array_like
        Integration limits, broadcast to a common shape ``(R,)``.
    atol, rtol : float
        Per-integral absolute / relative tolerance.
    init_panels : int, "graded" or array_like
        Initial partition of every integration interval: ``n`` equal panels,
        ``"graded"`` (default; panels refined geometrically towards both
        endpoints, which makes jumps and singularities very close to the
        boundary visible to the rule) or explicit relative breakpoints in
        :math:`[0,1]`.
    max_level : int
        Maximal number of bisection rounds.
    max_intervals : int
        Safety cap on the number of simultaneously active intervals.
    max_per_row : int
        Safety cap on the number of active intervals of a single integral.
    jump_search : bool
        Locate jump discontinuities by bisection on function values (one
        evaluation per step) instead of refining them with the full rule.
    breaks : array_like, optional
        Additional initial breakpoints, relative to ``[a, b]`` (e.g. known
        discontinuities of the integrand).
    budget : list of one int, optional
        Shared, mutable evaluation budget ``[remaining]``; when exhausted
        all active intervals are accepted (the error estimate reflects it).

    Returns
    -------
    values, errors : ndarray
        Arrays of shape ``(R,)``.
    """
    a, b = np.broadcast_arrays(
        np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
    )
    R = a.size
    a = a.ravel()
    b = b.ravel()
    width = np.abs(b - a)
    width_safe = np.where(width > 0, width, 1.0)

    t = _initial_breakpoints(init_panels, breaks)
    n0 = t.size - 1
    rows = np.repeat(np.arange(R), n0)
    lo = (a[:, None] + (b - a)[:, None] * t[None, :-1]).ravel()
    hi = (a[:, None] + (b - a)[:, None] * t[None, 1:]).ravel()
    nonempty = width[rows] > 0
    rows, lo, hi = rows[nonempty], lo[nonempty], hi[nonempty]

    done_val = np.zeros(R)
    done_err = np.zeros(R)
    done_ierr = np.zeros(R)  # integrated pointwise errors of f (if provided)
    if rows.size == 0:
        return done_val, done_err

    lower = np.minimum(a, b)
    upper = np.maximum(a, b)
    perr = np.full(rows.size, np.inf)  # error estimate of the parent interval
    # endpoints placed at a located jump are excluded from the blind-zone check
    nolo = np.zeros(rows.size, bool)
    nohi = np.zeros(rows.size, bool)
    tried = np.zeros(rows.size, bool)  # jump search already attempted in lineage
    for level in range(max_level + 1):
        lo_in = np.minimum(lo, hi) > lower[rows]
        hi_in = np.maximum(lo, hi) < upper[rows]
        lo_check = lo_in & ~nolo
        hi_check = hi_in & ~nohi
        if lo.size <= _CHUNK:
            k, err, resabs, ierr = _qk15(f, lo, hi, rows, lo_check, hi_check)
        else:
            parts = [
                _qk15(f, lo[s], hi[s], rows[s], lo_check[s], hi_check[s])
                for s in (slice(i, i + _CHUNK) for i in range(0, lo.size, _CHUNK))
            ]
            k = np.concatenate([p[0] for p in parts])
            err = np.concatenate([p[1] for p in parts])
            resabs = np.concatenate([p[2] for p in parts])
            ierr = np.concatenate([p[3] for p in parts])
        if budget is not None:
            budget[0] -= 17 * lo.size
        row_val = done_val + np.bincount(rows, np.nan_to_num(k), minlength=R)
        row_err = done_err + np.bincount(rows, err, minlength=R)
        tol = np.maximum(atol, rtol * np.abs(row_val))
        converged = row_err <= tol
        frac = np.abs(hi - lo) / width_safe[rows]
        tiny = np.abs(hi - lo) <= 64 * _EPS * np.maximum(1.0, np.abs(lo))
        # local acceptance: share of the row tolerance, or a *relative*
        # floor 0.1*rtol*int|f| (guards against refining evaluation noise,
        # e.g. cancellation in 1-u for u close to 1)
        accept = (
            converged[rows]
            | (err <= 0.5 * tol[rows] * frac)
            | (err <= 0.1 * rtol * resabs)
            | tiny
            | (level == max_level)
            | (budget is not None and budget[0] <= 0)
        )
        if rows.size > max_intervals:
            accept[:] = True
        # per-integral budget: rows with too many active intervals are
        # accepted as they are (their error estimate is kept)
        n_active = np.bincount(rows, minlength=R)
        over = n_active > max_per_row
        if np.any(over):
            accept |= over[rows]
        if np.any(accept):
            done_val += np.bincount(rows[accept], k[accept], minlength=R)
            done_err += np.bincount(rows[accept], err[accept], minlength=R)
            done_ierr += np.bincount(rows[accept], ierr[accept], minlength=R)
        split = ~accept
        if not np.any(split):
            break
        r_s, lo_s, hi_s, e_s = rows[split], lo[split], hi[split], err[split]
        mid = 0.5 * (lo_s + hi_s)
        right = mid.copy()
        located = np.zeros(r_s.size, bool)
        tried_s = tried[split].copy()
        if jump_search:
            # Jump localization: an interior interval whose error only halved
            # w.r.t. its parent behaves like a jump discontinuity (smooth
            # integrands gain a factor ~2^-15 per bisection).  Locate the jump
            # by bisection on function values (one evaluation per step) and
            # split there instead of at the midpoint.
            ratio = e_s / perr[split]
            sus = (
                (ratio > 0.35)
                & (ratio < 0.7)
                & ~tried[split]
                & lo_in[split]
                & hi_in[split]
                & (np.abs(hi_s - lo_s) < 0.2 * width_safe[r_s])
            )
            if np.any(sus):
                tried_s |= sus
                jl, jr = _locate_jumps(f, lo_s[sus], hi_s[sus], r_s[sus])
                if budget is not None:
                    budget[0] -= 10 * 31 * int(np.count_nonzero(sus))
                w_s = hi_s[sus] - lo_s[sus]
                # only use locations clearly inside the interval (otherwise
                # the bisection followed a smooth steep part; split normally)
                ok = ((jl - lo_s[sus]) > 1e-3 * w_s) & ((hi_s[sus] - jr) > 1e-3 * w_s)
                idx = np.flatnonzero(sus)[ok]
                mid[idx] = jl[ok]
                right[idx] = jr[ok]
                located[idx] = True
                # the dropped sliver [jl, jr] is ~1e-13 of the interval
        rows = np.concatenate([r_s, r_s])
        lo = np.concatenate([lo_s, right])
        hi = np.concatenate([mid, hi_s])
        perr = np.concatenate([e_s, e_s])
        tried = np.concatenate([tried_s, tried_s])
        nolo = np.concatenate([nolo[split], located])
        nohi = np.concatenate([located, nohi[split]])
    return done_val, done_err + done_ierr


def _locate_jumps(f, lo, hi, rows, n_iter: int = 9, m: int = 31):
    """Vectorized ``m``-section search for a jump of ``f`` in ``[lo, hi]``.

    Each iteration evaluates ``f`` at ``m`` equispaced interior points of the
    current bracket and keeps the sub-bracket with the largest increment;
    ``n_iter=9`` and ``m=31`` shrink the bracket by ``32**9 ~ 3.5e13``.
    """
    n = lo.size
    a = lo.copy()
    b = hi.copy()
    with np.errstate(all="ignore"):
        y = np.asarray(f(np.concatenate([a, b]), np.concatenate([rows, rows])), dtype=float)
    y = np.broadcast_to(y, (2 * n,))
    fa, fb = y[:n].copy(), y[n:].copy()
    t = np.arange(1, m + 1) / (m + 1)
    rr = np.repeat(rows, m)
    idx = np.arange(n)
    for _ in range(n_iter):
        x = a[:, None] + (b - a)[:, None] * t[None, :]
        with np.errstate(all="ignore"):
            g = np.broadcast_to(np.asarray(f(x.ravel(), rr), dtype=float), (n * m,)).reshape(n, m)
        gg = np.concatenate([fa[:, None], g, fb[:, None]], axis=1)
        d = np.abs(np.diff(gg, axis=1))
        d = np.where(np.isfinite(d), d, -1.0)
        j = np.argmax(d, axis=1)
        xx = np.concatenate([a[:, None], x, b[:, None]], axis=1)
        a, b = xx[idx, j], xx[idx, j + 1]
        fa, fb = gg[idx, j], gg[idx, j + 1]
    return a, b


def integrate_1d(
    f: Callable,
    a: float = 0.0,
    b: float = 1.0,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    init_panels="graded",
    max_level: int = 50,
    breaks=None,
) -> tuple[float, float]:
    r"""Adaptive Gauss--Kronrod integration of ``f`` over :math:`[a,b]`.

    ``f`` must accept and return 1-D arrays.  Only interior points of
    :math:`[a,b]` are evaluated.

    Returns
    -------
    (value, error_estimate)
    """
    val, err = integrate_1d_batch(
        lambda x, r: f(x),
        a,
        b,
        atol=atol,
        rtol=rtol,
        init_panels=init_panels,
        max_level=max_level,
        breaks=None if breaks is None else (np.asarray(breaks, float) - a) / (b - a),
    )
    return float(val[0]), float(err[0])


def integrate_2d(
    f: Callable,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    inner_limits: Callable | None = None,
    init_panels="graded",
    max_level: int = 50,
    max_evals: int = 3_000_000,
    u_breaks=None,
    v_breaks=None,
) -> tuple[float, float]:
    r"""Iterated adaptive integration :math:`\int_0^1\int_{a(v)}^{b(v)} f(u,v)\,du\,dv`.

    Parameters
    ----------
    f : callable
        ``f(u, v)`` evaluated elementwise on 1-D arrays.
    atol, rtol : float
        Target tolerances for the double integral.
    inner_limits : callable, optional
        ``inner_limits(v) -> (a, b)`` arrays of inner limits; defaults to
        :math:`[0, 1]`.
    u_breaks, v_breaks : array_like, optional
        Known discontinuity locations in :math:`u` (inner, only with the
        default inner limits) and :math:`v` (outer), used as initial
        breakpoints.
    max_evals : int
        Budget of integrand evaluations; when it is exhausted all pending
        intervals are accepted and the (then larger) error estimate is
        returned.  Protects against pathological integrands (essential
        singularities, noise) at the price of accuracy.

    Returns
    -------
    (value, error_estimate)
    """
    # inner integrals only need an absolute accuracy relative to the size of
    # the double integral -> cheap pre-estimate of its magnitude
    x, w = _composite_nodes(6, 4)
    uu = np.repeat(x, x.size)
    vv = np.tile(x, x.size)
    with np.errstate(all="ignore"):
        pre = np.asarray(f(uu, vv), dtype=float)
    scale = float(np.nansum(np.abs(pre) * np.repeat(w, w.size) * np.tile(w, w.size)))
    if not np.isfinite(scale):
        scale = 0.0
    inner_atol = 0.1 * max(atol, 0.5 * rtol * scale)
    inner_rtol = 0.1 * rtol
    budget = [int(max_evals)]

    def outer(v, _rows):
        v = np.asarray(v, float)
        if inner_limits is None:
            lo, hi = np.zeros_like(v), np.ones_like(v)
        else:
            lo, hi = inner_limits(v)
            lo = np.broadcast_to(np.asarray(lo, float), v.shape)
            hi = np.broadcast_to(np.asarray(hi, float), v.shape)
        vals, errs = integrate_1d_batch(
            lambda u, r: f(u, v[r]),
            lo,
            hi,
            atol=inner_atol,
            rtol=inner_rtol,
            init_panels=init_panels,
            max_level=max_level,
            budget=budget,
            breaks=u_breaks if inner_limits is None else None,
        )
        return vals, errs

    # the outer integrand is continuous (and every evaluation is expensive):
    # no jump localization there
    val, err = integrate_1d_batch(
        outer,
        0.0,
        1.0,
        atol=atol,
        rtol=rtol,
        init_panels=init_panels,
        max_level=max_level,
        jump_search=False,
        budget=budget,
        breaks=v_breaks,
    )
    return float(val[0]), float(err[0])


# ---------------------------------------------------------------------------
# Composite Gauss–Legendre with global panel doubling
# ---------------------------------------------------------------------------


@lru_cache(maxsize=64)
def gauss_legendre_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights on :math:`[0,1]` (interior nodes)."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


def _composite_nodes(n_nodes: int, panels: int, a: float = 0.0, b: float = 1.0):
    x, w = gauss_legendre_nodes(n_nodes)
    edges = np.linspace(a, b, panels + 1)
    h = np.diff(edges)
    nodes = (edges[:-1, None] + h[:, None] * x[None, :]).ravel()
    weights = (h[:, None] * w[None, :]).ravel()
    return nodes, weights


def gauss_legendre_1d(
    f: Callable,
    a: float = 0.0,
    b: float = 1.0,
    *,
    n_nodes: int = 10,
    panels: int = 4,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    max_level: int = 12,
) -> tuple[float, float]:
    r"""Composite Gauss--Legendre rule with panel doubling.

    The number of panels is doubled until two successive estimates agree to
    ``max(atol, rtol*|I|)`` or ``max_level`` doublings were performed.

    Returns
    -------
    (value, error_estimate) where the error estimate is the difference of
    the last two estimates.
    """
    prev = None
    p = int(panels)
    val = np.nan
    err = np.inf
    for _ in range(max_level + 1):
        x, w = _composite_nodes(n_nodes, p, a, b)
        with np.errstate(all="ignore"):
            val = float(np.dot(w, np.asarray(f(x), dtype=float)))
        if prev is not None:
            err = abs(val - prev)
            if err <= max(atol, rtol * abs(val)):
                break
        prev = val
        p *= 2
    return val, err


def gauss_legendre_2d(
    f: Callable,
    *,
    n_nodes: int = 8,
    panels: int = 4,
    atol: float = 1e-12,
    rtol: float = 1e-10,
    max_level: int = 7,
) -> tuple[float, float]:
    r"""Tensor-product composite Gauss--Legendre rule on :math:`[0,1]^2`.

    Panels are doubled in both directions until successive estimates agree
    to ``max(atol, rtol*|I|)`` or ``max_level`` doublings were performed.

    Returns
    -------
    (value, error_estimate)
    """
    prev = None
    p = int(panels)
    val = np.nan
    err = np.inf
    for _ in range(max_level + 1):
        x, w = _composite_nodes(n_nodes, p)
        uu = np.repeat(x, x.size)
        vv = np.tile(x, x.size)
        ww = np.repeat(w, w.size) * np.tile(w, w.size)
        with np.errstate(all="ignore"):
            val = float(np.dot(ww, np.asarray(f(uu, vv), dtype=float)))
        if prev is not None:
            err = abs(val - prev)
            if err <= max(atol, rtol * abs(val)):
                break
        prev = val
        p *= 2
    return val, err
