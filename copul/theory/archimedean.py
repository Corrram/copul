r"""
Archimedean copula theory.

Functions for *any* bivariate copula object of :mod:`copul` (closed forms are
used for Archimedean copulas, numerical integration otherwise) and for
Archimedean generators:

* the **Kendall distribution** :math:`K_C(t)=P(C(U,V)\le t)` and its
  quantile function (:func:`kendall_distribution`,
  :func:`kendall_distribution_inverse`);
* **generator theory**: validity of a generator, strictness, the zero curve
  and its singular mass, :math:`d`-monotonicity of the inverse generator,
  complete monotonicity and the maximal dimension
  (:func:`check_generator`, :func:`generator_properties`,
  :func:`max_dimension`, :func:`zero_curve`);
* **characterisation** of Archimedean copulas by associativity and the
  diagonal (:func:`associativity_defect`, :func:`is_archimedean`);
* **constructions** of numerical Archimedean copulas from a Kendall
  distribution, a Laplace transform or a numerical generator
  (:func:`archimedean_from_kendall_distribution`,
  :func:`from_laplace_transform`, :func:`archimedean_from_generator`).

Notation
--------
:math:`\varphi` is the generator, :math:`\psi=\varphi^{[-1]}` its
pseudo-inverse, :math:`C(u,v)=\psi(\varphi(u)+\varphi(v))`.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Ch. 4 and Sec. 5.1.
Genest, C. & Rivest, L.-P. (1993). Statistical inference procedures for
bivariate Archimedean copulas. *JASA* 88, 1034--1043.
Barbe, P., Genest, C., Ghoudi, K. & Rémillard, B. (1996). On Kendall's
process. *J. Multivariate Anal.* 58, 197--229.
McNeil, A. J. & Nešlehová, J. (2009). Multivariate Archimedean copulas,
d-monotone functions and l1-norm symmetric distributions. *Ann. Statist.*
37, 3059--3097.
Ling, C.-H. (1965). Representation of associative functions. *Publ. Math.
Debrecen* 12, 189--212.
Joe, H. (2014). *Dependence Modeling with Copulas*, CRC Press, Ch. 3--4.
Kimberling, C. H. (1974). A probabilistic interpretation of complete
monotonicity. *Aequationes Math.* 10, 152--164.
Williamson, R. E. (1956). Multiply monotone functions and their Laplace
transforms. *Duke Math. J.* 23, 189--207.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from copul.family.archimedean.numeric_archimedean import (
    ArchimedeanGenerator,
    NumericArchimedeanCopula,
    _call,
    fd_derivative,
    solve_increasing,
)

__all__ = [
    "GeneratorReport",
    "ZeroCurve",
    "archimedean_from_generator",
    "archimedean_from_kendall_distribution",
    "archimedean_generator",
    "associativity_defect",
    "check_generator",
    "from_laplace_transform",
    "generator_properties",
    "is_archimedean",
    "kendall_distribution",
    "kendall_distribution_inverse",
    "max_dimension",
    "zero_curve",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _finish(out, scalar: bool):
    out = np.asarray(out, dtype=float)
    return float(out.reshape(-1)[0]) if scalar else out


def _require_specified(C, what: str) -> None:
    from copul.measures.backend import free_parameters

    free = free_parameters(C)
    if free:
        raise ValueError(
            f"{what} needs a fully specified copula; {type(C).__name__} has free parameters {free}."
        )


def _backend(C):
    from copul.measures.backend import numeric_backend

    return numeric_backend(C)


def _is_type(C, module: str, name: str) -> bool:
    try:
        mod = __import__(module, fromlist=[name])
        return isinstance(C, getattr(mod, name))
    except Exception:  # pragma: no cover - defensive
        return False


def _is_upper_frechet(C) -> bool:
    return _is_type(C, "copul.family.frechet.upper_frechet", "UpperFrechet")


def _is_lower_frechet(C) -> bool:
    return _is_type(C, "copul.family.frechet.lower_frechet", "LowerFrechet")


def _w_generator() -> ArchimedeanGenerator:
    return ArchimedeanGenerator(
        phi=lambda t: 1.0 - np.asarray(t, float),
        dphi=lambda t: -np.ones_like(np.asarray(t, float)),
        psi=lambda s: np.clip(1.0 - np.asarray(s, float), 0.0, 1.0),
        phi0=1.0,
        d2phi=lambda t: np.zeros_like(np.asarray(t, float)),
        ratio=lambda t: np.asarray(t, float) - 1.0,
        source="closed",
    )


def archimedean_generator(C) -> ArchimedeanGenerator | None:
    r"""Numeric generator data of an Archimedean copula object, or ``None``.

    Recognises the Archimedean families of :mod:`copul` (symbolic generator,
    lambdified once per parameter value), Joe's Laplace-transform families
    (``BB1``, ...; evaluated on the log scale), numerical Archimedean copulas
    and the lower Fréchet bound :math:`W` (generator :math:`1-t`).  Copulas
    that are not *declared* Archimedean return ``None`` (use
    :func:`is_archimedean` for a numerical test).

    Parameters
    ----------
    C : copula
        A fully specified bivariate copula.

    Returns
    -------
    ArchimedeanGenerator or None
    """
    hook = getattr(C, "_archimedean_generator", None)
    if callable(hook):
        return hook()
    if _is_lower_frechet(C):
        return _w_generator()
    return None


# ---------------------------------------------------------------------------
# Kendall distribution
# ---------------------------------------------------------------------------


def _kendall_closed(gen: ArchimedeanGenerator, t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    k0 = gen.zero_curve_mass()
    with np.errstate(all="ignore"):
        k = t - gen.phi_over_dphi(t)
    k = np.where(np.isfinite(k), k, np.where(t < 0.5, k0, 1.0))
    k = np.where(t <= 0.0, k0, np.where(t >= 1.0, 1.0, k))
    return np.clip(np.maximum(k, k0), t, 1.0)


def _bisect_level(cdf, u, thr, lo, hi, n_iter: int = 60):
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        below = np.asarray(cdf(u, mid), float) <= thr
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
        if np.all(hi - lo <= 2e-16):
            break
    return hi


def _level_curve(cdf, u, t, n_iter: int = 60):
    r"""Upper end :math:`\sup\{v: C(u,v)\le t\}` of the level set (from above).

    Illinois iterations on :math:`v\mapsto C(u,v)-t` for :math:`t>0`, then
    bisection wherever the level set has a flat part (zero rectangle mass)
    beyond the root found; pure bisection for :math:`t=0` (the zero set).
    """
    u = np.asarray(u, dtype=float)
    t = np.broadcast_to(np.asarray(t, dtype=float), u.shape)
    thr = t + 4e-16 * t
    lo = np.minimum(t, u).astype(float)
    hi = np.ones_like(u, dtype=float)
    out = np.empty_like(u)
    zero = t <= 0.0
    if np.any(zero):
        out[zero] = _bisect_level(cdf, u[zero], thr[zero], lo[zero], hi[zero], n_iter)
    pos = np.flatnonzero(~zero)
    if pos.size:
        up, tp = u[pos], thr[pos]

        def g(v, idx):
            return np.asarray(cdf(up[idx], v), float) - tp[idx]

        root = solve_increasing(g, lo[pos], hi[pos], xtol=1e-15)
        above = np.minimum(root + 4e-15, 1.0)
        flat = np.asarray(cdf(up, above), float) <= tp
        if np.any(flat):
            above[flat] = _bisect_level(cdf, up[flat], tp[flat], above[flat], hi[pos][flat], n_iter)
        out[pos] = above
    return out


def _kendall_numeric(C, t: np.ndarray, rtol: float, atol: float) -> np.ndarray:
    from copul.measures.quadrature import integrate_1d_batch

    be = _backend(C)
    cdf, h1 = be.cdf, be.h1
    t = np.asarray(t, dtype=float)
    flat = t.ravel()
    out = np.where(flat >= 1.0, 1.0, 0.0)
    rows = np.flatnonzero((flat >= 0.0) & (flat < 1.0))
    if rows.size:
        tr = flat[rows]

        def f(u, r):
            tt = tr[r]
            # right limit of the conditional distribution function at the level
            # curve (includes singular mass on the curve); the small offset avoids
            # evaluating family formulas exactly on their kinks (Heaviside(0) = 1/2)
            v = _level_curve(cdf, u, tt)
            v = np.minimum(v * (1.0 + 1e-10), 1.0)
            return np.asarray(h1(u, v), float)

        vals, _ = integrate_1d_batch(f, tr, np.ones_like(tr), atol=atol, rtol=rtol)
        out[rows] = np.clip(tr + vals, tr, 1.0)
    return out.reshape(t.shape)


def _kendall_mc(C, t, n_samples: int, random_state) -> np.ndarray:
    X = np.asarray(C.rvs(int(n_samples), random_state=random_state), dtype=float)
    w = np.sort(np.asarray(_backend(C).cdf(X[:, 0], X[:, 1]), float))
    t = np.asarray(t, dtype=float)
    return np.searchsorted(w, t.ravel(), side="right").reshape(t.shape) / w.size


def kendall_distribution(
    C,
    t,
    *,
    method: str = "auto",
    rtol: float = 1e-9,
    atol: float = 1e-11,
    n_samples: int = 100_000,
    random_state=None,
):
    r"""Kendall distribution function :math:`K_C(t) = P\bigl(C(U,V)\le t\bigr)`.

    For :math:`(U,V)\sim C`, :math:`K_C` is the distribution function of the
    probability integral transform :math:`C(U,V)`; it satisfies
    :math:`t\le K_C(t)\le 1`, :math:`K_M(t)=t`, :math:`K_W\equiv 1`,
    :math:`K_\Pi(t)=t-t\log t` and

    .. math::

       \tau_C = 4\,E[C(U,V)] - 1 = 3 - 4\int_0^1 K_C(t)\,dt .

    **Archimedean copulas** (closed form, Genest & Rivest 1993; Nelsen 2006,
    Sec. 4.3):

    .. math::

       K_C(t) = t - \frac{\varphi(t)}{\varphi'(t^+)},\qquad
       K_C(0) = -\frac{\varphi(0)}{\varphi'(0^+)}

    (the mass of the zero curve).  **Any copula** (``method="numeric"``):
    since :math:`C(u,v)\le u`, conditioning on :math:`U=u` gives

    .. math::

       K_C(t) = t + \int_t^1 \partial_1 C\bigl(u, v_t(u)\bigr)\,du,\qquad
       v_t(u) = \sup\{v : C(u,v)\le t\},

    evaluated by adaptive Gauss--Kronrod quadrature (vectorised over all
    :math:`t`) with the level curve :math:`v_t` found by bisection and
    :math:`\partial_1 C` taken as the right-continuous conditional
    distribution function (so singular mass *on* the level curve, e.g. on
    the zero curve of a non-strict Archimedean copula, is included).

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    t : float or array_like
        Points in :math:`[0,1]`.
    method : {"auto", "closed", "numeric", "mc"}
        ``"auto"`` uses the closed form for Archimedean copulas (and
        :math:`M`), the integral representation otherwise; ``"mc"`` is the
        empirical distribution of :math:`C(U_i,V_i)` on ``n_samples``
        samples.
    rtol, atol : float
        Tolerances of the quadrature.
    n_samples, random_state
        Monte Carlo options.

    Returns
    -------
    float or numpy.ndarray

    References
    ----------
    Genest & Rivest (1993), Sec. 2; Nelsen (2006), Sec. 4.3 and Sec. 5.1;
    Barbe, Genest, Ghoudi & Rémillard (1996).

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.archimedean import kendall_distribution
    >>> round(kendall_distribution(cp.Clayton(2), 0.3), 12)  # t + t(1-t^2)/2
    0.4365
    """
    _require_specified(C, "kendall_distribution")
    method = str(method).lower()
    if method not in ("auto", "closed", "numeric", "mc"):
        raise ValueError(f"unknown method {method!r}")
    scalar = np.ndim(t) == 0
    tt = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    if method == "mc":
        return _finish(_kendall_mc(C, tt, n_samples, random_state), scalar)
    if method in ("auto", "closed"):
        if _is_upper_frechet(C):
            return _finish(tt, scalar)
        gen = archimedean_generator(C)
        if gen is not None:
            return _finish(_kendall_closed(gen, tt), scalar)
        if method == "closed":
            raise NotImplementedError(
                f"No closed-form Kendall distribution for {type(C).__name__} "
                "(not an Archimedean copula object)."
            )
    return _finish(_kendall_numeric(C, tt, rtol, atol), scalar)


def kendall_distribution_inverse(C, p, *, method: str = "auto", xtol: float = 1e-13, **kwargs):
    r"""Quantile function :math:`K_C^{-1}(p)=\inf\{t: K_C(t)\ge p\}`.

    Vectorised Illinois iterations on :math:`\log t` (the Kendall
    distribution is evaluated in batches, see :func:`kendall_distribution`).
    Values :math:`p\le K_C(0)` (the mass of the zero set) map to ``0``.

    For an Archimedean copula, :math:`T=C(U,V)\sim K_C` is independent of
    :math:`S=\varphi(U)/(\varphi(U)+\varphi(V))\sim U(0,1)` (Genest & Rivest
    1993; Nelsen 2006, Sec. 4.3), so :math:`(\psi(S\varphi(T)), \psi((1-S)\varphi(T)))` with
    :math:`T=K_C^{-1}(Q)` samples :math:`C`.

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    p : float or array_like
        Probabilities in :math:`[0,1]`.
    method : str
        Passed to :func:`kendall_distribution`.
    xtol : float
        Tolerance in :math:`\log t`.
    **kwargs
        Further options of :func:`kendall_distribution`.

    Returns
    -------
    float or numpy.ndarray
    """
    scalar = np.ndim(p) == 0
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    flat = p.ravel()
    out = np.where(flat >= 1.0, 1.0, 0.0)
    k0 = float(kendall_distribution(C, 0.0, method=method, **kwargs))
    act = np.flatnonzero((flat > k0) & (flat < 1.0))
    if act.size:
        pa = flat[act]
        # bracket on a geometric grid (one batched evaluation; K(t) >= t gives
        # K^{-1}(p) <= p), then Illinois iterations in log t
        grid = np.concatenate([[1e-300], np.geomspace(1e-16, 1.0, 33)])
        kg = np.empty(grid.size)
        kg[0] = k0  # (the numerical K is not resolved below the cdf's accuracy)
        kg[1:] = np.asarray(kendall_distribution(C, grid[1:], method=method, **kwargs), float)
        kg = np.maximum.accumulate(kg)
        j = np.clip(np.searchsorted(kg, pa, side="left"), 1, grid.size - 1)
        lo, hi = np.log(grid[j - 1]), np.log(grid[j])

        def g(y, idx):
            return np.asarray(kendall_distribution(C, np.exp(y), method=method, **kwargs)) - pa[idx]

        y = solve_increasing(g, lo, hi, kg[j - 1] - pa, kg[j] - pa, xtol=xtol)
        out[act] = np.exp(y)
    return _finish(out.reshape(p.shape), scalar)


# ---------------------------------------------------------------------------
# generator theory
# ---------------------------------------------------------------------------


@dataclass
class GeneratorReport:
    r"""Properties of an Archimedean generator (see :func:`check_generator`).

    Attributes
    ----------
    valid : bool
        :math:`\varphi` generates a bivariate copula: continuous, strictly
        decreasing, convex, :math:`\varphi(1)=0` (Nelsen 2006, Thm. 4.1.4).
    vanishes_at_one, decreasing, convex : bool
        The individual conditions.
    strict : bool
        :math:`\varphi(0)=\infty`.
    phi0 : float
        :math:`\varphi(0^+)`.
    zero_curve_mass : float
        Singular mass :math:`-\varphi(0)/\varphi'(0^+)` on the zero curve
        (Nelsen 2006, Sec. 4.3); ``0`` for strict generators.
    d_monotone : dict
        ``{d: bool}`` -- whether :math:`\psi` is :math:`d`-monotone on
        :math:`[0,\infty)`, i.e. generates a :math:`d`-dimensional copula
        (McNeil & Nešlehová 2009, Thm. 2.2), for :math:`d=2,\dots,d_{max}`.
    max_dimension : int or float
        Largest such :math:`d` (``inf`` if :math:`\psi` is known to be
        completely monotone; ``d_max`` if all checked orders pass).
    completely_monotone : bool or None
        ``True`` if :math:`\psi` is known to be a Laplace transform,
        ``False`` if some order fails, ``None`` if undecided.
    method : str
        ``"taylor"`` (high-precision Taylor series of :math:`\psi` from a
        symbolic :math:`\varphi`), ``"differences"`` (finite differences of
        a numerical :math:`\psi`) and/or ``"known"`` (family results).
    details : dict
        Diagnostics (violations, failing orders, references).
    """

    valid: bool
    vanishes_at_one: bool
    decreasing: bool
    convex: bool
    strict: bool
    phi0: float
    zero_curve_mass: float
    d_monotone: dict = field(default_factory=dict)
    max_dimension: float = 1
    completely_monotone: bool | None = None
    method: str = ""
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.valid)


def _t_grid() -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                np.geomspace(1e-12, 1e-2, 21),
                np.linspace(0.01, 0.99, 99),
                1.0 - np.geomspace(1e-2, 1e-9, 15),
            ]
        )
    )


def _basic_checks(phi, dphi, phi0: float, tol: float, d2phi=None) -> tuple[dict, dict]:
    r""":math:`\varphi(1^-)=0`, strictly decreasing and convex on a fine grid.

    Monotonicity and convexity are checked with the derivatives when they
    are given (symbolic generators), otherwise with divided differences on
    :math:`[10^{-6}, 1-10^{-6}]` with a relative tolerance.
    """
    t = _t_grid()
    f = _call(phi, t)
    f1 = float(_call(phi, np.array([1.0]))[0])
    if not np.isfinite(f1):  # removable singularity at t = 1
        f1 = float(_call(phi, np.array([1.0 - 1e-14]))[0])
    finite = np.isfinite(f)
    scale = max(1.0, float(np.max(np.abs(f[finite]))) if np.any(finite) else 1.0)
    vanishes = bool(np.isfinite(f1) and abs(f1) <= 1e-8 * scale)
    df = np.diff(f[finite])
    pos = f[finite] > 1e-290  # ignore underflow to 0 next to t = 1
    decreasing = bool(np.all(df <= 1e-14 * scale) and np.all(np.diff(f[finite][pos][::10]) < 0))
    if dphi is not None:
        d = _call(dphi, t)
        okd = np.isfinite(d)
        decreasing = decreasing and bool(np.all(d[okd] <= 0) and np.all(d[okd & (f > 1e-290)] < 0))
    if d2phi is not None:
        d2 = _call(d2phi, t)
        d1 = np.abs(_call(dphi, t)) if dphi is not None else np.ones_like(t)
        ok = np.isfinite(d2) & np.isfinite(d1)
        rel = d2[ok] / np.maximum(d1[ok], 1.0)
        conv_viol = float(max(0.0, -np.min(rel))) if rel.size else 0.0
    else:
        inner = finite & (t >= 1e-6) & (t <= 1 - 1e-6)
        tf, ff = t[inner], f[inner]
        slopes = np.diff(ff) / np.diff(tf)
        ds = np.diff(slopes)
        conv_viol = (
            float(max(0.0, -np.min(ds / np.maximum(np.abs(slopes[1:]), 1.0)))) if ds.size else 0.0
        )
        tol = max(tol, 1e-6)
    convex = bool(conv_viol <= tol)
    checks = {"vanishes_at_one": vanishes, "decreasing": decreasing, "convex": convex}
    details = {"phi_at_1": f1, "convexity_violation": conv_viol}
    return checks, details


# -- high-precision Taylor series of psi -----------------------------------------------


def _series_div(a, b, n):
    c = []
    for k in range(n + 1):
        s = a[k] if k < len(a) else 0
        for j in range(1, min(k, len(b) - 1) + 1):
            s -= b[j] * c[k - j]
        c.append(s / b[0])
    return c


def _psi_derivatives_mp(phi_mp, t0, n: int):
    r""":math:`\psi^{(k)}(\varphi(t_0))`, :math:`k=1..n`, via Taylor series in :math:`t`.

    With :math:`D_1=1/\varphi'` and :math:`D_{k+1}=D_k'/\varphi'`,
    :math:`D_k(t)=\psi^{(k)}(\varphi(t))`; the recursion is carried out on
    truncated Taylor series around :math:`t_0` (coefficients of
    :math:`\varphi` from high-precision finite differences).
    """
    import mpmath as mp

    # central steps t0 +- (n+1) h must stay inside (0, 1): h relative to the distance
    h = min(t0, 1 - t0) * mp.ldexp(1, -mp.mp.prec - 10)
    co = mp.taylor(phi_mp, t0, n + 1, h=h)
    dphi = [co[k + 1] * (k + 1) for k in range(n + 1)]
    D = _series_div([mp.mpf(1)], dphi, n)
    out = [D[0]]
    for _ in range(2, n + 1):
        der = [D[j + 1] * (j + 1) for j in range(len(D) - 1)]
        D = _series_div(der, dphi, len(der) - 1)
        out.append(D[0])
    return [mp.re(x) for x in out]


def _taylor_d_monotone(expr, t, d_max: int, strict: bool, dps: int = 30):
    """Max d (sign conditions and boundary conditions at phi(0)) from a symbolic phi."""
    import mpmath as mp

    f = sp.lambdify(t, expr, modules="mpmath")
    n = int(d_max)
    with mp.workdps(dps):
        grid = (
            [mp.mpf(10) ** (-k) for k in (1, 2, 3, 4, 6, 8, 12, 16, 24)]
            + [mp.mpf(i) / 40 for i in range(1, 40)]
            + [1 - mp.mpf(10) ** (-k) for k in (2, 3, 4, 6, 8)]
        )
        cols: list[list] = [[] for _ in range(n)]
        failed_points = 0
        for x in grid:
            try:
                ders = _psi_derivatives_mp(f, x, n)
            except Exception:
                failed_points += 1
                continue
            if not all(mp.isfinite(dv) for dv in ders):
                failed_points += 1
                continue
            for k in range(n):
                cols[k].append(ders[k])
        if not cols[0]:
            raise ValueError("high-precision evaluation of the generator failed")
        sign_ok = []
        scale = mp.mpf(1)
        for k in range(1, n + 1):
            col = cols[k - 1]
            # identically vanishing derivatives (e.g. polynomial psi) are rounding noise
            # relative to the size of the lower-order derivatives
            scale = max(scale, max(abs(c) for c in col))
            thr = scale * mp.mpf(10) ** (-(dps - 12))
            sign_ok.append(bool(min(((-1) ** k) * c for c in col) >= -thr))
        # boundary conditions psi^{(k)}(phi(0)-) = 0, k <= d - 2 (non-strict only)
        bc_ok = [True] * n
        if not strict and n >= 3:
            pts = [mp.mpf(10) ** (-30), mp.mpf(10) ** (-60), mp.mpf(10) ** (-120)]
            try:
                vals = [_psi_derivatives_mp(f, x, n - 2) for x in pts]
                for k in range(1, n - 1):
                    a, b = abs(vals[1][k - 1]), abs(vals[2][k - 1])
                    bc_ok[k - 1] = bool(b <= mp.mpf(10) ** -25 or b <= 1e-3 * a)
            except Exception:
                bc_ok = [False] * n
    d_mono = {}
    for d in range(2, n + 1):
        signs = all(sign_ok[:d])
        bcs = all(bc_ok[: max(d - 2, 0)])
        d_mono[d] = bool(signs and bcs)
    return d_mono, {"sign_ok": sign_ok, "boundary_ok": bc_ok, "failed_points": failed_points}


# -- finite differences of psi -------------------------------------------------------


def _difference_d_monotone(psi, phi0: float, d_max: int, scale: float, tol: float = 1e-12):
    r"""Max d from :math:`(-1)^k\Delta_h^k\psi(x)\ge 0`, :math:`k\le d`.

    A continuous function on :math:`[0,\infty)` is :math:`d`-monotone iff all
    differences of order :math:`k\le d` alternate in sign (Williamson 1956;
    McNeil & Nešlehová 2009, Sec. 2).
    """
    from scipy.special import comb

    if math.isfinite(phi0):
        x = np.unique(np.concatenate([[0.0], phi0 * np.linspace(0.0, 1.0, 81)[1:]]))
        h = phi0 * np.array([0.005, 0.02, 0.05, 0.1, 0.2])
    else:
        x = np.unique(np.concatenate([[0.0], scale * np.geomspace(1e-3, 50.0, 60)]))
        h = scale * np.array([0.01, 0.05, 0.2, 0.5, 1.0, 2.0])
    X, H = np.meshgrid(x, h, indexing="ij")
    X, H = X.ravel(), H.ravel()
    vals = [_call(psi, X + j * H) for j in range(d_max + 1)]
    vals = [np.where(np.isfinite(v), v, 0.0) for v in vals]
    ok = []
    worst = []
    for k in range(1, d_max + 1):
        delta = sum(((-1) ** (k - j)) * comb(k, j, exact=True) * vals[j] for j in range(k + 1))
        val = ((-1) ** k) * delta
        thr = tol * 2.0**k
        worst.append(float(np.min(val)))
        ok.append(bool(np.min(val) >= -thr))
    d_mono = {d: bool(all(ok[:d])) for d in range(2, d_max + 1)}
    return d_mono, {"difference_ok": ok, "min_differences": worst}


def _to_expr(phi):
    expr = sp.sympify(phi)
    syms = sorted(expr.free_symbols, key=str)
    if not syms:
        raise ValueError("the generator must depend on a variable t")
    t = next((s for s in syms if str(s) == "t"), None)
    if t is None:
        if len(syms) > 1:
            raise ValueError(f"cannot identify the variable among {syms}; use t")
        t = syms[0]
    free = [s for s in syms if s != t]
    if free:
        raise ValueError(f"the generator has free parameters {free}")
    return expr, t


def _phi0_symbolic(expr, t, phi_np) -> float:
    with np.errstate(all="ignore"):
        v = float(np.asarray(phi_np(np.array([0.0])), float)[0])
    if np.isfinite(v):
        return v
    try:
        lim = sp.limit(expr, t, 0, "+")
        if lim.is_finite:
            return float(lim)
    except Exception:
        pass
    return math.inf


def _report(checks, details, strict, phi0, mass, d_mono, method, known=None):
    valid = bool(checks["vanishes_at_one"] and checks["decreasing"] and checks["convex"])
    if not valid:
        d_mono = {d: False for d in d_mono}
    maxd: float = 1
    for d in sorted(d_mono):
        if d_mono[d]:
            maxd = d
        else:
            break
    cm = None if (d_mono and all(d_mono.values())) else (False if d_mono else None)
    if known is not None and valid:
        kd, kcm, ref = known
        details = {**details, "known": ref}
        method = f"{method}+known" if method else "known"
        if kcm:
            maxd, cm = math.inf, True
        elif kd is not None:
            maxd, cm = kd, False
            d_mono = {d: d <= kd for d in d_mono}
    return GeneratorReport(
        valid=valid,
        vanishes_at_one=checks["vanishes_at_one"],
        decreasing=checks["decreasing"],
        convex=checks["convex"],
        strict=strict,
        phi0=phi0,
        zero_curve_mass=mass,
        d_monotone=d_mono,
        max_dimension=maxd,
        completely_monotone=cm,
        method=method,
        details=details,
    )


def check_generator(
    phi,
    *,
    psi: Callable | None = None,
    dphi: Callable | None = None,
    d_max: int = 10,
    tol: float = 1e-10,
) -> GeneratorReport:
    r"""Check whether :math:`\varphi` is an Archimedean generator and how far it goes.

    * :math:`\varphi` generates a bivariate copula iff it is continuous,
      strictly decreasing and convex with :math:`\varphi(1)=0`
      (Nelsen 2006, Thm. 4.1.4);
    * it is *strict* iff :math:`\varphi(0)=\infty`; otherwise the zero curve
      :math:`\varphi(u)+\varphi(v)=\varphi(0)` carries the mass
      :math:`-\varphi(0)/\varphi'(0^+)` (Nelsen 2006, Sec. 4.3);
    * :math:`\psi=\varphi^{[-1]}` generates a :math:`d`-dimensional copula
      iff it is :math:`d`-monotone on :math:`[0,\infty)`:
      :math:`(-1)^k\psi^{(k)}\ge0` for :math:`k\le d-2` and
      :math:`(-1)^{d-2}\psi^{(d-2)}` is nonincreasing and convex
      (McNeil & Nešlehová 2009, Thm. 2.2); all :math:`d` iff :math:`\psi`
      is completely monotone, i.e. a Laplace transform (Kimberling 1974;
      Bernstein's theorem).

    For a **symbolic** generator (SymPy expression or string in ``t``) the
    derivatives :math:`\psi^{(k)}(\varphi(t))` are evaluated in high
    precision through truncated Taylor series in :math:`t`
    (:math:`D_1=1/\varphi'`, :math:`D_{k+1}=D_k'/\varphi'`) on a grid of
    :math:`t\in(0,1)` -- i.e. on the whole support of :math:`\psi` -- and, for
    non-strict generators, the boundary conditions
    :math:`\psi^{(k)}(\varphi(0)^-)=0`, :math:`k\le d-2`, are verified.  For
    a **numerical** (callable) generator the alternating signs of the
    differences :math:`(-1)^k\Delta_h^k\psi(x)` are checked.

    Parameters
    ----------
    phi : str, sympy.Expr or callable
        The generator.
    psi : callable, optional
        Inverse generator (numerical path; computed by inversion otherwise).
    dphi : callable, optional
        :math:`\varphi'` (numerical path).
    d_max : int
        Highest dimension checked.
    tol : float
        Tolerance of the convexity check.

    Returns
    -------
    GeneratorReport

    Examples
    --------
    >>> from copul.theory.archimedean import check_generator
    >>> rep = check_generator("(t**(-1/2) - 1) * 2")      # Clayton(1/2)
    >>> rep.valid, rep.strict, rep.max_dimension
    (True, True, 10)
    >>> rep = check_generator("(1 - t**(3/10)) / (3/10)")  # Clayton(-3/10)
    >>> rep.strict, rep.max_dimension                     # d <= 1 - 1/theta
    (False, 4)
    >>> check_generator("t * (1 - t)").valid               # not decreasing
    False
    """
    from copul.family.core.biv_core_copula import BivCoreCopula

    if isinstance(phi, BivCoreCopula):  # a copula object: use its generator
        return generator_properties(phi, d_max=d_max, tol=tol)
    from copul.numerics import to_numpy_callable

    d_max = max(int(d_max), 2)
    if isinstance(phi, (str, sp.Basic)):
        expr, t = _to_expr(phi)
        phi_np = to_numpy_callable(expr, [t])
        dexpr = sp.diff(expr, t)
        dphi_np = to_numpy_callable(dexpr, [t])
        d2phi_np = to_numpy_callable(sp.diff(expr, t, 2), [t])
        phi0 = _phi0_symbolic(expr, t, phi_np)
        strict = not math.isfinite(phi0)
        checks, details = _basic_checks(phi_np, dphi_np, phi0, tol, d2phi_np)
        gen = ArchimedeanGenerator(phi=phi_np, dphi=dphi_np, psi=lambda s: s, phi0=phi0)
        mass = gen.zero_curve_mass()
        try:
            d_mono, info = _taylor_d_monotone(expr, t, d_max, strict)
            method = "taylor"
        except Exception as e:  # pragma: no cover - fall back to differences
            C = NumericArchimedeanCopula(phi=phi_np, dphi=dphi_np, phi0=phi0)
            scale = float(np.nan_to_num(C._phi(np.array([0.5]))[0], nan=1.0)) or 1.0
            d_mono, info = _difference_d_monotone(C._psi, phi0, d_max, scale)
            info["taylor_error"] = repr(e)
            method = "differences"
        details.update(info)
        return _report(checks, details, strict, phi0, mass, d_mono, method)
    if not callable(phi):
        raise TypeError("phi must be a string, a SymPy expression or a callable")
    C = NumericArchimedeanCopula(phi=phi, psi=psi, dphi=dphi)
    return _numeric_report(C, d_max, tol)


def _numeric_report(C: NumericArchimedeanCopula, d_max: int, tol: float, known=None):
    gen = C._archimedean_generator()
    checks, details = _basic_checks(gen.phi, gen.dphi, gen.phi0, tol, C._d2phi_user)
    scale = float(np.nan_to_num(gen.phi(np.array([0.5]))[0], nan=1.0)) or 1.0
    d_mono, info = _difference_d_monotone(gen.psi, gen.phi0, d_max, scale)
    details.update(info)
    return _report(
        checks,
        details,
        gen.is_strict,
        gen.phi0,
        gen.zero_curve_mass(),
        d_mono,
        "differences",
        known,
    )


def _param(C, name: str):
    try:
        return float(getattr(C, name))
    except Exception:
        return None


def _known_dimension(C):
    """Family results ``(max_d, completely_monotone, reference)`` or ``None``."""
    name = type(C).__name__
    lt_ref = "psi is a Laplace transform (Joe 2014, Ch. 4)"
    try:
        from copul.family.bb.lt_archimedean import LTArchimedeanCopula

        if isinstance(C, LTArchimedeanCopula):
            return None, True, lt_ref
    except Exception:  # pragma: no cover
        pass
    th = _param(C, "theta")
    if name in ("BivClayton", "Clayton", "Nelsen1", "MultivariateClayton") and th is not None:
        if th > 0:
            return None, True, "Clayton: Gamma(1/theta) frailty (Joe 2014, Ch. 4)"
        if th < 0:
            d = math.floor(1.0 - 1.0 / th + 1e-12)
            return (
                d,
                False,
                "Clayton, theta<0: d-monotone iff theta >= -1/(d-1) (McNeil & Neslehova 2009)",
            )
    if name in ("GumbelHougaard", "Nelsen4"):
        return None, True, "Gumbel: positive stable frailty (Joe 2014, Ch. 4)"
    if name in ("Frank", "Nelsen5") and th is not None and th > 0:
        return None, True, "Frank, theta>0: logarithmic frailty (Joe 2014, Ch. 4)"
    if name in ("Joe", "Nelsen6"):
        return None, True, "Joe: Sibuya frailty (Joe 2014, Ch. 4)"
    if name in ("AliMikhailHaq", "Nelsen3") and th is not None and th >= 0:
        return None, True, "AMH, theta>=0: geometric frailty (Joe 2014, Ch. 4)"
    if name in ("Nelsen12", "Nelsen14"):
        return None, True, "BB1 subfamily: Laplace transform (Joe 2014, Ch. 4)"
    if name == "BivIndependenceCopula":
        return None, True, "independence: psi(s) = exp(-s)"
    if _is_lower_frechet(C):
        return 2, False, "W is Archimedean only for d = 2"
    return None


def _symbolic_generator_expr(C):
    """``(expr, t)`` of the raw symbolic generator with parameters substituted, or None."""
    if _is_lower_frechet(C):
        t = sp.Symbol("t", positive=True)
        return 1 - t, t
    t = getattr(C, "t", None)
    try:
        from copul.family.bb.lt_archimedean import LTArchimedeanCopula

        if isinstance(C, LTArchimedeanCopula):
            expr = C._phi_sym(C.t, *C._sym_params())
            return expr, C.t
    except Exception:
        pass
    try:
        g = C.generator
    except Exception:
        return None
    expr = getattr(g, "func", None)
    if not isinstance(expr, sp.Expr) or t is None:
        return None
    if isinstance(expr, sp.Piecewise) and len(expr.args) == 2:
        expr = expr.args[0][0]
    if expr.free_symbols - {t}:
        return None
    return expr, t


def generator_properties(C, d_max: int = 10, tol: float = 1e-10) -> GeneratorReport:
    r"""Generator properties of an Archimedean copula object.

    Like :func:`check_generator` applied to the generator of ``C`` (its
    symbolic generator when available, otherwise the numerical one),
    combined with standard family results: inverse generators that are
    Laplace transforms (Clayton :math:`\theta>0`, Gumbel--Hougaard, Frank
    :math:`\theta>0`, Joe, Ali--Mikhail--Haq :math:`\theta\ge0`, the BB
    families, Nelsen 12/14) are completely monotone, and the Clayton
    generator with :math:`\theta<0` is :math:`d`-monotone iff
    :math:`\theta\ge -1/(d-1)`, i.e. :math:`d\le 1-1/\theta`
    (McNeil & Nešlehová 2009).

    Parameters
    ----------
    C : copula
        Archimedean copula (raises ``TypeError`` otherwise).
    d_max : int
        Highest dimension checked numerically.
    tol : float
        Tolerance of the convexity check.

    Returns
    -------
    GeneratorReport
    """
    _require_specified(C, "generator_properties")
    gen = archimedean_generator(C)
    if gen is None:
        raise TypeError(f"{type(C).__name__} is not an Archimedean copula object.")
    known = _known_dimension(C)
    if known is not None and known[1]:
        # completely monotone: only the basic checks are needed
        checks, details = _basic_checks(gen.phi, gen.dphi, gen.phi0, tol, gen.d2phi)
        d_mono = {d: True for d in range(2, max(int(d_max), 2) + 1)}
        return _report(
            checks, details, gen.is_strict, gen.phi0, gen.zero_curve_mass(), d_mono, "", known
        )
    sym = _symbolic_generator_expr(C)
    if sym is not None:
        rep = check_generator(sym[0], d_max=d_max, tol=tol)
        if known is not None:
            rep = _report(
                {
                    "vanishes_at_one": rep.vanishes_at_one,
                    "decreasing": rep.decreasing,
                    "convex": rep.convex,
                },
                rep.details,
                rep.strict,
                rep.phi0,
                rep.zero_curve_mass,
                rep.d_monotone,
                rep.method,
                known,
            )
        return rep
    numeric = (
        C
        if isinstance(C, NumericArchimedeanCopula)
        else NumericArchimedeanCopula(
            phi=gen.phi, psi=gen.psi, dphi=gen.dphi, d2phi=gen.d2phi, phi0=gen.phi0
        )
    )
    return _numeric_report(numeric, max(int(d_max), 2), tol, known)


def max_dimension(C, d_max: int = 10, **kwargs) -> float:
    r"""Largest dimension :math:`d` for which the generator of ``C`` is valid.

    ``inf`` when the inverse generator is a Laplace transform (completely
    monotone), otherwise the largest :math:`d\le d_{max}` for which
    :math:`\psi` is :math:`d`-monotone (McNeil & Nešlehová 2009, Thm. 2.2),
    e.g. :math:`\lfloor 1-1/\theta\rfloor` for Clayton with
    :math:`\theta<0`.  ``d_max`` itself means "at least ``d_max``".

    Parameters
    ----------
    C : copula or str or sympy.Expr or callable
        An Archimedean copula object or a generator.
    d_max : int
        Highest dimension checked.

    Returns
    -------
    int or float
    """
    if isinstance(C, (str, sp.Basic)) or (callable(C) and not hasattr(C, "cdf")):
        return check_generator(C, d_max=d_max, **kwargs).max_dimension
    return generator_properties(C, d_max=d_max, **kwargs).max_dimension


# ---------------------------------------------------------------------------
# zero curve
# ---------------------------------------------------------------------------


@dataclass
class ZeroCurve:
    r"""Boundary of the zero set :math:`\{(u,v): C(u,v)=0\}`.

    The zero set is :math:`\{v\le z(u)\}` with the nonincreasing boundary
    :math:`z(u)=\sup\{v: C(u,v)=0\}`; for a non-strict Archimedean copula
    :math:`z(u)=\psi(\varphi(0)-\varphi(u))`, the level curve
    :math:`\varphi(u)+\varphi(v)=\varphi(0)`.

    Attributes
    ----------
    mass : float
        :math:`C`-measure of the zero curve, :math:`K_C(0)=P(C(U,V)=0)`;
        :math:`-\varphi(0)/\varphi'(0^+)` for Archimedean copulas
        (Nelsen 2006, Sec. 4.3).
    area : float
        Lebesgue measure :math:`\int_0^1 z(u)\,du` of the zero set.
    phi0 : float or None
        :math:`\varphi(0)` for Archimedean copulas.
    empty : bool
        Whether the zero set is null (strict generators, positive quadrant
        dependent copulas, ...).
    """

    mass: float
    area: float
    phi0: float | None
    empty: bool
    curve: Callable = field(repr=False, default=None)  # type: ignore[assignment]

    def __call__(self, u):
        r"""Boundary :math:`z(u)` (vectorised)."""
        scalar = np.ndim(u) == 0
        return _finish(self.curve(np.clip(np.asarray(u, float), 0.0, 1.0)), scalar)

    def contains(self, u, v, tol: float = 0.0):
        r"""Whether :math:`(u,v)` lies in the zero set (:math:`v\le z(u)`)."""
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        res = v <= self.curve(np.clip(u, 0.0, 1.0)) + tol
        return bool(res) if res.ndim == 0 else res


def zero_curve(C, *, n_iter: int = 60) -> ZeroCurve:
    r"""Zero curve of a copula and the singular mass on it.

    For a non-strict Archimedean copula the zero set
    :math:`\{\varphi(u)+\varphi(v)\ge\varphi(0)\}` has positive area and its
    boundary curve carries the singular mass
    :math:`-\varphi(0)/\varphi'(0^+)` (zero iff
    :math:`\varphi'(0^+)=-\infty`; Nelsen 2006, Thm. 4.1.11 and
    Cor. 4.1.12).  For other copulas the boundary
    :math:`z(u)=\sup\{v:C(u,v)=0\}` is found by bisection on the numerical
    cdf and the mass is :math:`K_C(0)` (:func:`kendall_distribution`).

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    n_iter : int
        Bisection steps of the numerical boundary.

    Returns
    -------
    ZeroCurve
    """
    from copul.measures.quadrature import integrate_1d

    _require_specified(C, "zero_curve")
    gen = archimedean_generator(C)
    if gen is not None:
        if gen.is_strict:
            return ZeroCurve(0.0, 0.0, math.inf, True, lambda u: np.zeros_like(u, dtype=float))
        phi0 = gen.phi0

        def curve(u):
            u = np.asarray(u, float)
            s = np.maximum(phi0 - _call(gen.phi, np.maximum(u, 1e-300)), 0.0)
            z = _call(gen.psi, s)
            return np.clip(np.where(u <= 0.0, 1.0, np.where(u >= 1.0, 0.0, z)), 0.0, 1.0)

        mass = gen.zero_curve_mass()
    else:
        cdf = _backend(C).cdf

        def curve(u):
            u = np.asarray(u, float)
            flat = u.ravel()
            z = _level_curve(cdf, np.maximum(flat, 1e-300), np.zeros_like(flat), n_iter)
            # hi end is above the boundary: step back if C(u, lo) > 0 everywhere
            z = np.where(np.asarray(cdf(flat, z), float) > 0, z - 2.0**-n_iter, z)
            z = np.where(flat <= 0.0, 1.0, np.where(flat >= 1.0, 0.0, np.maximum(z, 0.0)))
            return z.reshape(u.shape)

        mass = float(kendall_distribution(C, 0.0, method="numeric"))
        phi0 = None
    area, _ = integrate_1d(curve, atol=1e-12, rtol=1e-9)
    area = float(max(area, 0.0))
    return ZeroCurve(mass, area, phi0, area <= 1e-12 and mass <= 1e-12, curve)


# ---------------------------------------------------------------------------
# characterisation: associativity and the diagonal
# ---------------------------------------------------------------------------


def associativity_defect(C, n: int = 12, *, return_argmax: bool = False):
    r"""Maximal violation of associativity on a grid.

    .. math::

       \sup_{u,v,w}\bigl|C(C(u,v),w) - C(u,C(v,w))\bigr|

    over the interior grid :math:`\{i/(n+1)\}^3`.  Archimedean copulas,
    :math:`M` and ordinal sums of associative copulas are associative
    (defect zero up to rounding); Gaussian, Plackett and FGM copulas are not
    (Nelsen 2006, Sec. 4.1, Thm. 4.1.5).

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    n : int
        Grid points per axis.
    return_argmax : bool
        Also return the maximising point ``(u, v, w)``.

    Returns
    -------
    float or (float, tuple)
    """
    _require_specified(C, "associativity_defect")
    cdf = _backend(C).cdf
    g = np.arange(1, n + 1) / (n + 1.0)
    U, V, W = (a.ravel() for a in np.meshgrid(g, g, g, indexing="ij"))
    with np.errstate(all="ignore"):
        left = np.asarray(cdf(np.asarray(cdf(U, V), float), W), float)
        right = np.asarray(cdf(U, np.asarray(cdf(V, W), float)), float)
    diff = np.abs(left - right)
    k = int(np.nanargmax(diff))
    val = float(diff[k])
    if return_argmax:
        return val, (float(U[k]), float(V[k]), float(W[k]))
    return val


def _diagonal_gap(C, n: int = 400) -> tuple[float, float]:
    r"""Minimum of :math:`(t-\delta(t))/(t(1-t))` over :math:`(0,1)` and its location."""
    from scipy.optimize import minimize_scalar

    cdf = _backend(C).cdf

    def q(t):
        t = np.asarray(t, float)
        return (t - np.asarray(cdf(t, t), float)) / (t * (1.0 - t))

    t = np.concatenate(
        # C(t, t) close to t = 1 suffers from cancellation (e.g. generators with
        # phi(t) ~ (1 - t)^theta), so the grid stops at 0.999
        [np.geomspace(1e-6, 1e-2, 20), np.linspace(0.01, 0.99, n), 1 - np.geomspace(1e-2, 1e-3, 6)]
    )
    t = np.unique(t)
    vals = q(t)
    best_val = float(np.min(vals))
    best_t = float(t[int(np.argmin(vals))])
    # refine the smallest local minima (idempotents of ordinal sums lie between grid points)
    loc = np.flatnonzero((vals[1:-1] <= vals[:-2]) & (vals[1:-1] <= vals[2:])) + 1
    loc = loc[np.argsort(vals[loc])][:5]
    for i in loc:
        res = minimize_scalar(
            lambda x: float(q(np.array([x]))[0]),
            bounds=(t[i - 1], t[i + 1]),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if res.fun < best_val:
            best_val, best_t = float(res.fun), float(res.x)
    return best_val, best_t


def is_archimedean(
    C, tol: float = 1e-9, *, n: int = 12, diag_tol: float = 1e-6, return_details: bool = False
):
    r"""Numerical test whether ``C`` is an Archimedean copula.

    A copula is Archimedean iff it is associative and its diagonal satisfies
    :math:`\delta_C(t)=C(t,t)<t` for all :math:`t\in(0,1)` (Ling 1965;
    Nelsen 2006, Thms. 4.1.5 and 4.1.6).  Associativity is checked on a grid
    (:func:`associativity_defect` :math:`\le` ``tol``), the diagonal
    condition through :math:`\min_t (t-\delta_C(t))/(t(1-t)) >` ``diag_tol``
    (grid plus local minimisation, so that idempotents of ordinal sums are
    found).

    Parameters
    ----------
    C : copula
        Fully specified bivariate copula.
    tol : float
        Tolerance of the associativity defect.
    n : int
        Grid points per axis of the associativity check.
    diag_tol : float
        Tolerance of the diagonal condition.
    return_details : bool
        Also return a dict with the defect and the diagonal gap.

    Returns
    -------
    bool or (bool, dict)
    """
    defect, arg = associativity_defect(C, n, return_argmax=True)
    gap, t_gap = _diagonal_gap(C)
    res = bool(defect <= tol and gap > diag_tol)
    if return_details:
        return res, {
            "associativity_defect": defect,
            "associativity_argmax": arg,
            "diagonal_gap": gap,
            "diagonal_gap_at": t_gap,
        }
    return res


# ---------------------------------------------------------------------------
# constructions
# ---------------------------------------------------------------------------


def archimedean_from_generator(
    phi: Callable,
    *,
    dphi: Callable | None = None,
    d2phi: Callable | None = None,
    psi: Callable | None = None,
    phi0: float | None = None,
    check: bool = True,
) -> NumericArchimedeanCopula:
    r"""Archimedean copula from a numerical (vectorised) generator.

    Parameters
    ----------
    phi : callable
        Generator :math:`\varphi` on :math:`[0,1]`.
    dphi, d2phi : callable, optional
        Its first two derivatives (finite differences otherwise).
    psi : callable, optional
        Pseudo-inverse (computed by monotone inversion otherwise).
    phi0 : float, optional
        :math:`\varphi(0^+)`.
    check : bool
        Raise ``ValueError`` unless :math:`\varphi` is a valid generator
        (Nelsen 2006, Thm. 4.1.4).

    Returns
    -------
    NumericArchimedeanCopula
    """
    C = NumericArchimedeanCopula(phi=phi, psi=psi, dphi=dphi, d2phi=d2phi, phi0=phi0)
    if check:
        gen = C._archimedean_generator()
        checks, details = _basic_checks(gen.phi, gen.dphi, gen.phi0, 1e-9)
        if not all(checks.values()):
            raise ValueError(f"not an Archimedean generator: {checks} ({details})")
    return C


def from_laplace_transform(
    psi: Callable,
    *,
    dpsi: Callable | None = None,
    d2psi: Callable | None = None,
    check: bool = True,
    d_check: int = 6,
) -> NumericArchimedeanCopula:
    r"""Archimedean copula with inverse generator :math:`\psi` (a Laplace transform).

    If :math:`\psi` is the Laplace transform of a positive random variable
    (equivalently: completely monotone with :math:`\psi(0)=1`, Bernstein's
    theorem), :math:`C(u,v)=\psi(\psi^{-1}(u)+\psi^{-1}(v))` is a copula in
    every dimension (Kimberling 1974; Marshall & Olkin 1988).  The returned
    copula needs no frailty sampler: it is sampled by exact conditional
    inversion (see :class:`NumericArchimedeanCopula`); the generator
    :math:`\varphi=\psi^{-1}` is computed by monotone inversion.

    Parameters
    ----------
    psi : callable
        Vectorised :math:`\psi:[0,\infty)\to(0,1]`.
    dpsi, d2psi : callable, optional
        :math:`\psi'`, :math:`\psi''` (finite differences otherwise).
    check : bool
        Raise ``ValueError`` unless :math:`\psi(0)=1` and :math:`\psi` is
        2-monotone (a valid bivariate generator); warn if a difference of
        order :math:`\le` ``d_check`` has the wrong sign (then :math:`\psi`
        is not a Laplace transform).
    d_check : int
        Highest order of the complete-monotonicity spot check.

    Returns
    -------
    NumericArchimedeanCopula

    Examples
    --------
    >>> import numpy as np
    >>> from copul.theory.archimedean import from_laplace_transform
    >>> C = from_laplace_transform(lambda s: np.exp(-np.sqrt(s)))  # Gumbel(2)
    >>> round(C.kendalls_tau(), 8)                                 # 1 - 1/theta
    0.5
    """
    C = NumericArchimedeanCopula(psi=psi, dpsi=dpsi, d2psi=d2psi, name="Laplace transform")
    if check:
        p0 = float(_call(psi, np.array([0.0]))[0])
        if not abs(p0 - 1.0) <= 1e-10:
            raise ValueError(f"psi(0) must be 1, got {p0}")
        scale = float(np.nan_to_num(C._phi(np.array([0.5]))[0], nan=1.0)) or 1.0
        d_mono, info = _difference_d_monotone(psi, C.phi0, max(int(d_check), 2), scale)
        if not d_mono[2]:
            raise ValueError(f"psi is not 2-monotone (not a generator): {info}")
        if not all(d_mono.values()):
            warnings.warn(
                "psi fails a complete-monotonicity spot check (not a Laplace transform); "
                f"it is d-monotone at most for d = {max(d for d, ok in d_mono.items() if ok)}.",
                stacklevel=2,
            )
    return C


def archimedean_from_kendall_distribution(
    K: Callable,
    *,
    k: Callable | None = None,
    x_min: float = -30.0,
    x_max: float = 20.0,
    step: float = 0.05,
    check: bool = True,
) -> NumericArchimedeanCopula:
    r"""Archimedean copula with a given Kendall distribution.

    An Archimedean copula is determined by its Kendall distribution:
    with :math:`\lambda(t)=t-K(t)=\varphi(t)/\varphi'(t)<0`,

    .. math::

       \varphi(t) = \exp\Bigl(\int_{1/2}^{t}\frac{ds}{\lambda(s)}\Bigr),
       \qquad \varphi'=\frac{\varphi}{\lambda},\qquad
       \varphi''=\frac{\varphi\,K'}{\lambda^2}

    (Genest & Rivest 1993, Sec. 2; Nelsen 2006, Sec. 4.3);
    :math:`\varphi` is normalised by :math:`\varphi(1/2)=1` (generators are
    unique up to a positive factor).  Every distribution function :math:`K`
    on :math:`[0,1]` with :math:`K(t)>t` on :math:`(0,1)` arises in this way
    (Genest & Rivest 1993, Sec. 2) -- the generator is convex since
    :math:`K` is nondecreasing.

    Numerically, :math:`\log\varphi` is tabulated on a grid of
    :math:`x=\log(t/(1-t))` (where it is asymptotically linear at both ends)
    by Gauss--Legendre panels and evaluated anywhere by one more panel from
    the nearest node; beyond the grid it is extrapolated linearly in
    :math:`x`.

    Parameters
    ----------
    K : callable
        Vectorised Kendall distribution function on :math:`[0,1]`.
    k : callable, optional
        Its density :math:`K'` (finite differences otherwise).
    x_min, x_max, step : float
        Grid in :math:`x=\operatorname{logit}(t)`.
    check : bool
        Raise ``ValueError`` unless :math:`K(t)>t` on a grid of :math:`(0,1)`.

    Returns
    -------
    NumericArchimedeanCopula

    Examples
    --------
    >>> from copul.theory.archimedean import archimedean_from_kendall_distribution
    >>> C = archimedean_from_kendall_distribution(lambda t: t + t * (1 - t**2) / 2)
    >>> round(C.cdf(0.3, 0.7), 8) == round((0.3**-2 + 0.7**-2 - 1) ** -0.5, 8)  # Clayton(2)
    True
    """
    xg, wg = np.polynomial.legendre.leggauss(8)
    xg, wg = 0.5 * (xg + 1.0), 0.5 * wg

    def lam(t):
        t = np.asarray(t, float)
        return t - _call(K, t)

    if check:
        tt = np.linspace(0.001, 0.999, 999)
        if not np.all(lam(tt) < 0):
            raise ValueError("K(t) > t must hold on (0, 1) for an Archimedean copula.")

    def expit(x):
        return 1.0 / (1.0 + np.exp(-x))

    def g(x):  # d log(phi) / dx
        t = expit(x)
        with np.errstate(all="ignore"):
            out = t * (1.0 - t) / lam(t)
        return np.where(np.isfinite(out), out, 0.0)

    nodes = np.arange(x_min, x_max + 0.5 * step, step)
    i0 = int(np.argmin(np.abs(nodes)))
    nodes = nodes - nodes[i0]  # node at x = 0 (t = 1/2)
    a, b = nodes[:-1], nodes[1:]
    panel = (b - a) * np.sum(wg[None, :] * g(a[:, None] + (b - a)[:, None] * xg[None, :]), axis=1)
    L = np.concatenate([[0.0], np.cumsum(panel)])
    L = L - L[i0]
    g_lo, g_hi = float(g(np.array([nodes[0]]))[0]), float(g(np.array([nodes[-1]]))[0])
    # strict iff log(phi) keeps growing linearly as x -> -inf (g does not decay)
    g_ref = float(g(np.array([nodes[0] + 10.0]))[0])
    strict = abs(g_lo) > 1e-12 and abs(g_lo) >= 0.5 * abs(g_ref)
    phi0 = math.inf if strict else float(np.exp(L[0]))

    def log_phi_x(x):
        x = np.asarray(x, float)
        flat = x.ravel()
        j = np.clip(np.rint((flat - nodes[0]) / step).astype(int), 0, nodes.size - 1)
        xj = nodes[j]
        inside = (flat >= nodes[0]) & (flat <= nodes[-1])
        d = np.where(inside, flat - xj, 0.0)
        quad = d * np.sum(wg[None, :] * g(xj[:, None] + d[:, None] * xg[None, :]), axis=1)
        out = L[j] + quad
        out = np.where(flat > nodes[-1], L[-1] + g_hi * (flat - nodes[-1]), out)
        lo_ext = L[0] + (g_lo * (flat - nodes[0]) if strict else 0.0)
        out = np.where(flat < nodes[0], lo_ext, out)
        return out.reshape(x.shape)

    def phi(t):
        t = np.asarray(t, float)
        with np.errstate(all="ignore"):
            x = np.log(t) - np.log1p(-t)
            out = np.exp(log_phi_x(np.clip(x, -1e300, 1e300)))
        out = np.where(t >= 1.0, 0.0, np.where(t <= 0.0, phi0, out))
        return out

    def dphi(t):
        t = np.asarray(t, float)
        with np.errstate(all="ignore"):
            return phi(t) / lam(t)

    if k is None:

        def kd(t):
            return np.maximum(fd_derivative(K, t, 1, 0.0, 1.0), 0.0)
    else:
        kd = k

    def d2phi(t):
        t = np.asarray(t, float)
        with np.errstate(all="ignore"):
            return phi(t) * _call(kd, t) / lam(t) ** 2

    return NumericArchimedeanCopula(
        phi=phi, dphi=dphi, d2phi=d2phi, phi0=phi0, name="from Kendall distribution"
    )
