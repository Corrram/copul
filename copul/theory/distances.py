r"""
Distances between bivariate copulas and Trutschnig's dependence measure.

Metrics (:func:`copula_distance`)
---------------------------------
``"sup"``
    uniform metric :math:`d_\infty(A,B)=\sup_{(u,v)}|A(u,v)-B(u,v)|`
    (grid search with local zooming; exact at the grid nodes for
    independence-kernel checkerboards);
``"L1"``, ``"L2"``
    Lebesgue norms :math:`\int\!\!\int|A-B|` and
    :math:`\bigl(\int\!\!\int(A-B)^2\bigr)^{1/2}`;
``"D1"``, ``"D2"``, ``"Dinf"``
    the :math:`\partial`-metrics of Trutschnig (2011), defined through the
    Markov kernels :math:`K_A(x,[0,y])=\partial_1A(x,y)`:

    .. math::

       \Phi_{A,B}(y) &= \int_0^1 |\partial_1 A(x,y)-\partial_1 B(x,y)|\,dx,\\
       D_1(A,B) &= \int_0^1 \Phi_{A,B}(y)\,dy,\qquad
       D_\infty(A,B) = \sup_{y\in[0,1]}\Phi_{A,B}(y),\\
       D_2(A,B) &= \Bigl(\int_0^1\!\!\int_0^1
                  \bigl(\partial_1 A(x,y)-\partial_1 B(x,y)\bigr)^2
                  \,dx\,dy\Bigr)^{1/2}.

All six are metrics on the space of bivariate copulas; :math:`D_1`,
:math:`D_2`, :math:`D_\infty` generate the same (strong) topology, which is
finer than the topology of uniform convergence.  Elementary inequalities
(Trutschnig 2011, Sect. 3): since
:math:`|A(x,y)-B(x,y)|=|\int_0^x(\partial_1A-\partial_1B)(s,y)\,ds|\le\Phi_{A,B}(y)`,

.. math::

   d_\infty \le D_\infty,\qquad \textstyle\int\!\!\int|A-B| \le D_1
   \le D_\infty,\qquad D_2^2 \le D_1 \le D_2

(the last two because :math:`|\partial_1A-\partial_1B|\le1` and by
Cauchy--Schwarz).

Dependence measure
------------------
:math:`\zeta_1(C) = 3D_1(C,\Pi)\in[0,1]` (Trutschnig 2011):
:math:`\zeta_1(C)=0` iff :math:`C=\Pi`, :math:`\zeta_1(C)=1` iff :math:`C`
is completely dependent (:math:`\partial_1C\in\{0,1\}` a.e., i.e.
:math:`V=f(U)` a.s.).  Registered in :mod:`copul.measures` under the key
``"zeta1"`` and available as the copula method ``trutschnig_zeta``.

References
----------
* Trutschnig, W. (2011). On a strong metric on the space of copulas and its
  induced dependence measure. *J. Math. Anal. Appl.* 384, 690--705.
* Durante, F. & Sempi, C. (2016). *Principles of Copula Theory*. CRC Press
  (uniform metric, Markov kernels).

Examples
--------
>>> import copul as cp
>>> from copul.theory.distances import copula_distance, trutschnig_zeta
>>> round(copula_distance(cp.UpperFrechet(), cp.BivIndependenceCopula(), "sup"), 6)
0.25
>>> round(copula_distance(cp.UpperFrechet(), cp.BivIndependenceCopula(), "D1"), 6)
0.333333
>>> round(trutschnig_zeta(cp.FarlieGumbelMorgenstern(0.8)), 10)  # |theta| / 4
0.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "METRICS",
    "DistanceResult",
    "copula_distance",
    "trutschnig_zeta",
    "zeta1",
]

#: canonical metric names
METRICS = ("sup", "L1", "L2", "D1", "D2", "Dinf")

_ALIASES = {
    "sup": "sup",
    "uniform": "sup",
    "linf": "sup",
    "l_inf": "sup",
    "l1": "L1",
    "l2": "L2",
    "d1": "D1",
    "d_1": "D1",
    "d2": "D2",
    "d_2": "D2",
    "dinf": "Dinf",
    "d_inf": "Dinf",
    "d_infty": "Dinf",
    "dsup": "Dinf",
}


def _resolve_metric(metric: str) -> str:
    key = str(metric).strip()
    if key in METRICS:
        return key
    k = _ALIASES.get(key.lower().replace(" ", ""))
    if k is None:
        raise ValueError(f"Unknown metric {metric!r}; choose one of {METRICS}.")
    return k


@dataclass
class DistanceResult:
    """Value of a copula distance with diagnostics.

    Attributes
    ----------
    metric : str
        Canonical metric name.
    value : float
        The distance.
    error : float
        Error estimate (``0.0`` for exact evaluations).
    method : str
        ``"exact"`` or ``"numeric"``.
    where : dict or None
        Maximizer for ``"sup"`` (``{"u", "v"}``) and ``"Dinf"`` (``{"v"}``).
    info : dict
        Further diagnostics.
    """

    metric: str
    value: float
    error: float
    method: str
    where: dict | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.value)


def _backend(C):
    from copul.measures.backend import numeric_backend

    return numeric_backend(C)


def _breaks(*bes):
    """Union of the known discontinuity lines of several backends (or ``None``)."""
    ub, vb = [], []
    for be in bes:
        br = getattr(be, "breaks", None)
        if br is None:
            continue
        if br[0] is not None:
            ub.append(np.asarray(br[0], float))
        if br[1] is not None:
            vb.append(np.asarray(br[1], float))
    u = np.unique(np.concatenate(ub)) if ub else None
    v = np.unique(np.concatenate(vb)) if vb else None
    return u, v


def _pi_checkerboard_matrix(C):
    """Mass matrix if ``C`` equals an independence-kernel checkerboard (or Pi)."""
    from copul.checkerboard import _biv_engine as eng
    from copul.checkerboard._biv_mixin import BivCheckerboardMixin
    from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
    from copul.family.other.independence_copula import IndependenceCopula

    if isinstance(C, (BivIndependenceCopula, IndependenceCopula)):
        return np.ones((1, 1))
    if isinstance(C, BivCheckerboardMixin):
        P = np.asarray(C.matr, dtype=float)
        P = P / P.sum()
        S = eng._signs(P, C._kernel_signs())
        if S is None or not np.any((S != 0) & (P > 0)):
            return P
    return None


# ---------------------------------------------------------------------------
# exact evaluation for independence-kernel checkerboards
# ---------------------------------------------------------------------------


def _pi_cdf(P, u, v):
    from copul.checkerboard import _biv_engine as eng

    return eng.cdf(P, None, u, v)


def _pi_h1(P, u, v):
    from copul.checkerboard import _biv_engine as eng

    return eng.cond_distr(P, None, 1, u, v)


def _exact_checkerboard(PA, PB, metric):
    r"""Exact distances of two independence-kernel checkerboards.

    On the common refinement of both grids :math:`A-B` is bilinear, so
    :math:`|A-B|` is maximal at a node; :math:`\partial_1A-\partial_1B` is
    constant in :math:`u` on every refined row and linear in :math:`v`
    between refined column lines, so :math:`D_1`, :math:`D_2` are integrals
    of piecewise linear functions and :math:`\Phi_{A,B}` is convex between
    column lines (maximal at a line).
    """
    rows = np.unique(np.concatenate([np.arange(k + 1) / k for k in (PA.shape[0], PB.shape[0])]))
    cols = np.unique(np.concatenate([np.arange(k + 1) / k for k in (PA.shape[1], PB.shape[1])]))
    if metric == "sup":
        U, V = np.meshgrid(rows, cols, indexing="ij")
        D = np.abs(_pi_cdf(PA, U, V) - _pi_cdf(PB, U, V))
        k = int(np.argmax(D))
        return float(D.flat[k]), {"u": float(U.flat[k]), "v": float(V.flat[k])}
    mid = 0.5 * (rows[:-1] + rows[1:])
    w = np.diff(rows)
    U, V = np.meshgrid(mid, cols, indexing="ij")
    d = _pi_h1(PA, U, V) - _pi_h1(PB, U, V)  # (rows, col edges)
    a, b = d[:, :-1], d[:, 1:]
    lens = np.diff(cols)[None, :]
    if metric == "D1":
        same = a * b >= 0
        den = np.where(same, 1.0, np.abs(a) + np.abs(b))
        den = np.where(den > 0, den, 1.0)
        piece = np.where(same, 0.5 * (np.abs(a) + np.abs(b)), 0.5 * (a**2 + b**2) / den)
        return float(np.sum(w[:, None] * lens * piece)), None
    if metric == "D2":
        piece = (a**2 + a * b + b**2) / 3.0
        return float(np.sqrt(np.sum(w[:, None] * lens * piece))), None
    # Dinf
    phi = np.sum(w[:, None] * np.abs(d), axis=0)
    k = int(np.argmax(phi))
    return float(phi[k]), {"v": float(cols[k])}


# ---------------------------------------------------------------------------
# numerical evaluation
# ---------------------------------------------------------------------------


def _sup_numeric(fa, fb, extra_u=None, extra_v=None, n_grid=129, n_starts=8, n_zoom=14):
    x = (np.arange(n_grid) + 0.5) / n_grid
    gu = np.unique(np.concatenate([x] + ([extra_u] if extra_u is not None else [])))
    gv = np.unique(np.concatenate([x] + ([extra_v] if extra_v is not None else [])))
    U, V = np.meshgrid(gu, gv, indexing="ij")

    def g(u, v):
        with np.errstate(all="ignore"):
            return np.abs(np.asarray(fa(u, v), float) - np.asarray(fb(u, v), float))

    vals = g(U.ravel(), V.ravel())
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    order = np.argsort(-vals)[:n_starts]
    best = float(vals[order[0]])
    where = {"u": float(U.ravel()[order[0]]), "v": float(V.ravel()[order[0]])}
    r0 = 1.0 / n_grid
    for idx in order:
        cu, cv = float(U.ravel()[idx]), float(V.ravel()[idx])
        r, cur = r0, float(vals[idx])
        for _ in range(n_zoom):
            s = np.linspace(-r, r, 11)
            PU, PV = np.meshgrid(np.clip(cu + s, 0, 1), np.clip(cv + s, 0, 1), indexing="ij")
            gv_ = g(PU.ravel(), PV.ravel())
            gv_ = np.where(np.isfinite(gv_), gv_, -np.inf)
            j = int(np.argmax(gv_))
            if gv_[j] >= cur:
                cur = float(gv_[j])
                cu, cv = float(PU.ravel()[j]), float(PV.ravel()[j])
            r /= 4.0
        if cur > best:
            best, where = cur, {"u": cu, "v": cv}
    # |A - B| is 4-Lipschitz (each copula is 2-Lipschitz); remaining radius
    return best, 4 * 2 * r0 * 4.0**-n_zoom, where


def _phi_numeric(ha, hb, ys, rtol, atol, x_breaks=None):
    from copul.measures.quadrature import integrate_1d_batch

    ys = np.asarray(ys, float)

    def f(x, rows):
        y = ys[rows]
        with np.errstate(all="ignore"):
            return np.abs(np.asarray(ha(x, y), float) - np.asarray(hb(x, y), float))

    brk = None if x_breaks is None or len(x_breaks) == 0 else np.asarray(x_breaks, float)
    val, err = integrate_1d_batch(
        f,
        np.zeros(ys.size),
        np.ones(ys.size),
        atol=atol,
        rtol=rtol,
        jump_search=True,
        breaks=brk,
    )
    return np.asarray(val, float), np.asarray(err, float)


def _dinf_numeric(ha, hb, rtol, atol, x_breaks=None, y_extra=None, n_grid=129):
    ys = (np.arange(n_grid) + 0.5) / n_grid
    if y_extra is not None:
        ys = np.unique(np.concatenate([ys, np.clip(y_extra, 1e-12, 1 - 1e-12)]))
    phi, err = _phi_numeric(ha, hb, ys, rtol, atol, x_breaks)
    k = int(np.argmax(phi))
    best, best_err, where = float(phi[k]), float(err[k]), float(ys[k])
    r = 1.0 / n_grid
    for _ in range(8):  # golden-ish zoom around the maximizer
        cand = np.clip(where + np.linspace(-r, r, 9), 1e-12, 1 - 1e-12)
        ph, er = _phi_numeric(ha, hb, cand, rtol, atol, x_breaks)
        j = int(np.argmax(ph))
        if ph[j] >= best:
            best, best_err, where = float(ph[j]), float(er[j]), float(cand[j])
        r /= 4.0
    # Phi is 2-Lipschitz (Trutschnig 2011): |Phi(y) - Phi(y')| <= 2|y - y'|
    return best, best_err + 2 * r, {"v": where}


def copula_distance(
    A,
    B,
    metric: str = "sup",
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    n_grid: int = 129,
    full_output: bool = False,
):
    r"""Distance between two bivariate copulas.

    Parameters
    ----------
    A, B : bivariate copulas
        Fully specified copula objects of :mod:`copul`.
    metric : {"sup", "L1", "L2", "D1", "D2", "Dinf"}
        See the module docstring (aliases: ``"uniform"``/``"linf"`` for
        ``"sup"``, ``"D_inf"`` for ``"Dinf"``).
    rtol, atol : float
        Tolerances of the adaptive quadrature (integral metrics).
    n_grid : int
        Initial grid size of the maximizations (``"sup"``, ``"Dinf"``).
    full_output : bool
        Return a :class:`DistanceResult` instead of a float.

    Returns
    -------
    float or DistanceResult

    Notes
    -----
    Exact for two independence-kernel checkerboards (including :math:`\Pi`)
    in every metric except ``"L1"``/``"L2"`` (which use quadrature with the
    grid lines as breakpoints).  The :math:`\partial`-metrics are not
    symmetric under transposition: :math:`D_1(A^\top,B^\top)` compares the
    conditional distributions of :math:`U` given :math:`V`.
    """
    metric = _resolve_metric(metric)
    PA, PB = _pi_checkerboard_matrix(A), _pi_checkerboard_matrix(B)
    if PA is not None and PB is not None and metric not in ("L1", "L2"):
        val, where = _exact_checkerboard(PA, PB, metric)
        res = DistanceResult(metric, val, 0.0, "exact", where)
        return res if full_output else res.value
    bea, beb = _backend(A), _backend(B)
    ub, vb = _breaks(bea, beb)
    if metric == "sup":
        val, err, where = _sup_numeric(bea.cdf, beb.cdf, ub, vb, n_grid=n_grid)
        res = DistanceResult(metric, val, err, "numeric", where)
    elif metric in ("L1", "L2"):
        from copul.measures.quadrature import integrate_2d

        p = 1 if metric == "L1" else 2

        def g(u, v):
            with np.errstate(all="ignore"):
                return (
                    np.abs(np.asarray(bea.cdf(u, v), float) - np.asarray(beb.cdf(u, v), float)) ** p
                )

        val, err = integrate_2d(g, rtol=rtol, atol=atol, u_breaks=ub, v_breaks=vb)
        if p == 2:
            val, err = float(np.sqrt(max(val, 0.0))), float(err / (2 * np.sqrt(max(val, 1e-300))))
        res = DistanceResult(metric, float(val), float(err), "numeric")
    elif metric in ("D1", "D2"):
        from copul.measures.quadrature import integrate_2d

        ha, hb = bea.get("h1"), beb.get("h1")
        p = 1 if metric == "D1" else 2

        def g(u, v):
            with np.errstate(all="ignore"):
                return np.abs(np.asarray(ha(u, v), float) - np.asarray(hb(u, v), float)) ** p

        val, err = integrate_2d(g, rtol=rtol, atol=atol, u_breaks=ub, v_breaks=vb)
        if p == 2:
            val, err = float(np.sqrt(max(val, 0.0))), float(err / (2 * np.sqrt(max(val, 1e-300))))
        res = DistanceResult(metric, float(val), float(err), "numeric")
    else:
        val, err, where = _dinf_numeric(
            bea.get("h1"), beb.get("h1"), rtol, atol, x_breaks=ub, y_extra=vb, n_grid=n_grid
        )
        res = DistanceResult(metric, val, err, "numeric", where)
    return res if full_output else res.value


def trutschnig_zeta(C, **kwargs) -> float:
    r"""Trutschnig's dependence measure :math:`\zeta_1(C) = 3D_1(C,\Pi)`.

    Shortcut for ``copul.measures.compute(C, "zeta1", **kwargs)``; exact for
    checkerboards, shuffles of :math:`M` and the Fréchet bounds.  See the
    module docstring.
    """
    from copul.measures.engine import compute

    return compute(C, "zeta1", **kwargs)


#: alias of :func:`trutschnig_zeta`
zeta1 = trutschnig_zeta
