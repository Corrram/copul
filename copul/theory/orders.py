r"""
Dependence orderings of bivariate copulas.

Concordance order
-----------------
:math:`C_1` is smaller than :math:`C_2` in the *concordance* (PQD) order,
:math:`C_1\preceq_c C_2`, if :math:`C_1(u,v)\le C_2(u,v)` for all
:math:`(u,v)` (Nelsen 2006, Def. 2.8.1; Joe 1997, Sect. 2.2; Müller &
Stoyan 2002, Ch. 3 -- in two dimensions equivalent to the order of the
survival functions).  :math:`W\preceq_c C\preceq_c M` for every copula
(Fréchet--Hoeffding bounds), :math:`\Pi\preceq_c C` iff :math:`C` is PQD.
Every measure of concordance in the sense of Scarsini (1984) is monotone in
this order; in particular Spearman's :math:`\rho`, Kendall's :math:`\tau`
(Nelsen 2006, Sect. 5.1), and -- directly from their formulas, which are
integrals of :math:`C` with nonnegative weights -- Blomqvist's
:math:`\beta`, Gini's :math:`\gamma`, Spearman's footrule and Blest's
:math:`\nu`.  A parametric family is *positively ordered* if
:math:`\theta_1\le\theta_2` implies :math:`C_{\theta_1}\preceq_c
C_{\theta_2}` (e.g. Clayton, Gumbel--Hougaard, Frank, AMH, Joe, Gaussian,
FGM, Plackett; Nelsen 2006, Sect. 2.8 and 4.4; Joe 1997, Ch. 5).

More stochastically increasing order
------------------------------------
:math:`C_2` is *more SI* (more regression dependent) than :math:`C_1`,
:math:`C_1\preceq_{SI} C_2`, if for every :math:`v` the map

.. math::

   u\mapsto \bigl(\partial_1 C_2(u,\cdot)\bigr)^{-1}\bigl(\partial_1 C_1(u,v)\bigr)

is nondecreasing (Yanagimoto & Okamoto 1969; Schriever 1987; Joe 1997,
Sect. 2.2), i.e. the quantile transform of the conditional distributions of
:math:`C_1` into those of :math:`C_2` is increasing in the conditioning
variable.  :math:`\Pi\preceq_{SI} C` iff :math:`C` is SI(V|U), and
:math:`C\preceq_{SI} M` for every :math:`C`.

References
----------
* Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman &
  Hall.
* Müller, A. & Stoyan, D. (2002). *Comparison Methods for Stochastic Models
  and Risks*. Wiley.
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer.
* Scarsini, M. (1984). On measures of concordance. *Stochastica* 8,
  201--218.
* Schriever, B. F. (1987). An ordering for positive dependence. *Ann.
  Statist.* 15, 1208--1214.
* Slepian, D. (1962). The one-sided barrier problem for Gaussian noise.
  *Bell System Tech. J.* 41, 463--501.
* Yanagimoto, T. & Okamoto, M. (1969). Partial orderings of permutations
  and monotonicity of a rank correlation statistic. *Ann. Inst. Statist.
  Math.* 21, 489--506.

Examples
--------
>>> import copul as cp
>>> from copul.theory.orders import concordance_order, is_concordance_ordered
>>> bool(concordance_order(cp.Clayton(1), cp.Clayton(3)))
True
>>> is_concordance_ordered(cp.Frank, [-4, -1, 2, 6]).direction
'increasing'
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from copul.theory.dependence import PropertyResult, _nodes, _zoom_points

__all__ = [
    "CONCORDANCE_MEASURES",
    "FamilyOrderResult",
    "concordance_order",
    "is_concordance_ordered",
    "is_more_si",
    "measures_along",
]

#: measures that are monotone in the concordance order
CONCORDANCE_MEASURES = ("rho", "tau", "beta", "gamma", "footrule", "nu")

_NAME = "concordance_order"


def _backend(C):
    from copul.measures.backend import numeric_backend

    return numeric_backend(C)


def _exact_order(C1, C2):
    """Exact answer for C1 <=_c C2 from published facts, or ``None``."""
    from copul.family.elliptical.gaussian import Gaussian
    from copul.family.frechet.lower_frechet import LowerFrechet
    from copul.family.frechet.upper_frechet import UpperFrechet
    from copul.family.other.farlie_gumbel_morgenstern import FarlieGumbelMorgenstern
    from copul.theory.dependence import _float_attr, exact_facts
    from copul.theory.distances import _pi_checkerboard_matrix

    if isinstance(C1, LowerFrechet) or isinstance(C2, UpperFrechet):
        return True, "Fréchet--Hoeffding bounds W <= C <= M (Nelsen 2006, Thm. 2.2.3)"
    if type(C1) is Gaussian and type(C2) is Gaussian:
        r1, r2 = _float_attr(C1, "rho"), _float_attr(C2, "rho")
        if r1 is not None and r2 is not None:
            return r1 <= r2, "the bivariate normal cdf increases in rho (Slepian 1962)"
    if type(C1) is FarlieGumbelMorgenstern and type(C2) is FarlieGumbelMorgenstern:
        t1, t2 = _float_attr(C1, "theta"), _float_attr(C2, "theta")
        if t1 is not None and t2 is not None:
            return t1 <= t2, "FGM: C - Pi = theta uv(1-u)(1-v)"
    P1, P2 = _pi_checkerboard_matrix(C1), _pi_checkerboard_matrix(C2)
    if P1 is not None and P2 is not None:
        if P1.shape == (1, 1):  # C1 = Pi: C2 PQD
            return bool(exact_facts(C2)["PQD"]), "Pi <= C iff C is PQD (exact checkerboard check)"
        if P2.shape == (1, 1):
            return bool(exact_facts(C1)["NQD"]), "C <= Pi iff C is NQD (exact checkerboard check)"
        return None  # handled by the exact node check
    if P1 is not None and P1.shape == (1, 1):
        f = exact_facts(C2).get("PQD")
        if f is not None and f.method == "exact":
            return f.holds, "Pi <= C iff C is PQD: " + f.reason
    if P2 is not None and P2.shape == (1, 1):
        f = exact_facts(C1).get("NQD")
        if f is not None and f.method == "exact":
            return f.holds, "C <= Pi iff C is NQD: " + f.reason
    return None


def concordance_order(
    C1,
    C2,
    *,
    method: str = "auto",
    n_grid: int | None = None,
    tol: float | None = None,
    refine: bool = True,
) -> PropertyResult:
    r"""Check :math:`C_1\preceq_c C_2`, i.e. :math:`C_1(u,v)\le C_2(u,v)` everywhere.

    Parameters
    ----------
    C1, C2 : bivariate copulas
        Fully specified copulas.
    method : {"auto", "grid"}
        ``"auto"`` first applies exact rules (Fréchet--Hoeffding bounds,
        Gaussian and FGM families, comparisons with :math:`\Pi` through exact
        PQD/NQD characterizations, exact node comparison of two
        independence-kernel checkerboards -- their difference is bilinear on
        the common refinement of the grids); otherwise, and with
        ``"grid"``, :math:`C_2-C_1` is minimized over a Chebyshev grid
        (including the grid lines of checkerboards) with local zooming.
    n_grid : int, optional
        Grid size per axis (default 65).
    tol : float, optional
        Tolerance (default :math:`10^{-10}`).

    Returns
    -------
    PropertyResult
        ``worst_violation`` is :math:`\max(C_1-C_2)^+` found.
    """
    from copul.theory.distances import _breaks, _pi_checkerboard_matrix

    t = 1e-10 if tol is None else float(tol)
    if method == "auto":
        ex = _exact_order(C1, C2)
        if ex is not None:
            return PropertyResult(_NAME, bool(ex[0]), "exact", 0.0, None, ex[1])
        P1, P2 = _pi_checkerboard_matrix(C1), _pi_checkerboard_matrix(C2)
        if P1 is not None and P2 is not None:
            from copul.checkerboard import _biv_engine as eng

            rows = np.unique(
                np.concatenate([np.arange(k + 1) / k for k in (P1.shape[0], P2.shape[0])])
            )
            cols = np.unique(
                np.concatenate([np.arange(k + 1) / k for k in (P1.shape[1], P2.shape[1])])
            )
            U, V = np.meshgrid(rows, cols, indexing="ij")
            d = eng.cdf(P1, None, U, V) - eng.cdf(P2, None, U, V)
            k = int(np.argmax(d))
            worst = float(max(d.flat[k], 0.0))
            return PropertyResult(
                _NAME,
                worst <= 1e-12,
                "exact",
                worst,
                {"u": float(U.flat[k]), "v": float(V.flat[k])},
                "independence-kernel checkerboards: C2 - C1 is bilinear on the common "
                "refinement of the grids, so its minimum is attained at a node",
            )
    elif method != "grid":
        raise ValueError("method must be 'auto' or 'grid'")
    b1, b2 = _backend(C1), _backend(C2)
    n = n_grid or 65
    x = _nodes(n)
    ub, vb = _breaks(b1, b2)
    gu = np.unique(np.concatenate([x] + ([ub] if ub is not None else [])))
    gv = np.unique(np.concatenate([x] + ([vb] if vb is not None else [])))
    U, V = np.meshgrid(gu, gv, indexing="ij")

    def diff(u, v):
        with np.errstate(all="ignore"):
            return np.asarray(b1.cdf(u, v), float) - np.asarray(b2.cdf(u, v), float)

    D = diff(U, V)
    order = np.argsort(D, axis=None)[::-1][:4]
    k0 = order[0]
    best = float(D.flat[k0])
    where = {"u": float(U.flat[k0]), "v": float(V.flat[k0])}
    if refine:
        lo = 0.5 * float(x[0])
        for idx in order:
            cu, cv = float(U.flat[idx]), float(V.flat[idx])
            r = 2.0 / n
            for _ in range(6):
                pu, pv = np.meshgrid(
                    _zoom_points(cu, r, lo=lo), _zoom_points(cv, r, lo=lo), indexing="ij"
                )
                dz = diff(pu, pv)
                j = int(np.nanargmax(dz))
                if dz.flat[j] > best:
                    best = float(dz.flat[j])
                    where = {"u": float(pu.flat[j]), "v": float(pv.flat[j])}
                cu, cv = float(pu.flat[j]), float(pv.flat[j])
                r *= 0.4
    return PropertyResult(
        _NAME,
        best <= t,
        "grid",
        max(best, 0.0),
        where,
        "grid check of C1(u,v) <= C2(u,v)",
        {"tol": t, "n_grid": n},
    )


@dataclass
class FamilyOrderResult:
    """Concordance ordering of a family along a sequence of parameter values.

    Attributes
    ----------
    values : list
        The (sorted) parameter values.
    increasing, decreasing : bool
        Whether consecutive members are ordered increasingly / decreasingly.
    pairs : list of tuple
        ``(up, down)`` :class:`PropertyResult` of ``C_k <= C_{k+1}`` and
        ``C_{k+1} <= C_k`` for consecutive members.
    """

    values: list
    increasing: bool
    decreasing: bool
    pairs: list = field(default_factory=list)

    @property
    def holds(self) -> bool:
        """Whether the family is ordered (in some direction) along ``values``."""
        return self.increasing or self.decreasing

    @property
    def direction(self) -> str | None:
        """``"increasing"``, ``"decreasing"``, ``"constant"`` or ``None``."""
        if self.increasing and self.decreasing:
            return "constant"
        if self.increasing:
            return "increasing"
        if self.decreasing:
            return "decreasing"
        return None

    def __bool__(self) -> bool:
        return self.holds


def _instantiate(family, value, param=None):
    from copul.measures.backend import free_parameters

    if isinstance(family, type):
        return family(**{param: value}) if param else family(value)
    free = free_parameters(family) if hasattr(family, "params") else []
    if free:
        return family(**{param or free[0]: value})
    if callable(family):
        return family(value)
    raise TypeError("family must be a copula class, a copula with a free parameter or a callable")


def is_concordance_ordered(
    family,
    values: Sequence[float],
    *,
    param: str | None = None,
    n_grid: int | None = None,
    tol: float | None = None,
) -> FamilyOrderResult:
    r"""Check whether a family is concordance ordered along ``values``.

    Parameters
    ----------
    family : copula class, copula with a free parameter, or callable
        ``family(value)`` (or ``family(**{param: value})``) must return a
        fully specified copula.
    values : sequence of float
        Parameter values (sorted internally).
    param : str, optional
        Parameter name (for families with several parameters, the others
        fixed by ``family``).
    n_grid, tol
        See :func:`concordance_order`.

    Returns
    -------
    FamilyOrderResult
    """
    vals = sorted(float(v) for v in values)
    cops = [_instantiate(family, v, param) for v in vals]
    pairs = []
    for a, b in itertools.pairwise(cops):
        up = concordance_order(a, b, n_grid=n_grid, tol=tol)
        down = concordance_order(b, a, n_grid=n_grid, tol=tol)
        pairs.append((up, down))
    inc = all(p[0].holds for p in pairs)
    dec = all(p[1].holds for p in pairs)
    return FamilyOrderResult(vals, inc, dec, pairs)


def measures_along(
    family,
    values: Sequence[float],
    keys: Iterable[str] = CONCORDANCE_MEASURES,
    *,
    param: str | None = None,
    method: str = "auto",
) -> dict[str, np.ndarray]:
    """Evaluate dependence measures along parameter values of a family.

    Returns ``{key: array}`` (values sorted like ``sorted(values)``).  For a
    concordance-ordered family every concordance measure
    (:data:`CONCORDANCE_MEASURES`) is monotone along the values.
    """
    from copul.measures.engine import compute

    vals = sorted(float(v) for v in values)
    keys = list(keys)
    out = {k: np.empty(len(vals)) for k in keys}
    for j, v in enumerate(vals):
        res = compute(_instantiate(family, v, param), keys, method=method)
        for k in keys:
            out[k][j] = float(res[k])
    return out


def is_more_si(
    C1, C2, *, n_grid: int | None = None, tol: float | None = None, eps: float = 1e-12
) -> PropertyResult:
    r"""Check :math:`C_1\preceq_{SI}C_2` (:math:`C_2` more stochastically increasing).

    Evaluates :math:`\psi(u,v) = Q_2\bigl(u, \partial_1C_1(u,v)\bigr)` with the
    conditional quantile function :math:`Q_2(u,\cdot)` of :math:`C_2`
    (``cond_distr_1_inv``) on a Chebyshev grid and checks that it is
    nondecreasing in :math:`u` for every :math:`v` (all pairs of grid
    points).  Conditional probabilities are clipped to
    :math:`[\varepsilon, 1-\varepsilon]` so that atoms of the conditional
    distributions (singular copulas) are handled consistently.

    Returns
    -------
    PropertyResult
        ``worst_violation`` is the largest decrease of :math:`\psi` found.
    """
    t = 1e-8 if tol is None else float(tol)
    n = n_grid or 49
    x = _nodes(n)
    U, V = np.meshgrid(x, x, indexing="ij")
    b1 = _backend(C1)
    from copul.theory.markov import conditional_quantile

    q2 = conditional_quantile(C2)
    with np.errstate(all="ignore"):
        w = np.clip(np.asarray(b1.get("h1")(U, V), float), eps, 1 - eps)
        psi = np.asarray(q2(U.ravel(), w.ravel()), float).reshape(U.shape)
    # largest decrease along u (axis 0) over all pairs a < b
    run_max = np.maximum.accumulate(psi, axis=0)
    drop = run_max - psi
    k = int(np.argmax(drop))
    worst = float(drop.flat[k])
    i, j = np.unravel_index(k, drop.shape)
    return PropertyResult(
        "more_si",
        worst <= t,
        "grid",
        worst,
        {"u": float(x[i]), "v": float(x[j])},
        "grid check that u -> Q_2(u, d1 C_1(u, v)) is nondecreasing "
        "(Yanagimoto & Okamoto 1969; Joe 1997, Sect. 2.2)",
        {"tol": t, "n_grid": n},
    )
