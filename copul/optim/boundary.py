r"""
Tracing boundaries of attainable regions over checkerboard copulas.

Two strategies are provided by :func:`trace_boundary`:

``method="mu"`` -- supporting-hyperplane (scalarisation) sweep
    For the upper boundary solve :math:`\max_P\; y(P)-\mu\,x(P)` for a range of
    :math:`\mu`, for the lower one :math:`\min_P\; y(P)+\mu\,x(P)`.  With
    :math:`x=\xi` (convex) and :math:`y` affine these are concave/convex
    programs for :math:`\mu\ge0`; every solution is a point of the boundary of
    the convex hull of the attainable set of :math:`(x,y)` on the grid.  The
    problem is compiled once with a ``cvxpy`` parameter for :math:`\mu`.
``method="target"`` -- constraint sweep
    For :math:`t` on a grid solve :math:`\max_P\{y(P): x(P)\le t\}` (or
    :math:`x(P)=t` if :math:`x` is affine), which gives the boundary value at
    :math:`x=t` directly whenever the boundary is non-decreasing in :math:`x`.

Since checkerboards of a fixed size form a subset of all copulas, traced upper
boundaries lie *below* the exact ones and converge as :math:`n\to\infty`.

Examples
--------
>>> from copul.optim import trace_boundary  # doctest: +SKIP
>>> tr = trace_boundary("xi", "rho", side="upper", n=24)  # doctest: +SKIP
>>> tr.points[:3]  # doctest: +SKIP
>>> tr.plot()  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from copul.optim._backend import require_cvxpy
from copul.optim._backend import solve as _solve
from copul.optim.checkerboard_formulas import QuadraticForm
from copul.optim.problem import (
    CheckerboardProblem,
    NonConvexError,
    OptimResult,
    _marginal_residual,
    balance,
    to_cvxpy,
)
from copul.regions.measures import resolve

__all__ = ["BoundaryTrace", "trace_boundary"]


@dataclass
class BoundaryTrace:
    """Points on a numerically traced boundary.

    Attributes
    ----------
    x, y : str
        Measure keys of the axes.
    side : str
        ``"upper"`` or ``"lower"``.
    points : numpy.ndarray
        ``(k, 2)`` array of attained ``(x, y)`` values, sorted by ``x``.
    params : numpy.ndarray
        The sweep parameters (``mu`` or targets) belonging to ``points``.
    results : list of OptimResult
        The underlying optimisation results (same order as ``points``).
    n, m : int
        Grid size.
    kind : str
        Checkerboard kind.
    method : str
        ``"mu"`` or ``"target"``.
    """

    x: str
    y: str
    side: str
    points: np.ndarray
    params: np.ndarray
    results: list = field(repr=False, default_factory=list)
    n: int = 0
    m: int = 0
    kind: str = "pi"
    method: str = "mu"
    constraints: tuple = ()

    @property
    def copulas(self) -> list:
        """The optimal checkerboard copulas (built lazily)."""
        return [r.copula for r in self.results]

    @property
    def xs(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def ys(self) -> np.ndarray:
        return self.points[:, 1]

    def __len__(self) -> int:
        return len(self.points)

    def interpolate(self, x) -> np.ndarray:
        """Piecewise-linear interpolation of the traced boundary at ``x``."""
        return np.interp(x, self.points[:, 0], self.points[:, 1])

    def plot(self, ax=None, label: str | None = None, marker: str = "o", **style):
        """Plot the traced points (and connecting line) on ``ax``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        from copul.regions.style import apply_paper_axes

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
            apply_paper_axes(ax, self.x, self.y)
        if label is None:
            label = f"{self.side} boundary, n={self.n} ({self.kind})"
        kw = {"ms": 3, "lw": 1.2}
        kw.update(style)
        ax.plot(self.points[:, 0], self.points[:, 1], marker=marker, label=label, **kw)
        return ax


def _default_mus(n_points: int, x_affine: bool) -> np.ndarray:
    if x_affine:
        th = np.linspace(-np.pi / 2, np.pi / 2, n_points + 2)[1:-1]
        return np.tan(th)
    return np.concatenate([[0.0], np.geomspace(1e-2, 1e2, max(n_points - 1, 1))])


def trace_boundary(
    x: str,
    y: str,
    side: str = "upper",
    n: int = 32,
    m: int | None = None,
    kind: str = "pi",
    method: str = "mu",
    mus: Sequence[float] | None = None,
    targets: Sequence[float] | None = None,
    n_points: int = 25,
    constraints: Iterable = (),
    solver: str | Sequence[str] | None = None,
    dedupe: float = 1e-9,
) -> BoundaryTrace:
    r"""Trace the upper or lower boundary of the attainable :math:`(x, y)` region.

    Parameters
    ----------
    x, y : str
        Measure keys for the horizontal and vertical axis.
    side : {"upper", "lower"}
        Which boundary (in the :math:`y`-direction) to trace.
    n, m : int
        Grid size (``m`` defaults to ``n``).
    kind : {"pi", "min", "w"}
        Checkerboard kind.
    method : {"mu", "target"}
        Supporting-hyperplane sweep or constraint sweep (see module docstring).
    mus : sequence of float, optional
        Values of :math:`\mu` for ``method="mu"``.  Defaults to
        ``[0] + geomspace(1e-2, 1e2)`` if :math:`x` is not affine and to
        :math:`\tan\theta` on a uniform :math:`\theta` grid otherwise.
    targets : sequence of float, optional
        Targets :math:`t` for ``method="target"`` (default: ``n_points``
        equispaced values in the range of :math:`x`); infeasible targets
        (e.g. :math:`\rho=1` on a ``BivCheckPi`` grid) are skipped.
    n_points : int
        Number of sweep values when ``mus``/``targets`` are not given.
    constraints : iterable
        Constraints added to the problem (shape names such as ``"si"``,
        ``(measure, op, value)`` tuples or raw ``cvxpy`` constraints).
    solver : str or sequence of str, optional
        Solver(s) to use.
    dedupe : float
        Points closer than this (in max-norm) to the previous one are dropped.

    Returns
    -------
    BoundaryTrace

    Raises
    ------
    NonConvexError
        If the requested sweep is not a convex program (e.g. the upper
        boundary in the :math:`\xi`-direction).
    """
    cp = require_cvxpy()
    side = side.lower()
    if side not in ("upper", "lower"):
        raise ValueError("side must be 'upper' or 'lower'")
    xk, yk = resolve(x), resolve(y)
    prob = CheckerboardProblem(n=n, m=m, kind=kind, constraints=constraints)
    fx: QuadraticForm = prob.form(xk)
    fy: QuadraticForm = prob.form(yk)
    ex, ey = to_cvxpy(fx, prob.P), to_cvxpy(fy, prob.P)
    base = prob._base_constraints()
    for f, op, val in prob._measure_constraints:
        e = to_cvxpy(f, prob.P)
        base.append(e <= val if op == "<=" else e >= val if op == ">=" else e == val)
    sgn = 1.0 if side == "upper" else -1.0  # maximise sgn * y

    results: list[OptimResult] = []
    params_used: list[float] = []

    def record(param: float, problem, used: str) -> None:
        raw = np.asarray(prob.P.value, dtype=float)
        Pb = balance(raw)
        res = OptimResult(
            P=Pb,
            kind=prob.kind,
            status=problem.status,
            objective=float(problem.value),
            sense="max",
            solver=used,
            method="convex",
            residual=_marginal_residual(raw),
        )
        results.append(res)
        params_used.append(float(param))

    if method == "mu":
        x_curv = fx.curvature()
        if (sgn * fy).curvature() not in ("affine", "concave"):
            raise NonConvexError(
                f"The {side} boundary in direction of {yk!r} is not a convex sweep."
            )
        if mus is None:
            mus = _default_mus(n_points, x_curv == "affine")
        mus = np.asarray(mus, dtype=float)
        if x_curv == "convex" and np.any(mus < 0):
            raise NonConvexError("mu must be >= 0 when x is convex (e.g. xi).")
        if x_curv == "concave" and np.any(mus > 0):
            raise NonConvexError("mu must be <= 0 when x is concave.")
        if x_curv == "indefinite":
            raise NonConvexError(f"{xk!r} is indefinite; no convex sweep possible.")
        if x_curv == "affine":
            mu_par = cp.Parameter(name="mu")
            obj = sgn * ey - mu_par * ex
        else:
            mu_par = (
                cp.Parameter(nonneg=True, name="mu")
                if x_curv == "convex"
                else cp.Parameter(nonpos=True, name="mu")
            )
            obj = sgn * ey - mu_par * ex
        problem = cp.Problem(cp.Maximize(obj), base)
        for mu in mus:
            mu_par.value = float(mu)
            try:
                used = _solve(problem, solver=solver)
            except RuntimeError:
                continue
            record(mu, problem, used)
    elif method == "target":
        x_curv = fx.curvature()
        if (sgn * fy).curvature() not in ("affine", "concave"):
            raise NonConvexError(
                f"The {side} boundary in direction of {yk!r} is not a convex sweep."
            )
        t = cp.Parameter(name="t")
        if x_curv == "affine":
            cons = [*base, ex == t]
        elif x_curv == "convex":
            cons = [*base, ex <= t]
        elif x_curv == "concave":
            cons = [*base, ex >= t]
        else:
            raise NonConvexError(f"{xk!r} is indefinite; no convex sweep possible.")
        if targets is None:
            from copul.regions.measures import MEASURES

            lo, hi = MEASURES[xk].range
            targets = np.linspace(lo, hi, n_points)
        problem = cp.Problem(cp.Maximize(sgn * ey), cons)
        for tv in targets:
            t.value = float(tv)
            try:
                used = _solve(problem, solver=solver)
            except RuntimeError:
                continue
            record(tv, problem, used)
    else:
        raise ValueError("method must be 'mu' or 'target'")

    pts = np.array([[fx.value(r.P), fy.value(r.P)] for r in results], dtype=float).reshape(-1, 2)
    order = np.lexsort((pts[:, 1], pts[:, 0])) if len(pts) else np.array([], int)
    keep: list[int] = []
    for i in order:
        if keep and np.max(np.abs(pts[i] - pts[keep[-1]])) < dedupe:
            continue
        keep.append(int(i))
    return BoundaryTrace(
        x=xk,
        y=yk,
        side=side,
        points=pts[keep],
        params=np.asarray(params_used)[keep] if keep else np.array([]),
        results=[results[i] for i in keep],
        n=prob.n,
        m=prob.m,
        kind=prob.kind,
        method=method,
        constraints=tuple(constraints),
    )
