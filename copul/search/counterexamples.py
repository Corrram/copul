r"""
Random search for counterexamples to inequalities between dependence measures.

For an inequality :math:`L(C)\;\mathrm{rel}\;R(C)` the *margin* of a copula is

.. math::

   \operatorname{margin}(C)=\begin{cases}
   L(C)-R(C), & \mathrm{rel}=\text{"<="},\\
   R(C)-L(C), & \mathrm{rel}=\text{">="},\\
   |L(C)-R(C)|, & \mathrm{rel}=\text{"=="},\end{cases}

so that a copula violates the inequality iff its margin exceeds ``tol``.
:func:`check_inequality` samples copulas, tracks the maximal margin and
optionally refines the worst checkerboard by a local random search on its mass
matrix (convex mixing with random doubly stochastic matrices and
marginal-preserving 2x2 mass swaps).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from copul.optim.checkerboard_formulas import checkerboard_copula, measure_form
from copul.regions.measures import evaluate, resolve
from copul.search.sampling import (
    _apply_condition,
    random_checkerboards,
    random_mass_matrix,
)

__all__ = [
    "Counterexample",
    "InequalityReport",
    "check_inequality",
    "find_counterexample",
]

Side = str | float | int | Callable[[Any], float]
_RELATIONS = ("<=", ">=", "==")


@dataclass
class Counterexample:
    """A copula violating an inequality.

    Attributes
    ----------
    copula : object
        The violating copula.
    lhs, rhs : float
        Values of both sides.
    margin : float
        Amount of violation (positive).
    relation : str
        The tested relation.
    """

    copula: Any
    lhs: float
    rhs: float
    margin: float
    relation: str

    @property
    def matrix(self) -> np.ndarray | None:
        """Mass matrix if the counterexample is a checkerboard."""
        m = getattr(self.copula, "matr", None)
        return None if m is None else np.asarray(m, dtype=float)


@dataclass
class InequalityReport:
    """Summary of :func:`check_inequality`.

    Attributes
    ----------
    holds : bool
        ``max_violation <= tol`` on all tested copulas.
    max_violation : float
        Largest margin found (negative if the inequality held strictly).
    argmax : object
        Copula attaining :attr:`max_violation`.
    lhs, rhs : float
        Both sides at :attr:`argmax`.
    n_samples : int
        Number of sampled copulas (excluding refinement steps).
    n_violations : int
        Number of sampled copulas with margin ``> tol``.
    refined : bool
        Whether the local refinement improved the sampled maximum.
    relation : str
    tol : float
    """

    holds: bool
    max_violation: float
    argmax: Any
    lhs: float
    rhs: float
    n_samples: int
    n_violations: int
    refined: bool
    relation: str
    tol: float

    @property
    def counterexample(self) -> Counterexample | None:
        if self.holds:
            return None
        return Counterexample(self.argmax, self.lhs, self.rhs, self.max_violation, self.relation)


def _kind_of(copula: Any) -> str | None:
    try:
        from copul.checkerboard.biv_check_min import BivCheckMin
        from copul.checkerboard.biv_check_pi import BivCheckPi
        from copul.checkerboard.biv_check_w import BivCheckW
    except ImportError:  # pragma: no cover
        return None
    if isinstance(copula, BivCheckW):
        return "w"
    if isinstance(copula, BivCheckMin):
        return "min"
    if isinstance(copula, BivCheckPi):
        return "pi"
    return None


def _side_value(side: Side, copula: Any, P: np.ndarray | None, kind: str | None) -> float:
    if isinstance(side, (int, float, np.floating)):
        return float(side)
    if isinstance(side, str):
        key = resolve(side)
        if P is not None and kind is not None:
            m, n = P.shape
            try:
                return measure_form(key, m, n, kind).value(P)
            except ValueError:  # e.g. footrule on rectangular grids
                return float("nan")
        return evaluate(copula, key)
    return float(side(copula))


def _margin(lhs: float, rhs: float, relation: str) -> float:
    if relation == "<=":
        return lhs - rhs
    if relation == ">=":
        return rhs - lhs
    return abs(lhs - rhs)


def _iter_sampler(sampler, n_iter: int, rng, condition, kind, grid) -> Iterable:
    if sampler is None:
        return random_checkerboards(n_iter, grid=grid, kind=kind, condition=condition, rng=rng)
    if callable(sampler) and not hasattr(sampler, "__iter__"):
        return (sampler() for _ in range(n_iter))
    return (c for _, c in zip(range(n_iter), sampler))


def check_inequality(
    lhs: Side,
    rhs: Side,
    relation: str = "<=",
    sampler=None,
    n_iter: int = 2000,
    tol: float = 1e-10,
    refine: bool = True,
    refine_steps: int = 200,
    condition: str | Callable | None = None,
    kind: str = "pi",
    grid: int | tuple[int, int] = (2, 50),
    rng=None,
) -> InequalityReport:
    r"""Test ``lhs relation rhs`` on random copulas.

    Parameters
    ----------
    lhs, rhs : str, float or callable
        Measure keys (evaluated exactly for checkerboards), constants, or
        callables ``copula -> float``.
    relation : {"<=", ">=", "=="}
    sampler : iterable, callable or None
        Source of copulas: an iterable/iterator of copulas, a zero-argument
        callable returning one, or ``None`` for
        :func:`~copul.search.random_checkerboards` with ``kind``, ``grid`` and
        ``condition``.
    n_iter : int
        Number of sampled copulas.
    tol : float
        Violations are margins ``> tol``.
    refine : bool
        Locally improve the worst checkerboard (only for checkerboards).
    refine_steps : int
        Number of refinement proposals.
    condition : str or callable, optional
        Structural condition for the default sampler, re-applied to every
        refinement proposal (see :func:`~copul.search.random_checkerboards`).
    kind : {"pi", "min", "w"}
        Checkerboard class of the default sampler.
    grid : int or (int, int)
        Grid size (range) of the default sampler.
    rng : numpy.random.Generator or int, optional

    Returns
    -------
    InequalityReport

    Examples
    --------
    >>> from copul.search import check_inequality
    >>> check_inequality("footrule", -0.5, ">=", n_iter=50, rng=0).holds
    True
    >>> check_inequality("xi", "rho", "<=", n_iter=200, rng=0).holds
    False
    """
    relation = {"<": "<=", ">": ">=", "=": "=="}.get(relation, relation)
    if relation not in _RELATIONS:
        raise ValueError(f"relation must be one of {_RELATIONS}")
    rng = np.random.default_rng(rng)
    best = (-np.inf, None, np.nan, np.nan)
    n_samples = 0
    n_viol = 0
    for cop in _iter_sampler(sampler, n_iter, rng, condition, kind, grid):
        k = _kind_of(cop)
        P = np.asarray(cop.matr, dtype=float) if k is not None else None
        a = _side_value(lhs, cop, P, k)
        b = _side_value(rhs, cop, P, k)
        n_samples += 1
        mg = _margin(a, b, relation)
        if not np.isfinite(mg):
            continue
        if mg > tol:
            n_viol += 1
        if mg > best[0]:
            best = (mg, cop, a, b)
    if best[1] is None:
        raise RuntimeError("no copula could be evaluated")

    refined = False
    if refine and _kind_of(best[1]) is not None and refine_steps > 0:
        new = _refine(best, lhs, rhs, relation, refine_steps, rng, condition)
        if new[0] > best[0]:
            best, refined = new, True
    mg, cop, a, b = best
    return InequalityReport(
        holds=bool(mg <= tol),
        max_violation=float(mg),
        argmax=cop,
        lhs=float(a),
        rhs=float(b),
        n_samples=n_samples,
        n_violations=n_viol,
        refined=refined,
        relation=relation,
        tol=tol,
    )


def _refine(best, lhs, rhs, relation, steps, rng, condition):
    r"""Local random search on the mass matrix of the worst checkerboard.

    Two marginal-preserving moves are used: a *2x2 swap* moving mass
    :math:`\delta` from cells ``(i,l),(k,j)`` to ``(i,j),(k,l)``, and convex
    mixing with a random permutation matrix.  The step size adapts to the
    success rate.
    """
    mg, cop, a, b = best
    kind = _kind_of(cop)
    P = np.asarray(cop.matr, dtype=float)
    P = P / P.sum()
    m, n = P.shape
    eps = 0.2
    for _ in range(steps):
        if rng.random() < 0.7 and m > 1 and n > 1:
            i, k = rng.choice(m, size=2, replace=False)
            j, l = rng.choice(n, size=2, replace=False)
            delta = eps * min(P[i, l], P[k, j])
            if delta <= 0:
                eps = max(1e-4, eps * 0.9)
                continue
            prop = P.copy()
            prop[i, j] += delta
            prop[k, l] += delta
            prop[i, l] -= delta
            prop[k, j] -= delta
        else:
            perm = np.zeros((m, n))
            if m == n:
                perm[np.arange(n), rng.permutation(n)] = 1.0 / n
            else:
                perm = random_mass_matrix(n, m, rng=rng)
            prop = (1.0 - eps) * P + eps * perm
        if isinstance(condition, str):
            prop = _apply_condition(prop, condition)  # preserves the marginals
        c = checkerboard_copula(prop, kind, clean=False)
        if callable(condition) and not condition(c):
            eps = max(1e-4, eps * 0.8)
            continue
        pa = _side_value(lhs, c, prop, kind)
        pb = _side_value(rhs, c, prop, kind)
        pm = _margin(pa, pb, relation)
        if np.isfinite(pm) and pm > mg:
            mg, cop, a, b, P = pm, c, pa, pb, prop
            eps = min(0.5, eps * 1.5)
        else:
            eps = max(1e-4, eps * 0.9)
    return (mg, cop, a, b)


def find_counterexample(
    lhs: Side,
    rhs: Side,
    relation: str = "<=",
    sampler=None,
    n_iter: int = 2000,
    tol: float = 1e-10,
    refine: bool = True,
    **kwargs,
) -> Counterexample | None:
    """Search for a copula violating ``lhs relation rhs``.

    Returns the copula with the *largest* violation found (after optional
    refinement), or ``None`` if every sampled copula satisfies the inequality
    up to ``tol``.  See :func:`check_inequality` for the parameters.

    Examples
    --------
    >>> from copul.search import find_counterexample
    >>> ce = find_counterexample("tau", "rho", "<=", n_iter=300, rng=1)
    >>> ce is not None and ce.lhs > ce.rhs
    True
    """
    rep = check_inequality(
        lhs,
        rhs,
        relation,
        sampler=sampler,
        n_iter=n_iter,
        tol=tol,
        refine=refine,
        **kwargs,
    )
    return rep.counterexample
