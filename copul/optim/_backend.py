"""Optional ``cvxpy`` backend: lazy import and solver fallbacks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_INSTALL_HINT = (
    "copul.optim requires the optional dependency 'cvxpy'. "
    "Install it with `pip install copul[optim]` (or `pip install cvxpy`)."
)

#: Preferred solver order; the first installed one that succeeds is used.
SOLVER_PREFERENCE: tuple[str, ...] = ("CLARABEL", "ECOS", "SCS", "OSQP")
_LP_PREFERENCE: tuple[str, ...] = ("HIGHS", "CLARABEL", "ECOS", "SCS")


def require_cvxpy():
    """Import and return :mod:`cvxpy`, raising a helpful ``ImportError`` if absent."""
    try:
        import cvxpy
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_INSTALL_HINT) from exc
    return cvxpy


def installed_solvers() -> list[str]:
    """Names of the solvers available to ``cvxpy``."""
    return list(require_cvxpy().installed_solvers())


def candidate_solvers(problem: Any, solver: str | Sequence[str] | None = None) -> list[str]:
    """Ordered list of solvers to try for ``problem``."""
    avail = set(installed_solvers())
    if solver is not None:
        wanted = [solver] if isinstance(solver, str) else list(solver)
        return [s for s in wanted if s.upper() in avail] or list(wanted)
    try:
        is_lp = problem.objective.args[0].is_affine() and _is_lp(problem)
    except (AttributeError, IndexError):  # pragma: no cover - defensive
        is_lp = False
    pref = _LP_PREFERENCE if is_lp else SOLVER_PREFERENCE
    return [s for s in pref if s in avail]


def _is_lp(problem: Any) -> bool:
    for c in problem.constraints:
        for a in c.args:
            if not a.is_affine():
                return False
    return True


def solve(
    problem: Any,
    solver: str | Sequence[str] | None = None,
    verbose: bool = False,
    **kwargs,
) -> str:
    """Solve ``problem`` trying several solvers; return the name of the one used.

    Raises
    ------
    RuntimeError
        If no solver reaches an ``optimal`` (or ``optimal_inaccurate``) status.
    """
    cp = require_cvxpy()
    errors: list[str] = []
    for name in candidate_solvers(problem, solver):
        try:
            problem.solve(solver=name, verbose=verbose, **kwargs)
        except (cp.error.SolverError, ValueError, ArithmeticError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            return name
        errors.append(f"{name}: status {problem.status}")
        if problem.status in ("infeasible", "unbounded"):
            break
    raise RuntimeError("cvxpy could not solve the problem. Attempts: " + "; ".join(errors))
