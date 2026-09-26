r"""
Exact optimisation of dependence measures over checkerboard copulas.

The decision variable is the mass matrix :math:`P\in\mathbb R^{m\times n}_{\ge0}`
of a checkerboard copula (row sums :math:`1/m`, column sums :math:`1/n`).
All measures are expressed *exactly* via
:mod:`copul.optim.checkerboard_formulas`: :math:`\rho,\nu,\psi,\gamma,\beta` are
affine, :math:`\xi` is convex quadratic and :math:`\tau` is indefinite
quadratic in :math:`P`.  Consequently

* ``maximize(rho - mu * xi)``, ``maximize("rho", subject_to={"xi": ("<=", t)})``,
  ``minimize("xi", subject_to={"beta": ("==", b)})`` ... are convex programs
  solved to global optimality (``method="convex"``, the default);
* problems that are not DCP -- e.g. maximising :math:`\xi`, constraining
  :math:`\xi\ge t`, or anything involving :math:`\tau` -- raise
  :class:`NonConvexError` unless ``method="ccp"`` is requested, which runs a
  penalty convex--concave procedure (a local method: it returns a feasible
  checkerboard with a *lower* bound on the maximum).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from copul.optim._backend import require_cvxpy
from copul.optim._backend import solve as _solve
from copul.optim.checkerboard_formulas import (
    Bilinear,
    QuadraticForm,
    RowGram,
    _psd_factor,
    checkerboard_copula,
    measure_form,
    measure_values,
    normalize_kind,
)
from copul.optim.shapes import cvxpy_shape_constraints, resolve_shape, shape_violation
from copul.regions.measures import resolve

__all__ = [
    "CheckerboardProblem",
    "NonConvexError",
    "OptimResult",
    "balance",
    "to_cvxpy",
]

Objective = Any  # str | QuadraticForm | Mapping[str, float] | cvxpy expression
_OPS = ("<=", ">=", "==")


class NonConvexError(ValueError):
    """Raised when a requested problem is not a valid convex (DCP) program."""


def _norm_op(op: str) -> str:
    op = {"<": "<=", ">": ">=", "=": "=="}.get(op, op)
    if op not in _OPS:
        raise ValueError(f"relation must be one of {_OPS}, got {op!r}")
    return op


def balance(P: np.ndarray, iters: int = 20000, tol: float = 1e-15) -> np.ndarray:
    """Clip ``P`` at zero and Sinkhorn-balance it to row sums ``1/m``, column sums ``1/n``.

    Solver output satisfies the marginal constraints only up to the solver
    tolerance; this removes the (tiny) residual so that the checkerboard is an
    exact copula.  Rows/columns without mass are left untouched.
    """
    P = np.clip(np.asarray(P, dtype=float), 0.0, None)
    m, n = P.shape
    if P.sum() <= 0:
        return np.full((m, n), 1.0 / (m * n))
    P = P / P.sum()
    for _ in range(iters):
        r = P.sum(axis=1)
        P = P * np.divide(1.0 / m, r, out=np.ones_like(r), where=r > 0)[:, None]
        c = P.sum(axis=0)
        P = P * np.divide(1.0 / n, c, out=np.ones_like(c), where=c > 0)[None, :]
        if np.max(np.abs(P.sum(axis=1) - 1.0 / m)) <= tol:
            break
    return P


def to_cvxpy(form: QuadraticForm, P) -> Any:
    """Translate a :class:`QuadraticForm` into a DCP ``cvxpy`` expression of ``P``.

    Raises
    ------
    NonConvexError
        If the form contains an indefinite quadratic term (e.g. Kendall's tau).
    """
    cp = require_cvxpy()
    expr = cp.sum(cp.multiply(form.W, P)) + form.const
    for a, term in form.terms:
        if a == 0:
            continue
        if isinstance(term, RowGram):
            expr = expr + a * cp.sum_squares(P @ term.factor)
        elif isinstance(term, Bilinear):
            curv = term.curvature()
            if curv == "indefinite":
                raise NonConvexError(
                    f"The quadratic term of {form.name or 'this form'} is indefinite "
                    "(e.g. Kendall's tau) and has no DCP representation; use "
                    "method='ccp' for a local (heuristic) optimisation."
                )
            m, n = form.shape
            Q = term.dense(m, n)
            sgn = 1.0 if curv == "convex" else -1.0
            F = _psd_factor(sgn * Q)
            expr = expr + a * sgn * cp.sum_squares(F.T @ cp.vec(P, order="C"))
        else:  # pragma: no cover
            raise TypeError(f"Unknown quadratic term {term!r}")
    return expr


@dataclass
class OptimResult:
    """Result of a checkerboard optimisation.

    Attributes
    ----------
    P : numpy.ndarray
        Optimal mass matrix (clipped and Sinkhorn-balanced, i.e. an exact copula).
    kind : str
        Checkerboard kind (``"pi"``, ``"min"`` or ``"w"``).
    status : str
        Solver status.
    objective : float
        Objective value at ``P`` (recomputed exactly when the objective is a
        measure/:class:`QuadraticForm`).
    sense : str
        ``"max"`` or ``"min"``.
    solver : str
        Name of the solver that produced the solution.
    method : str
        ``"convex"`` or ``"ccp"``.
    iterations : int
        Number of convex subproblems solved.
    residual : float
        Largest violation of the marginal constraints by the raw solver output.
    history : list of float
        Objective values of the CCP iterates (``method="ccp"`` only).
    """

    P: np.ndarray
    kind: str
    status: str
    objective: float
    sense: str
    solver: str
    method: str = "convex"
    iterations: int = 1
    residual: float = 0.0
    history: list = field(default_factory=list)
    _values: dict | None = field(default=None, repr=False)

    @property
    def value(self) -> float:
        """Alias of :attr:`objective`."""
        return self.objective

    @property
    def values(self) -> dict[str, float]:
        """All seven measures at the optimum (exact closed forms)."""
        if self._values is None:
            self._values = measure_values(self.P, self.kind)
        return self._values

    def __getitem__(self, key: str) -> float:
        return self.values[resolve(key)]

    @property
    def copula(self):
        """The optimal checkerboard as a copul copula object (e.g. ``BivCheckPi``)."""
        return checkerboard_copula(self.P, self.kind)


class CheckerboardProblem:
    r"""Optimisation problem over :math:`m\times n` checkerboard copulas.

    Parameters
    ----------
    n : int
        Number of columns (and rows unless ``m`` is given).
    m : int, optional
        Number of rows.
    kind : {"pi", "min", "w"}
        Local copula of the checkerboard (``BivCheckPi``, ``BivCheckMin``,
        ``BivCheckW``).
    constraints : iterable, optional
        Constraints applied to every solve; see :meth:`add_constraint`.

    Examples
    --------
    >>> import copul.optim as co  # doctest: +SKIP
    >>> prob = co.CheckerboardProblem(n=16)  # doctest: +SKIP
    >>> prob.maximize("rho").value  # 1 - 1/n^2  # doctest: +SKIP
    0.99609375
    >>> res = prob.maximize(prob.expr("rho") - 0.5 * prob.expr("xi"))  # doctest: +SKIP
    >>> res.values["xi"], res.values["rho"]  # doctest: +SKIP
    """

    def __init__(
        self,
        n: int,
        m: int | None = None,
        kind: str = "pi",
        constraints: Iterable = (),
    ) -> None:
        cp = require_cvxpy()
        self.n = int(n)
        self.m = int(m) if m is not None else int(n)
        if self.m < 1 or self.n < 1:
            raise ValueError("grid sizes must be positive")
        self.kind = normalize_kind(kind)
        self.P = cp.Variable((self.m, self.n), nonneg=True, name="P")
        self._marginals = [
            cp.sum(self.P, axis=1) == 1.0 / self.m,
            cp.sum(self.P, axis=0) == 1.0 / self.n,
        ]
        self._shapes: list[str] = []
        self._measure_constraints: list[tuple[QuadraticForm, str, float]] = []
        self._raw: list = []
        for c in constraints:
            self.add_constraint(c)

    def __repr__(self) -> str:
        extra = ", ".join(self._shapes)
        return (
            f"CheckerboardProblem(m={self.m}, n={self.n}, kind={self.kind!r}"
            + (f", shapes=[{extra}]" if extra else "")
            + ")"
        )

    # ------------------------------------------------------------------
    # measures
    # ------------------------------------------------------------------
    def form(self, measure: str | QuadraticForm) -> QuadraticForm:
        """Exact :class:`QuadraticForm` of ``measure`` on this grid."""
        if isinstance(measure, QuadraticForm):
            if measure.shape != (self.m, self.n):
                raise ValueError("form has a different grid shape")
            return measure
        return measure_form(measure, self.m, self.n, self.kind)

    def expr(self, measure: str | QuadraticForm):
        """Exact ``cvxpy`` expression of ``measure`` in the variable :attr:`P`.

        Affine for ``rho, nu, footrule, gamma, beta``, convex for ``xi``.

        Raises
        ------
        NonConvexError
            For Kendall's tau (indefinite); use :meth:`form` with ``method="ccp"``.
        """
        return to_cvxpy(self.form(measure), self.P)

    def _as_form(self, objective: Objective) -> QuadraticForm | None:
        if isinstance(objective, (str, QuadraticForm)):
            return self.form(objective)
        if isinstance(objective, Mapping):
            total = None
            for k, c in objective.items():
                f = float(c) * self.form(k)
                total = f if total is None else total + f
            if total is None:
                raise ValueError("empty objective mapping")
            return total
        return None  # a raw cvxpy expression

    # ------------------------------------------------------------------
    # constraints
    # ------------------------------------------------------------------
    def add_constraint(self, constraint, op: str | None = None, value: float | None = None):
        """Add a constraint applied to all subsequent solves.

        Parameters
        ----------
        constraint : str, tuple, QuadraticForm or cvxpy constraint
            * a shape name -- ``"si"``, ``"sd"``, ``"si2"``, ``"ltd"``, ``"lti"``,
              ``"rti"``, ``"rtd"``, ``"pqd"``, ``"nqd"``, ``"exchangeable"``,
              ``"radially_symmetric"`` (see :mod:`copul.optim.shapes`);
            * a measure key/form together with ``op`` and ``value``, or a tuple
              ``(measure, op, value)``, e.g. ``("xi", "<=", 0.3)``;
            * a raw ``cvxpy`` constraint in the variable :attr:`P`.

        Returns
        -------
        CheckerboardProblem
            ``self`` (for chaining).
        """
        if isinstance(constraint, tuple) and op is None:
            constraint, op, value = constraint
        if op is not None:
            self._measure_constraints.append((self.form(constraint), _norm_op(op), float(value)))
            return self
        if isinstance(constraint, str):
            try:
                name = resolve_shape(constraint)
            except KeyError:
                raise ValueError(
                    f"{constraint!r} is neither a shape constraint nor accompanied "
                    "by a relation; use add_constraint('xi', '<=', 0.3)."
                ) from None
            cvxpy_shape_constraints(self.P, name, self.kind)  # validates kind
            if name not in self._shapes:
                self._shapes.append(name)
            return self
        self._raw.append(constraint)
        return self

    def _parse_subject_to(self, subject_to) -> tuple[list, list]:
        forms: list[tuple[QuadraticForm, str, float]] = []
        raw: list = []
        if subject_to is None:
            return forms, raw
        if isinstance(subject_to, Mapping):
            items = []
            for k, spec in subject_to.items():
                if isinstance(spec, tuple):
                    items.append((k, spec[0], spec[1]))
                else:
                    items.append((k, "==", spec))
        elif (
            isinstance(subject_to, (list, tuple))
            and subject_to
            and isinstance(subject_to[0], str)
            and len(subject_to) == 3
            and subject_to[1] in (*_OPS, "<", ">", "=")
        ):
            items = [tuple(subject_to)]
        else:
            items = list(subject_to)
        for it in items:
            if isinstance(it, tuple) and len(it) == 3:
                forms.append((self.form(it[0]), _norm_op(it[1]), float(it[2])))
            elif isinstance(it, str):
                raw.extend(cvxpy_shape_constraints(self.P, resolve_shape(it), self.kind))
            else:
                raw.append(it)
        return forms, raw

    def _base_constraints(self) -> list:
        cons = list(self._marginals)
        for s in self._shapes:
            cons.extend(cvxpy_shape_constraints(self.P, s, self.kind))
        cons.extend(self._raw)
        return cons

    # ------------------------------------------------------------------
    # solving
    # ------------------------------------------------------------------
    def maximize(self, objective: Objective, subject_to=None, **kwargs) -> OptimResult:
        """Maximise ``objective``; see :meth:`solve` for the arguments."""
        return self.solve(objective, "max", subject_to=subject_to, **kwargs)

    def minimize(self, objective: Objective, subject_to=None, **kwargs) -> OptimResult:
        """Minimise ``objective``; see :meth:`solve` for the arguments."""
        return self.solve(objective, "min", subject_to=subject_to, **kwargs)

    def solve(
        self,
        objective: Objective,
        sense: str = "max",
        subject_to=None,
        method: str = "convex",
        solver: str | Sequence[str] | None = None,
        verbose: bool = False,
        **ccp_options,
    ) -> OptimResult:
        """Optimise ``objective`` over the checkerboards satisfying all constraints.

        Parameters
        ----------
        objective : str, QuadraticForm, dict or cvxpy expression
            A measure key, a :class:`QuadraticForm` (e.g.
            ``prob.form("rho") - 0.5 * prob.form("xi")``), a mapping
            ``{measure: coefficient}`` or a ``cvxpy`` expression in :attr:`P`
            (e.g. ``prob.expr("rho") - 0.5 * prob.expr("xi")``; convex method only).
        sense : {"max", "min"}
        subject_to : dict, tuple or list, optional
            Additional constraints for this solve only: ``{"xi": ("<=", 0.3)}``,
            ``{"beta": 0.5}`` (equality), ``("xi", "<=", 0.3)``, a list of such
            tuples, shape names or raw ``cvxpy`` constraints.
        method : {"convex", "ccp"}
            ``"convex"`` requires a DCP-valid problem (global optimum);
            ``"ccp"`` runs the penalty convex--concave procedure of
            :func:`copul.optim.ccp.solve_ccp` (local optimum; objective and
            measure constraints must be measures/forms).
        solver : str or sequence of str, optional
            Solver(s) to try; defaults to the installed ones among
            CLARABEL/ECOS/SCS/OSQP (HiGHS first for LPs).
        **ccp_options
            Passed to :func:`copul.optim.ccp.solve_ccp` (``x0``, ``max_iter``,
            ``tol``, ``n_starts`` ...).

        Returns
        -------
        OptimResult
        """
        cp = require_cvxpy()
        sense = {"maximize": "max", "minimize": "min"}.get(sense, sense)
        if sense not in ("max", "min"):
            raise ValueError("sense must be 'max' or 'min'")
        form = self._as_form(objective)
        extra_forms, extra_raw = self._parse_subject_to(subject_to)
        mforms = self._measure_constraints + extra_forms
        base = self._base_constraints() + extra_raw

        if method == "ccp":
            if form is None:
                raise TypeError("method='ccp' needs a measure key, mapping or QuadraticForm")
            from copul.optim.ccp import solve_ccp

            P, hist, iters, solver_used = solve_ccp(
                self.P,
                base,
                form,
                sense,
                mforms,
                kind=self.kind,
                solver=solver,
                **ccp_options,
            )
            Pb = balance(P)
            return OptimResult(
                P=Pb,
                kind=self.kind,
                status="ccp_converged" if iters else "ccp",
                objective=form.value(Pb),
                sense=sense,
                solver=solver_used,
                method="ccp",
                iterations=iters,
                residual=_marginal_residual(P),
                history=hist,
            )
        if method != "convex":
            raise ValueError("method must be 'convex' or 'ccp'")

        obj_expr = to_cvxpy(form, self.P) if form is not None else objective
        if sense == "max" and not obj_expr.is_concave():
            raise NonConvexError(
                "Maximising a non-concave objective is not a convex program "
                "(e.g. maximising xi). Use method='ccp' for a local solution."
            )
        if sense == "min" and not obj_expr.is_convex():
            raise NonConvexError(
                "Minimising a non-convex objective is not a convex program. "
                "Use method='ccp' for a local solution."
            )
        cons = list(base)
        for f, op, val in mforms:
            e = to_cvxpy(f, self.P)
            ok = (
                (op == "<=" and e.is_convex())
                or (op == ">=" and e.is_concave())
                or (op == "==" and e.is_affine())
            )
            if not ok:
                raise NonConvexError(
                    f"Constraint {f.name or 'form'} {op} {val} is not convex "
                    f"({f.curvature()} left-hand side). Use method='ccp'."
                )
            cons.append(e <= val if op == "<=" else e >= val if op == ">=" else e == val)
        prob = cp.Problem(cp.Maximize(obj_expr) if sense == "max" else cp.Minimize(obj_expr), cons)
        used = _solve(prob, solver=solver, verbose=verbose)
        raw_P = np.asarray(self.P.value, dtype=float)
        Pb = balance(raw_P)
        if form is not None:
            obj_val = form.value(Pb)
        else:
            obj_val = float(prob.value)
        return OptimResult(
            P=Pb,
            kind=self.kind,
            status=prob.status,
            objective=obj_val,
            sense=sense,
            solver=used,
            method="convex",
            residual=_marginal_residual(raw_P),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def evaluate(self, P: np.ndarray) -> dict[str, float]:
        """All measures of the mass matrix ``P`` (exact)."""
        return measure_values(P, self.kind)

    def check(self, P: np.ndarray) -> dict[str, float]:
        """Violation of every registered shape/measure constraint by ``P``."""
        out = {s: shape_violation(P, s, self.kind) for s in self._shapes}
        for f, op, val in self._measure_constraints:
            v = f.value(P)
            viol = v - val if op == "<=" else val - v if op == ">=" else abs(v - val)
            out[f"{f.name} {op} {val:g}"] = max(0.0, viol)
        return out


def _marginal_residual(P: np.ndarray) -> float:
    m, n = P.shape
    return float(
        max(
            np.max(np.abs(P.sum(axis=1) - 1.0 / m)),
            np.max(np.abs(P.sum(axis=0) - 1.0 / n)),
            max(0.0, -float(P.min())),
        )
    )
