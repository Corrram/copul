r"""
Penalty convex--concave procedure (CCP) for non-convex checkerboard problems.

Every measure form :math:`f` (:class:`~copul.optim.checkerboard_formulas.QuadraticForm`)
is split into a difference of convex functions

.. math::

   f(P)=\langle W,P\rangle+c+\sum_k g_k(P)-\sum_l h_l(P),
   \qquad g_k,h_l \text{ convex sums of squares}

(:class:`RowGram` terms are already of this type; indefinite
:class:`Bilinear` terms such as Kendall's :math:`\tau` are split along the
eigen-decomposition of their Hessian).  To *maximise* :math:`f` the convex
parts :math:`g_k` are replaced by their tangent planes at the current iterate
:math:`P^{(t)}`, which yields a concave minorant; constraints
:math:`f\le t` are replaced by the convex majorant obtained by linearising the
:math:`h_l`.  Each majorised non-convex constraint receives a slack
:math:`s\ge0` penalised by :math:`\tau^{(t)}s` with increasing
:math:`\tau^{(t)}` (Lipp & Boyd, 2016, *Variations and extension of the
convex--concave procedure*).  The objective values of the iterates are
monotone once the slacks vanish; the procedure converges to a critical point,
i.e. a *local* optimum -- use several starts for important computations.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from copul.optim._backend import require_cvxpy
from copul.optim._backend import solve as _solve
from copul.optim.checkerboard_formulas import (
    Bilinear,
    QuadraticForm,
    RowGram,
)

__all__ = ["dc_split", "solve_ccp", "starting_points"]


class _SqPiece:
    """Convex piece ``c * ||P @ R||^2`` (row type) or ``c * ||F^T vec(P)||^2`` (dense)."""

    def __init__(self, coef: float, R: np.ndarray, dense: bool, shape):
        self.coef = float(coef)
        self.R = R
        self.dense = dense
        self.shape = shape

    def value(self, P: np.ndarray) -> float:
        if self.dense:
            return self.coef * float(np.sum((self.R.T @ P.reshape(-1)) ** 2))
        return self.coef * float(np.sum((P @ self.R) ** 2))

    def grad(self, P: np.ndarray) -> np.ndarray:
        if self.dense:
            g = 2.0 * self.R @ (self.R.T @ P.reshape(-1))
            return self.coef * g.reshape(self.shape)
        return self.coef * 2.0 * P @ self.R @ self.R.T

    def expr(self, Pvar):
        cp = require_cvxpy()
        if self.dense:
            return self.coef * cp.sum_squares(self.R.T @ cp.vec(Pvar, order="C"))
        return self.coef * cp.sum_squares(Pvar @ self.R)


def dc_split(form: QuadraticForm) -> tuple[list[_SqPiece], list[_SqPiece]]:
    """Split the quadratic part of ``form`` into ``(g_pieces, h_pieces)``.

    ``form = affine + sum(g) - sum(h)`` with all pieces convex.
    """
    m, n = form.shape
    g: list[_SqPiece] = []
    h: list[_SqPiece] = []
    for a, term in form.terms:
        if a == 0:
            continue
        if isinstance(term, RowGram):
            piece = _SqPiece(abs(a), term.factor, False, (m, n))
            (g if a > 0 else h).append(piece)
        elif isinstance(term, Bilinear):
            Q = a * term.dense(m, n)
            w, V = np.linalg.eigh(Q)
            tol = 1e-12 * max(1.0, float(np.max(np.abs(w))))
            pos, neg = w > tol, w < -tol
            if pos.any():
                g.append(_SqPiece(1.0, V[:, pos] * np.sqrt(w[pos]), True, (m, n)))
            if neg.any():
                h.append(_SqPiece(1.0, V[:, neg] * np.sqrt(-w[neg]), True, (m, n)))
        else:  # pragma: no cover
            raise TypeError(term)
    return g, h


def starting_points(m: int, n: int, rng=None, n_random: int = 3) -> list[np.ndarray]:
    """Candidate starting mass matrices: :math:`\\Pi`, :math:`M`, :math:`W` and random ones."""
    rng = np.random.default_rng(rng)
    # checkerboard approximations of M and W (exact overlaps of the cells)
    ri = np.arange(m + 1) / m
    cj = np.arange(n + 1) / n
    lo = np.maximum(ri[:-1, None], cj[None, :-1])
    hi = np.minimum(ri[1:, None], cj[None, 1:])
    Mm = np.clip(hi - lo, 0.0, None)
    pts = [np.full((m, n), 1.0 / (m * n)), Mm, Mm[:, ::-1].copy()]
    from copul.optim.problem import balance

    for _ in range(n_random):
        pts.append(balance(rng.random((m, n)) ** 3))
    return pts


def solve_ccp(
    Pvar,
    base_constraints: list,
    objective: QuadraticForm,
    sense: str,
    measure_constraints: Sequence[tuple[QuadraticForm, str, float]] = (),
    kind: str = "pi",
    x0: np.ndarray | None = None,
    n_starts: int = 1,
    max_iter: int = 60,
    tol: float = 1e-8,
    tau0: float = 1.0,
    tau_mult: float = 2.0,
    tau_max: float = 1e6,
    rng=None,
    solver=None,
):
    """Run the penalty CCP; returns ``(P, history, iterations, solver)``.

    Parameters
    ----------
    Pvar : cvxpy.Variable
        The mass-matrix variable.
    base_constraints : list
        Convex constraints (marginals, shapes, raw constraints).
    objective : QuadraticForm
    sense : {"max", "min"}
    measure_constraints : sequence of (QuadraticForm, op, value)
    x0 : numpy.ndarray, optional
        Starting mass matrix.  By default the best of :func:`starting_points`
        (w.r.t. objective minus constraint violation) is used; with
        ``n_starts > 1`` the procedure is run from the best ``n_starts``
        candidates and the best feasible result is returned.
    max_iter, tol : int, float
        Iteration limit and relative objective tolerance.
    tau0, tau_mult, tau_max : float
        Penalty schedule for the constraint slacks.
    """
    cp = require_cvxpy()
    m, n = Pvar.shape
    sgn = 1.0 if sense == "max" else -1.0
    obj = sgn * objective  # we always maximise obj
    og, oh = dc_split(obj)

    # constraints in the form F(P) <= t
    cons_le: list[tuple[QuadraticForm, float]] = []
    for f, op, val in measure_constraints:
        if op in ("<=", "=="):
            cons_le.append((f, val))
        if op in (">=", "=="):
            cons_le.append((-1.0 * f, -val))

    def violation(P):
        return sum(max(0.0, f.value(P) - t) for f, t in cons_le)

    # ---- build the parametrised surrogate once -------------------------
    Gobj = cp.Parameter((m, n), name="grad_obj")
    obj_expr = cp.sum(cp.multiply(obj.W + 0.0, Pvar)) + cp.sum(cp.multiply(Gobj, Pvar))
    for piece in oh:
        obj_expr = obj_expr - piece.expr(Pvar)
    cons = list(base_constraints)
    tau = cp.Parameter(nonneg=True, name="tau")
    slack_terms = []
    params = []
    for f, t in cons_le:
        g, h = dc_split(f)
        if not h:  # convex constraint: no linearisation, no slack
            e = cp.sum(cp.multiply(f.W, Pvar)) + f.const
            for piece in g:
                e = e + piece.expr(Pvar)
            cons.append(e <= t)
            params.append(None)
            continue
        Gh = cp.Parameter((m, n), name="grad_h")
        ch = cp.Parameter(name="const_h")
        s = cp.Variable(nonneg=True)
        e = cp.sum(cp.multiply(f.W, Pvar)) + f.const - cp.sum(cp.multiply(Gh, Pvar)) - ch
        for piece in g:
            e = e + piece.expr(Pvar)
        cons.append(e <= t + s)
        slack_terms.append(s)
        params.append((Gh, ch, h))
    penalty = tau * cp.sum(cp.hstack(slack_terms)) if slack_terms else 0.0
    problem = cp.Problem(cp.Maximize(obj_expr - penalty), cons)

    def set_params(P, tau_val):
        grad = np.zeros((m, n))
        for piece in og:
            grad += piece.grad(P)
        Gobj.value = grad
        tau.value = tau_val
        for prm in params:
            if prm is None:
                continue
            Gh, ch, hs = prm
            gr = np.zeros((m, n))
            cst = 0.0
            for piece in hs:
                gP = piece.grad(P)
                gr += gP
                cst += piece.value(P) - float(np.sum(gP * P))
            Gh.value = gr
            ch.value = cst

    # ---- starting points ------------------------------------------------
    if x0 is not None:
        starts = [np.asarray(x0, dtype=float)]
    else:
        cands = starting_points(m, n, rng=rng)
        merit = [obj.value(P) - 10.0 * violation(P) for P in cands]
        order = np.argsort(merit)[::-1]
        starts = [cands[i] for i in order[: max(1, int(n_starts))]]

    best = None
    used = ""
    for P0 in starts:
        P = P0
        hist = [obj.value(P)]
        tau_val = tau0
        it = 0
        for it in range(1, max_iter + 1):
            set_params(P, tau_val)
            used = _solve(problem, solver=solver)
            P_new = np.asarray(Pvar.value, dtype=float)
            val = obj.value(P_new)
            hist.append(val)
            slack = sum(float(s.value) for s in slack_terms) if slack_terms else 0.0
            done = abs(val - hist[-2]) <= tol * (1.0 + abs(val)) and slack <= 1e-9
            P = P_new
            if done and it > 1:
                break
            tau_val = min(tau_val * tau_mult, tau_max)
        feas = violation(P)
        cand = (feas > 1e-6, -obj.value(P), P, [sgn * v for v in hist], it)
        if best is None or cand[:2] < best[:2]:
            best = cand
    _, _, P, hist, it = best
    return P, hist, it, used
