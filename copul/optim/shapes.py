r"""
Exact linear shape constraints for checkerboard mass matrices.

Positive-dependence and symmetry properties of a checkerboard copula
:math:`C_P` are *linear* conditions on its mass matrix :math:`P`.  With the
cumulative sums

.. math::

   S = P L_n^\top \;(S_{ij}=\textstyle\sum_{l\le j}P_{il}),\qquad
   G_{ab} = \sum_{i<a,\,j<b} P_{ij} = C_P\!\big(\tfrac am,\tfrac bn\big),

(:math:`L_k` the lower-triangular ones matrix) the following equivalences hold
for ``kind="pi"`` (bilinear interpolation inside every cell):

``"si"`` (stochastically increasing, :math:`u\mapsto\partial_1C(u,v)` non-increasing)
    :math:`S_{i,j}\ge S_{i+1,j}` for all :math:`i,j`, since
    :math:`\partial_1C_P(u,v)=m\,(S_{i,j-1}+P_{ij}t)` is constant in :math:`u`
    on every row and affine in :math:`v` on every column.
``"sd"``
    the reverse inequalities.
``"si2"`` / ``"sd2"``
    the same for :math:`v\mapsto\partial_2C(u,v)` (apply the above to
    :math:`P^\top`).
``"ltd"`` (:math:`u\mapsto C(u,v)/u` non-increasing)
    :math:`a\,G_{a+1,b}\le(a+1)\,G_{a,b}` for :math:`1\le a<m`, :math:`1\le b<n`.
    On row :math:`a` the ratio is :math:`m(G_a(v)+s\,r_a(v))/(a+s)` whose
    derivative in :math:`s` has the sign of :math:`a\,r_a(v)-G_a(v)`; both
    terms are affine in :math:`v` on every column.
``"lti"``
    the reverse inequalities.
``"rti"`` / ``"rtd"``
    ``"ltd"`` / ``"lti"`` of the survival copula, whose mass matrix is the
    :math:`180^\circ` rotation of :math:`P`.
``"pqd"`` / ``"nqd"``
    :math:`G_{ab}\ge ab/(mn)` (resp. :math:`\le`), because
    :math:`C_P(u,v)-uv` is bilinear on every cell.
``"exchangeable"``
    :math:`P=P^\top`.
``"radially_symmetric"``
    :math:`P=JPJ` with the anti-identity :math:`J`.

For ``kind="min"`` (local copula :math:`M`) the exact conditions are
``"si"``: :math:`S_{i,j-1}\ge S_{i+1,j}` (with :math:`S_{i,-1}=0`), ``"pqd"``
(same grid condition: :math:`C_M\ge C_\Pi` cell-wise and both agree on the
grid) and the two symmetries; for ``kind="w"``: ``"nqd"`` and the two
symmetries.  Other combinations raise :class:`NotImplementedError`.

Every constraint is expressed with ``@`` and ``-`` only, so the very same
code produces ``cvxpy`` constraints and ``numpy`` residuals.

Examples
--------
>>> import numpy as np
>>> from copul.optim.shapes import shape_violation
>>> shape_violation(np.eye(3) / 3, "si")  # M-like checkerboards are SI
0.0
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from copul.optim.checkerboard_formulas import normalize_kind

__all__ = [
    "SHAPES",
    "available_shapes",
    "cvxpy_shape_constraints",
    "resolve_shape",
    "satisfies_shape",
    "shape_residuals",
    "shape_violation",
]

_SHAPE_ALIASES = {
    "si": "si",
    "ci": "si",
    "cis": "si",
    "sto_incr": "si",
    "stochastically_increasing": "si",
    "sd": "sd",
    "cd": "sd",
    "cds": "sd",
    "stochastically_decreasing": "sd",
    "si2": "si2",
    "sd2": "sd2",
    "ltd": "ltd",
    "lti": "lti",
    "rti": "rti",
    "rtd": "rtd",
    "pqd": "pqd",
    "plod": "pqd",
    "nqd": "nqd",
    "exchangeable": "exchangeable",
    "symmetric": "exchangeable",
    "radially_symmetric": "radially_symmetric",
    "radial": "radially_symmetric",
}


def resolve_shape(name: str) -> str:
    """Canonical name of a shape constraint (aliases like ``"ci"`` accepted)."""
    k = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if k not in _SHAPE_ALIASES:
        raise KeyError(f"Unknown shape constraint {name!r}; known: {sorted(SHAPES)}")
    return _SHAPE_ALIASES[k]


def _lower_ones(k: int) -> np.ndarray:
    return np.tri(k)


def _anti(k: int) -> np.ndarray:
    return np.fliplr(np.eye(k))


def _G(P, m: int, n: int):
    """``G[a-1, b-1] = C(a/m, b/n)`` for ``a, b >= 1`` (shape ``(m, n)``)."""
    return _lower_ones(m) @ P @ _lower_ones(n).T


def _si(P, m, n):
    S = P @ _lower_ones(n).T
    return [S[:-1, :-1] - S[1:, :-1]], []


def _sd(P, m, n):
    S = P @ _lower_ones(n).T
    return [S[1:, :-1] - S[:-1, :-1]], []


def _ltd_residual(P, m, n, sign: float):
    if m < 2 or n < 2:
        return [], []
    G = _G(P, m, n)
    # (a+1) G_{a,b} - a G_{a+1,b}  for a = 1..m-1, b = 1..n-1
    Da = np.diag(np.arange(2, m + 1, dtype=float))  # a+1
    Db = np.diag(np.arange(1, m, dtype=float))  # a
    top = G[:-1, :-1]  # G_{a,b}, a = 1..m-1
    nxt = G[1:, :-1]  # G_{a+1,b}
    res = Da @ top - Db @ nxt
    return [sign * res], []


def _rot(P, m, n):
    return _anti(m) @ P @ _anti(n)


def _pqd(P, m, n, sign: float):
    G = _G(P, m, n)
    grid = np.outer(np.arange(1, m + 1), np.arange(1, n + 1)) / (m * n)
    return [sign * (G - grid)], []


def _square(m, n, name):
    if m != n:
        raise ValueError(f"shape constraint {name!r} requires a square grid.")


def _min_si(P, m, n):
    S = P @ _lower_ones(n).T
    # S_{i,j-1} >= S_{i+1,j}, with S_{i,-1} = 0  ->  P_{i+1,0} <= 0
    return [S[:-1, :-1] - S[1:, 1:], -P[1:, 0:1]], []


SHAPES: dict[str, dict[str, Callable]] = {
    "si": {"pi": _si, "min": _min_si},
    "sd": {"pi": _sd},
    "si2": {"pi": lambda P, m, n: _si(P.T, n, m)},
    "sd2": {"pi": lambda P, m, n: _sd(P.T, n, m)},
    "ltd": {"pi": lambda P, m, n: _ltd_residual(P, m, n, 1.0)},
    "lti": {"pi": lambda P, m, n: _ltd_residual(P, m, n, -1.0)},
    "rti": {"pi": lambda P, m, n: _ltd_residual(_rot(P, m, n), m, n, 1.0)},
    "rtd": {"pi": lambda P, m, n: _ltd_residual(_rot(P, m, n), m, n, -1.0)},
    "pqd": {
        "pi": lambda P, m, n: _pqd(P, m, n, 1.0),
        "min": lambda P, m, n: _pqd(P, m, n, 1.0),
    },
    "nqd": {
        "pi": lambda P, m, n: _pqd(P, m, n, -1.0),
        "w": lambda P, m, n: _pqd(P, m, n, -1.0),
    },
    "exchangeable": {
        k: (lambda P, m, n: (_square(m, n, "exchangeable"), ([], [P - P.T]))[1])
        for k in ("pi", "min", "w")
    },
    "radially_symmetric": {
        k: (lambda P, m, n: ([], [P - _rot(P, m, n)])) for k in ("pi", "min", "w")
    },
}


def available_shapes(kind: str = "pi") -> list[str]:
    """Names of the shape constraints implemented for ``kind``."""
    k = normalize_kind(kind)
    return sorted(name for name, impl in SHAPES.items() if k in impl)


def shape_residuals(P, name: str, kind: str = "pi"):
    """Return ``(ineqs, eqs)``: expressions that must be ``>= 0`` resp. ``== 0``.

    ``P`` may be a :class:`numpy.ndarray` or a ``cvxpy`` expression.
    """
    s = resolve_shape(name)
    k = normalize_kind(kind)
    impl = SHAPES[s].get(k)
    if impl is None:
        raise NotImplementedError(
            f"The shape constraint {s!r} is not implemented for kind={k!r} "
            f"(available: {available_shapes(k)})."
        )
    m, n = P.shape
    return impl(P, m, n)


def cvxpy_shape_constraints(P, name: str, kind: str = "pi") -> list:
    """``cvxpy`` constraints encoding the shape ``name`` for the variable ``P``."""
    ineqs, eqs = shape_residuals(P, name, kind)
    return [e >= 0 for e in ineqs] + [e == 0 for e in eqs]


def shape_violation(P: np.ndarray, name: str, kind: str = "pi") -> float:
    """Maximal violation (``0.0`` if satisfied) of the shape ``name`` by ``P``."""
    P = np.asarray(P, dtype=float)
    ineqs, eqs = shape_residuals(P, name, kind)
    v = 0.0
    for e in ineqs:
        if np.size(e):
            v = max(v, float(np.max(-np.asarray(e))))
    for e in eqs:
        if np.size(e):
            v = max(v, float(np.max(np.abs(e))))
    return max(v, 0.0)


def satisfies_shape(P: np.ndarray, name: str, kind: str = "pi", tol: float = 1e-10) -> bool:
    """Whether ``P`` satisfies the shape ``name`` up to ``tol``."""
    return shape_violation(P, name, kind) <= tol
