r"""
Exact dependence-measure formulas for bivariate checkerboard copulas.

A bivariate checkerboard copula on an :math:`m\times n` grid is determined by
its mass matrix :math:`P\in\mathbb{R}_{\ge0}^{m\times n}` with row sums
:math:`1/m` and column sums :math:`1/n` (so that :math:`\sum_{ij}P_{ij}=1`)
together with a *local* copula :math:`D` that distributes the mass of every
cell :math:`[\tfrac im,\tfrac{i+1}m]\times[\tfrac jn,\tfrac{j+1}n]`:

* ``kind="pi"``  -- :math:`D=\Pi` (:class:`~copul.checkerboard.biv_check_pi.BivCheckPi`),
* ``kind="min"`` -- :math:`D=M` (:class:`~copul.checkerboard.biv_check_min.BivCheckMin`),
* ``kind="w"``   -- :math:`D=W` (:class:`~copul.checkerboard.biv_check_w.BivCheckW`).

For all three kinds, Spearman's :math:`\rho`, Blest's :math:`\nu`, Spearman's
footrule :math:`\psi`, Gini's :math:`\gamma` and Blomqvist's :math:`\beta` are
**affine** functions of :math:`P`, Chatterjee's :math:`\xi` is a **convex
quadratic** function of :math:`P` and Kendall's :math:`\tau` is an
**indefinite** quadratic function of :math:`P`.  This module exposes these
functions as :class:`QuadraticForm` objects

.. math::

   f(P)=\langle W,P\rangle + c + \sum_k \alpha_k\, q_k(P),

with quadratic terms :math:`q_k` of the two types

* :class:`RowGram` -- :math:`q(P)=\operatorname{tr}(P\,G\,P^\top)=\sum_i p_i G p_i^\top`
  with :math:`G\succeq0` (convex; used for :math:`\xi`),
* :class:`Bilinear` -- :math:`q(P)=\operatorname{tr}(A\,P\,B\,P^\top)`
  (in general indefinite; used for :math:`\tau`).

Derivation
----------
Writing :math:`U=(i+s)/m`, :math:`V=(j+t)/n` inside cell :math:`(i,j)` with
:math:`(s,t)\sim D` and :math:`a=m-i`, :math:`b=n-j` (0-based :math:`i,j`),

.. math::

   \rho = 12\,\mathbb E[(1-U)(1-V)]-3,\qquad
   \nu = 12\,\mathbb E[(1-U)^2(1-V)]-2,

so that with :math:`e_1=\mathbb E_D[st]` and :math:`e_2=\mathbb E_D[s^2t]`
(:math:`\Pi: \tfrac14,\tfrac16`; :math:`M: \tfrac13,\tfrac14`;
:math:`W: \tfrac16,\tfrac1{12}`)

.. math::

   \rho = \frac{12}{mn}\sum_{ij}P_{ij}\big(ab-\tfrac a2-\tfrac b2+e_1\big)-3,\qquad
   \nu  = \frac{12}{m^2n}\sum_{ij}P_{ij}
          \big(a^2b-\tfrac{a^2}2-ab+2ae_1+\tfrac b3-e_2\big)-2 .

For square grids the diagonal and anti-diagonal sections give

.. math::

   \int_0^1 C(t,t)\,dt=\frac1n\sum_{ij}P_{ij}\,
      \begin{cases} n-\max(i,j)-\tfrac12,& i\ne j,\\ n-1-i+d,& i=j,\end{cases}
   \qquad
   \int_0^1 C(t,1-t)\,dt=\frac1n\sum_{ij}P_{ij}\,
      \begin{cases} n-1-i-j,& i+j<n-1,\\ d',& i+j=n-1,\\ 0,&\text{else},\end{cases}

with :math:`d=\int_0^1D(s,s)ds` (:math:`\tfrac13,\tfrac12,\tfrac14`) and
:math:`d'=\int_0^1 D(s,1-s)ds` (:math:`\tfrac16,\tfrac14,0`).  Blomqvist's
:math:`\beta=4\sum_{ij}P_{ij}D(\alpha_i,\beta_j)-1` with the covered cell
fractions :math:`\alpha_i=\mathrm{clip}(m/2-i,0,1)`,
:math:`\beta_j=\mathrm{clip}(n/2-j,0,1)`.  Finally, with the strictly upper
triangular ones matrix :math:`T` and
:math:`\mathcal M=TT^\top+T^\top+\tfrac13I` (as in
:meth:`BivCheckPi.chatterjees_xi`),

.. math::

   \xi = \frac{6m}{n}\operatorname{tr}\!\big(P\,G\,P^\top\big)-2,\qquad
   G=\tfrac12(\mathcal M+\mathcal M^\top)+\tfrac{\kappa}{6}I,

where :math:`\kappa=0` for ``"pi"`` and :math:`\kappa=1` for ``"min"`` and
``"w"`` (inside a cell the conditional distribution of :math:`M`/:math:`W` is
an indicator).  :math:`G` is positive definite, hence :math:`\xi` is strictly
convex in :math:`P`.  Kendall's
:math:`\tau=1-\operatorname{tr}(\Xi_mP\Xi_nP^\top)+\sigma\operatorname{tr}(PP^\top)`
with :math:`\Xi_k=2L_k-I_k` (:math:`L_k` lower-triangular ones incl. diagonal)
and :math:`\sigma=0,+1,-1` for ``"pi"``, ``"min"``, ``"w"``.

All formulas are verified against independent numerical integration in
``tests/optim/test_checkerboard_formulas.py``.

Notes
-----
The formulas for ``"min"``/``"w"`` are derived independently of the methods of
:class:`BivCheckMin` / :class:`BivCheckW`.  At the time of writing,
``BivCheckMin.blests_nu`` only adds a diagonal term, whereas the exact add-on
is :math:`+1/(mn)` (and :math:`-1/(mn)` for ``BivCheckW``), exactly as for
Spearman's :math:`\rho`.

Examples
--------
>>> import numpy as np
>>> from copul.optim.checkerboard_formulas import measure_form
>>> P = np.eye(4) / 4
>>> round(measure_form("rho", 4, 4, "pi").value(P), 6)  # 1 - 1/n^2
0.9375
>>> round(measure_form("rho", 4, 4, "min").value(P), 6)  # the copula M
1.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import numpy as np

from copul.regions.measures import resolve

__all__ = [
    "KINDS",
    "Bilinear",
    "QuadraticForm",
    "RowGram",
    "affine_constraints",
    "checkerboard_copula",
    "is_feasible_mass_matrix",
    "measure_form",
    "measure_values",
    "normalize_kind",
]

Kind = Literal["pi", "min", "w"]
KINDS: tuple[str, ...] = ("pi", "min", "w")

_KIND_ALIASES = {
    "pi": "pi",
    "checkpi": "pi",
    "bivcheckpi": "pi",
    "min": "min",
    "m": "min",
    "checkmin": "min",
    "bivcheckmin": "min",
    "w": "w",
    "checkw": "w",
    "bivcheckw": "w",
}

# local-copula moments: e1=E[st], e2=E[s^2 t], d=int D(s,s), d'=int D(s,1-s)
_LOCAL = {
    "pi": {
        "e1": 1 / 4,
        "e2": 1 / 6,
        "d": 1 / 3,
        "dp": 1 / 6,
        "kappa": 0.0,
        "sigma": 0.0,
    },
    "min": {
        "e1": 1 / 3,
        "e2": 1 / 4,
        "d": 1 / 2,
        "dp": 1 / 4,
        "kappa": 1.0,
        "sigma": 1.0,
    },
    "w": {
        "e1": 1 / 6,
        "e2": 1 / 12,
        "d": 1 / 4,
        "dp": 0.0,
        "kappa": 1.0,
        "sigma": -1.0,
    },
}


def normalize_kind(kind: str) -> str:
    """Return the canonical checkerboard kind (``"pi"``, ``"min"`` or ``"w"``)."""
    k = str(kind).strip().lower()
    if k not in _KIND_ALIASES:
        raise ValueError(f"Unknown checkerboard kind {kind!r}; use one of {KINDS}.")
    return _KIND_ALIASES[k]


def _local_cdf(kind: str, s, t):
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    if kind == "pi":
        return s * t
    if kind == "min":
        return np.minimum(s, t)
    return np.maximum(s + t - 1.0, 0.0)


# ----------------------------------------------------------------------------
# Quadratic terms
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class RowGram:
    r"""Quadratic term :math:`q(P)=\operatorname{tr}(P G P^\top)` with :math:`G\succeq 0`.

    Parameters
    ----------
    G : numpy.ndarray
        Symmetric positive semidefinite ``(n, n)`` matrix.
    """

    G: np.ndarray

    def value(self, P: np.ndarray) -> float:
        return float(np.sum((P @ self.G) * P))

    def grad(self, P: np.ndarray) -> np.ndarray:
        return 2.0 * P @ self.G

    @property
    def factor(self) -> np.ndarray:
        """Matrix :math:`R` with :math:`G=RR^\\top` (so :math:`q=\\|PR\\|_F^2`)."""
        return _psd_factor(self.G)

    def curvature(self) -> str:
        return "convex"

    def dense(self, m: int, n: int) -> np.ndarray:
        """Symmetric ``(mn, mn)`` matrix of the form in row-major ``vec(P)``."""
        return np.kron(np.eye(m), self.G)


@dataclass(frozen=True)
class Bilinear:
    r"""Quadratic term :math:`q(P)=\operatorname{tr}(A P B P^\top)`.

    Parameters
    ----------
    A : numpy.ndarray
        ``(m, m)`` matrix.
    B : numpy.ndarray
        ``(n, n)`` matrix.
    """

    A: np.ndarray
    B: np.ndarray

    def value(self, P: np.ndarray) -> float:
        return float(np.trace(self.A @ P @ self.B @ P.T))

    def grad(self, P: np.ndarray) -> np.ndarray:
        return self.A @ P @ self.B + self.A.T @ P @ self.B.T

    def dense(self, m: int, n: int) -> np.ndarray:
        """Symmetric ``(mn, mn)`` matrix of the form in row-major ``vec(P)``."""
        Q = np.kron(self.A, self.B.T)
        return 0.5 * (Q + Q.T)

    def curvature(self) -> str:
        m, n = self.A.shape[0], self.B.shape[0]
        ev = np.linalg.eigvalsh(self.dense(m, n))
        tol = 1e-10 * max(1.0, float(np.max(np.abs(ev))))
        if ev.min() >= -tol:
            return "convex"
        if ev.max() <= tol:
            return "concave"
        return "indefinite"


def _psd_factor(G: np.ndarray) -> np.ndarray:
    """Return ``R`` with ``R @ R.T == G`` for symmetric PSD ``G``."""
    w, V = np.linalg.eigh(0.5 * (G + G.T))
    w = np.clip(w, 0.0, None)
    keep = w > 1e-14 * max(1.0, float(w.max(initial=0.0)))
    return V[:, keep] * np.sqrt(w[keep])


# ----------------------------------------------------------------------------
# Quadratic forms
# ----------------------------------------------------------------------------
@dataclass
class QuadraticForm:
    r"""A function :math:`f(P)=\langle W,P\rangle+c+\sum_k\alpha_k q_k(P)`.

    Instances support ``+``, ``-``, scalar ``*`` and ``/`` so that objectives
    such as ``rho - 0.5 * xi`` can be assembled before being handed to a
    solver.

    Attributes
    ----------
    W : numpy.ndarray
        Linear weight matrix of shape ``(m, n)``.
    const : float
        Constant offset :math:`c`.
    terms : list of tuple
        Pairs ``(alpha, term)`` with ``term`` a :class:`RowGram` or
        :class:`Bilinear`.
    name : str
        Optional description.
    """

    W: np.ndarray
    const: float = 0.0
    terms: list = field(default_factory=list)
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.W.shape)  # type: ignore[return-value]

    @property
    def is_affine(self) -> bool:
        return all(abs(a) == 0 for a, _ in self.terms)

    def value(self, P: np.ndarray) -> float:
        """Evaluate :math:`f(P)`."""
        P = np.asarray(P, dtype=float)
        val = float(np.sum(self.W * P)) + self.const
        for a, t in self.terms:
            val += a * t.value(P)
        return val

    __call__ = value

    def grad(self, P: np.ndarray) -> np.ndarray:
        """Euclidean gradient :math:`\\nabla f(P)` (same shape as ``P``)."""
        P = np.asarray(P, dtype=float)
        g = np.array(self.W, dtype=float, copy=True)
        for a, t in self.terms:
            g = g + a * t.grad(P)
        return g

    def curvature(self) -> str:
        """``"affine"``, ``"convex"``, ``"concave"`` or ``"indefinite"``."""
        if self.is_affine:
            return "affine"
        signs = set()
        for a, t in self.terms:
            if a == 0:
                continue
            c = t.curvature()
            if c == "indefinite":
                return self._dense_curvature()
            convex = (c == "convex") == (a > 0)
            signs.add("convex" if convex else "concave")
        if len(signs) == 1:
            return signs.pop()
        return self._dense_curvature()

    def dense_hessian(self) -> np.ndarray:
        """Symmetric matrix :math:`Q` with :math:`f=x^\\top Qx+\\dots` for row-major ``x=vec(P)``."""
        m, n = self.shape
        Q = np.zeros((m * n, m * n))
        for a, t in self.terms:
            if a != 0:
                Q += a * t.dense(m, n)
        return Q

    def _dense_curvature(self) -> str:
        ev = np.linalg.eigvalsh(self.dense_hessian())
        tol = 1e-10 * max(1.0, float(np.max(np.abs(ev))))
        if ev.min() >= -tol:
            return "convex"
        if ev.max() <= tol:
            return "concave"
        return "indefinite"

    # --- arithmetic ------------------------------------------------------
    def _check(self, other: QuadraticForm) -> None:
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")

    def __add__(self, other):
        if isinstance(other, QuadraticForm):
            self._check(other)
            return QuadraticForm(
                self.W + other.W,
                self.const + other.const,
                list(self.terms) + list(other.terms),
                f"({self.name} + {other.name})",
            )
        if np.isscalar(other):
            return QuadraticForm(
                self.W.copy(), self.const + float(other), list(self.terms), self.name
            )
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return self * (-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, s):
        if not np.isscalar(s):
            return NotImplemented
        s = float(s)
        return QuadraticForm(
            s * self.W,
            s * self.const,
            [(s * a, t) for a, t in self.terms],
            f"{s:g}*{self.name}",
        )

    __rmul__ = __mul__

    def __truediv__(self, s):
        return self * (1.0 / float(s))


# ----------------------------------------------------------------------------
# Measure constructors
# ----------------------------------------------------------------------------
def _ab(m: int, n: int):
    a = (m - np.arange(m, dtype=float))[:, None]
    b = (n - np.arange(n, dtype=float))[None, :]
    return a, b


def _chatterjee_gram(n: int) -> np.ndarray:
    T = np.ones((n, n)) - np.tri(n)
    M = T @ T.T + T.T + np.eye(n) / 3.0
    return 0.5 * (M + M.T)


def _square(measure: str, m: int, n: int) -> None:
    if m != n:
        raise ValueError(
            f"The exact {measure!r} formula for checkerboards is implemented for "
            f"square grids only (got {m}x{n})."
        )


@lru_cache(maxsize=256)
def _measure_form_cached(key: str, m: int, n: int, kind: str) -> QuadraticForm:
    loc = _LOCAL[kind]
    a, b = _ab(m, n)
    if key == "rho":
        W = 12.0 / (m * n) * (a * b - a / 2 - b / 2 + loc["e1"])
        return QuadraticForm(W, -3.0, [], "rho")
    if key == "nu":
        K = a**2 * b - a**2 / 2 - a * b + 2 * a * loc["e1"] + b / 3 - loc["e2"]
        return QuadraticForm(12.0 / (m * m * n) * K, -2.0, [], "nu")
    if key in ("footrule", "gamma"):
        _square(key, m, n)
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        D = (n - np.maximum(i, j) - 0.5).astype(float)
        D[np.arange(n), np.arange(n)] = n - 1 - np.arange(n) + loc["d"]
        D /= n
        if key == "footrule":
            return QuadraticForm(6.0 * D, -2.0, [], "footrule")
        s = i + j
        A = np.where(s < n - 1, n - 1 - s, 0).astype(float)
        A[s == n - 1] = loc["dp"]
        A /= n
        return QuadraticForm(4.0 * (D + A), -2.0, [], "gamma")
    if key == "beta":
        al = np.clip(m / 2.0 - np.arange(m), 0.0, 1.0)[:, None]
        be = np.clip(n / 2.0 - np.arange(n), 0.0, 1.0)[None, :]
        return QuadraticForm(4.0 * _local_cdf(kind, al, be), -1.0, [], "beta")
    if key == "xi":
        G = _chatterjee_gram(n) + loc["kappa"] / 6.0 * np.eye(n)
        return QuadraticForm(np.zeros((m, n)), -2.0, [(6.0 * m / n, RowGram(G))], "xi")
    if key == "tau":
        Xm = 2 * np.tri(m) - np.eye(m)
        Xn = 2 * np.tri(n) - np.eye(n)
        terms = [(-1.0, Bilinear(Xm, Xn))]
        if loc["sigma"] != 0:
            terms.append((loc["sigma"], RowGram(np.eye(n))))
        return QuadraticForm(np.zeros((m, n)), 1.0, terms, "tau")
    raise KeyError(key)  # pragma: no cover


def measure_form(measure: str, m: int, n: int | None = None, kind: str = "pi") -> QuadraticForm:
    r"""Exact representation of a dependence measure on ``m x n`` checkerboards.

    Parameters
    ----------
    measure : str
        Measure key (see :mod:`copul.regions.measures`).
    m, n : int
        Grid size (``n`` defaults to ``m``).
    kind : {"pi", "min", "w"}
        Local copula of the checkerboard.

    Returns
    -------
    QuadraticForm
        A fresh copy that may be modified freely.

    Raises
    ------
    ValueError
        For footrule/gamma on non-square grids.
    """
    n = m if n is None else n
    q = _measure_form_cached(resolve(measure), int(m), int(n), normalize_kind(kind))
    return QuadraticForm(q.W.copy(), q.const, list(q.terms), q.name)


def measure_values(
    P: np.ndarray,
    kind: str = "pi",
    measures: Sequence[str] = ("xi", "rho", "tau", "footrule", "gamma", "beta", "nu"),
) -> dict[str, float]:
    """Evaluate several measures exactly on the mass matrix ``P``.

    Footrule and gamma are reported as ``nan`` on non-square grids.
    """
    P = np.asarray(P, dtype=float)
    m, n = P.shape
    out: dict[str, float] = {}
    for key in measures:
        k = resolve(key)
        try:
            out[k] = measure_form(k, m, n, kind).value(P)
        except ValueError:
            out[k] = float("nan")
    return out


# ----------------------------------------------------------------------------
# Feasibility helpers
# ----------------------------------------------------------------------------
def affine_constraints(m: int, n: int):
    """Row/column-sum targets ``(row_target, col_target) = (1/m, 1/n)``."""
    return np.full(m, 1.0 / m), np.full(n, 1.0 / n)


def is_feasible_mass_matrix(P: np.ndarray, tol: float = 1e-8) -> bool:
    """Check ``P >= 0`` with row sums ``1/m`` and column sums ``1/n`` (up to ``tol``)."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        return False
    m, n = P.shape
    return bool(
        P.min() >= -tol
        and np.allclose(P.sum(axis=1), 1.0 / m, atol=tol)
        and np.allclose(P.sum(axis=0), 1.0 / n, atol=tol)
    )


def checkerboard_copula(P: np.ndarray, kind: str = "pi", clean: bool = True):
    """Build the copul checkerboard copula object for the mass matrix ``P``.

    Parameters
    ----------
    P : numpy.ndarray
        Mass matrix (it is clipped at zero and rescaled to total mass one when
        ``clean`` is true; solver output typically has tiny negative entries).
    kind : {"pi", "min", "w"}
        Which checkerboard class to instantiate.
    """
    P = np.asarray(P, dtype=float)
    if clean:
        P = np.clip(P, 0.0, None)
        P = P / P.sum()
    k = normalize_kind(kind)
    if k == "pi":
        from copul.checkerboard.biv_check_pi import BivCheckPi

        return BivCheckPi(P)
    if k == "min":
        from copul.checkerboard.biv_check_min import BivCheckMin

        return BivCheckMin(P)
    from copul.checkerboard.biv_check_w import BivCheckW

    return BivCheckW(P)
