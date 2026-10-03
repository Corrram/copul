r"""
Best-possible pointwise bounds on sets of copulas.

For a set :math:`\mathcal S` of copulas the pointwise bounds
:math:`\underline C(u,v)=\inf_{C\in\mathcal S}C(u,v)` and
:math:`\overline C(u,v)=\sup_{C\in\mathcal S}C(u,v)` are quasi-copulas
(Nelsen, Quesada-Molina, Rodríguez-Lallena & Úbeda-Flores 2004); they are
*best-possible* by definition.  Implemented sets :math:`\mathcal S`:

==================================  ===============================================  ==============================
set of copulas                      bounds                                           copulas?
==================================  ===============================================  ==============================
all copulas                         :math:`W\le C\le M` (Fréchet–Hoeffding)          yes
:math:`C(a,b)=\theta`               :math:`C_L\le C\le C_U` (shuffles of M)          yes (Nelsen 2006, Thm 3.2.3)
Kendall's :math:`\tau(C)=t`         :math:`T_t^L\le C\le T_t^U`                      yes
Spearman's :math:`\rho(C)=t`        :math:`P_t^L\le C\le P_t^U`                      yes
Blomqvist's :math:`\beta(C)=b`      :math:`C_L, C_U` at :math:`(\tfrac12,\tfrac12)`  yes
diagonal section :math:`\delta`     :math:`B_\delta\le C\le A_\delta`                lower yes, upper in general no
symmetric, diagonal :math:`\delta`  :math:`B_\delta\le C\le K_\delta`                yes
==================================  ===============================================  ==============================

The bounds given :math:`\tau` and :math:`\rho` are those of Nelsen,
Quesada-Molina, Rodríguez-Lallena & Úbeda-Flores (2001).  They follow from
Theorem 3.2.3 of Nelsen (2006): for fixed :math:`(a,b)` the copulas with
:math:`C(a,b)=\theta` form a convex set with smallest element
:math:`C_L^{a,b,\theta}` and largest element :math:`C_U^{a,b,\theta}`, and
:math:`\kappa\in\{\tau,\rho\}` is continuous and monotone in the pointwise
order, so :math:`\{\kappa(C) : C(a,b)=\theta\} = [\kappa(C_L), \kappa(C_U)]`.
With

.. math::

   \tau(C_U) = 1-4(a-\theta)(b-\theta),\qquad
   \tau(C_L) = 4\theta(1-a-b+\theta)-1,

   \rho(C_U) = 1-6(a-\theta)(b-\theta)(a+b-2\theta),\qquad
   \rho(C_L) = 6\theta(1-a-b+\theta)(1-a-b+2\theta)-1,

the smallest (largest) admissible :math:`\theta` given :math:`\kappa=t`
solves :math:`\kappa(C_U)=t` (:math:`\kappa(C_L)=t`), or equals
:math:`W(a,b)` (:math:`M(a,b)`) if that equation has no admissible root,
which gives the closed forms of :func:`kendall_tau_lower_bound` etc.  Each bound is
attained at every point :math:`(a,b)` by a member of the set: by the
shuffle :math:`C_U^{a,b,\theta^*}` (resp. :math:`C_L^{a,b,\theta^*}`) if
:math:`\theta^*\ne W(a,b)` (resp. :math:`\theta^*\ne M(a,b)`), and by a
convex combination of :math:`C_L^{a,b,\theta^*}` and
:math:`C_U^{a,b,\theta^*}` otherwise.  The test suite validates the
formulas, their attainment and the pointwise containment of random
checkerboard copulas.

References
----------
Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Thm 2.2.3 (Fréchet–Hoeffding bounds), Thm 3.2.3, Sects. 3.2.3 (shuffles of
M), 3.2.6 (diagonals), 6.2 (quasi-copulas).

Nelsen, R. B., Quesada-Molina, J. J., Rodríguez-Lallena, J. A. &
Úbeda-Flores, M. (2001). Bounds on bivariate distribution functions with
given margins and measures of association. *Communications in Statistics
- Theory and Methods* 30, 1155–1162.

Nelsen, R. B., Quesada-Molina, J. J., Rodríguez-Lallena, J. A. &
Úbeda-Flores, M. (2004). Best-possible bounds on sets of bivariate
distribution functions. *Journal of Multivariate Analysis* 90, 348–358.

Fredricks, G. A. & Nelsen, R. B. (1997). Copulas constructed from diagonal
sections. In *Distributions with Given Marginals and Moment Problems*,
Kluwer, 129–136.

Mikusiński, P., Sherwood, H. & Taylor, M. D. (1992). Shuffles of Min.
*Stochastica* 13, 61–74.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from copul.family.constructions._base import NumericBivCopula
from copul.theory.diagonal import (
    BertinoCopula,
    DiagonalCopula,
    IntervalMin,
    as_diagonal,
    check_diagonal,
)
from copul.theory.quasi import (
    InversionSamplingCopula,
    NumericQuasiCopula,
    cdf_function,
    is_copula,
)

__all__ = [
    "BoundsResult",
    "MeasureBoundCopula",
    "ShuffleOfM",
    "blomqvist_beta_bounds",
    "bounds_given_diagonal",
    "bounds_given_measure",
    "bounds_given_value",
    "check_frechet_bounds",
    "diagonal_upper_bound",
    "frechet_lower",
    "frechet_upper",
    "kendall_tau_bounds",
    "kendall_tau_lower_bound",
    "kendall_tau_upper_bound",
    "point_value_lower",
    "point_value_upper",
    "spearman_rho_bounds",
    "spearman_rho_cubic_root",
    "spearman_rho_lower_bound",
    "spearman_rho_upper_bound",
]


# ---------------------------------------------------------------------------
# Fréchet–Hoeffding bounds
# ---------------------------------------------------------------------------


def _out(x, *args):
    return float(x) if all(np.ndim(a) == 0 for a in args) else x


def frechet_lower(u, v):
    r"""Lower Fréchet–Hoeffding bound :math:`W(u,v)=\max(u+v-1,0)` (vectorized)."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    return _out(np.maximum(u + v - 1.0, 0.0), u, v)


def frechet_upper(u, v):
    r"""Upper Fréchet–Hoeffding bound :math:`M(u,v)=\min(u,v)` (vectorized)."""
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    return _out(np.minimum(u, v), u, v)


def check_frechet_bounds(C: Any, m: int = 100, tol: float = 1e-10) -> bool:
    r"""Whether :math:`W\le C\le M` holds on the grid :math:`\{k/m\}^2`.

    Every copula and every quasi-copula satisfies the Fréchet–Hoeffding
    bounds (Nelsen 2006, Thm 2.2.3 and Sect. 6.2).
    """
    g = np.linspace(0.0, 1.0, int(m) + 1)
    U, V = np.meshgrid(g, g, indexing="ij")
    Z = cdf_function(C)(U, V)
    return bool(np.all(frechet_lower(U, V) - tol <= Z) and np.all(frechet_upper(U, V) + tol >= Z))


# ---------------------------------------------------------------------------
# shuffles of M with arbitrary strips
# ---------------------------------------------------------------------------


class ShuffleOfM(NumericBivCopula):
    r"""Shuffle of :math:`M` with finitely many strips of arbitrary widths.

    Each strip :math:`i` maps :math:`u\in[a_i, a_i+\ell_i]` linearly onto
    :math:`v\in[c_i, c_i+\ell_i]`, increasing (``direction=+1``) or
    decreasing (``direction=-1``, a "flipped" strip); the
    :math:`u`-intervals and the :math:`v`-intervals must each partition
    :math:`[0,1]` (Mikusiński, Sherwood & Taylor 1992; Nelsen 2006, Sect.
    3.2.3).  The mass is spread uniformly on the segments, so with
    :math:`x_i=\min(\max(u-a_i,0),\ell_i)`, :math:`y_i=\min(\max(v-c_i,0),\ell_i)`

    .. math::

       C(u,v) = \sum_{i:\,+} \min(x_i, y_i) + \sum_{i:\,-}\max(0, x_i+y_i-\ell_i).

    Exact closed forms: Kendall's :math:`\tau = \sum_i s_i\ell_i^2 +
    2\sum_{i<j}\ell_i\ell_j\,\mathrm{sgn}\bigl((a_j-a_i)(c_j-c_i)\bigr)`,
    Spearman's :math:`\rho = 12\,E[UV]-3`, footrule and Gini's gamma
    (piecewise linear diagonals), the tail coefficients, and
    :math:`\xi = 1` (:math:`V` is a function of :math:`U` and vice versa).

    Parameters
    ----------
    pieces : sequence of (a, c, length, direction)
        Strips; zero-length strips are ignored.

    Examples
    --------
    >>> from copul.theory.bounds import ShuffleOfM
    >>> C = ShuffleOfM([(0.0, 0.5, 0.5, 1), (0.5, 0.0, 0.5, 1)])
    >>> C.kendalls_tau(), C.spearmans_rho()      # tau = 2/4 - 2/4, rho = 12 E[UV] - 3
    (0.0, -0.5)
    """

    def __init__(self, pieces: Sequence[Sequence[float]], tol: float = 1e-9):
        rows = []
        for p in pieces:
            a, c, ell, s = p
            a, c, ell, s = float(a), float(c), float(ell), int(s)
            if s not in (1, -1):
                raise ValueError(f"direction must be +1 or -1, got {s}")
            if ell < -tol:
                raise ValueError(f"strip lengths must be nonnegative, got {ell}")
            if ell > tol:
                rows.append((a, c, ell, s))
        if not rows:
            raise ValueError("a shuffle of M needs at least one strip of positive length")
        rows.sort(key=lambda r: r[0])
        arr = np.array(rows, dtype=float)
        a, c, ell = arr[:, 0], arr[:, 1], arr[:, 2]
        for name, start in (("u", a), ("v", c)):
            order = np.argsort(start)
            st, ln = start[order], ell[order]
            ends = st + ln
            if (
                abs(st[0]) > tol
                or abs(ends[-1] - 1.0) > tol
                or np.any(np.abs(st[1:] - ends[:-1]) > tol)
            ):
                raise ValueError(f"the {name}-intervals of the strips must partition [0, 1]")
        self._a, self._c, self._l = a, c, ell
        self._s = arr[:, 3].astype(int)
        self._c_order = np.argsort(c)
        super().__init__()

    @classmethod
    def from_permutation(cls, lengths, perm, flips=None) -> ShuffleOfM:
        r"""Shuffle from strip widths (in :math:`u`-order) and a permutation.

        Strip :math:`i` (of width ``lengths[i]``) becomes the
        ``perm[i]``-th strip (0-based) on the :math:`v`-axis; strips with
        ``flips[i]`` true are decreasing.
        """
        ell = np.asarray(lengths, dtype=float)
        perm = np.asarray(perm, dtype=int)
        if sorted(perm.tolist()) != list(range(ell.size)):
            raise ValueError("perm must be a permutation of 0..len(lengths)-1")
        flips = np.zeros(ell.size, bool) if flips is None else np.asarray(flips, bool)
        a = np.concatenate([[0.0], np.cumsum(ell)[:-1]])
        inv = np.argsort(perm)
        c_sorted = np.concatenate([[0.0], np.cumsum(ell[inv])[:-1]])
        c = np.empty_like(ell)
        c[inv] = c_sorted
        return cls([(a[i], c[i], ell[i], -1 if flips[i] else 1) for i in range(ell.size)])

    @property
    def pieces(self) -> list[tuple[float, float, float, int]]:
        """The strips ``(a, c, length, direction)`` in :math:`u`-order."""
        return [
            (float(a), float(c), float(l), int(s))
            for a, c, l, s in zip(self._a, self._c, self._l, self._s)
        ]

    def __repr__(self) -> str:
        parts = ", ".join(f"({a:.6g}, {c:.6g}, {l:.6g}, {s:+d})" for a, c, l, s in self.pieces)
        return f"ShuffleOfM([{parts}])"

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    # -- evaluation -----------------------------------------------------------
    def support(self, u):
        r"""The support function :math:`v=\varphi(u)` (:math:`V=\varphi(U)` a.s.)."""
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        i = np.clip(np.searchsorted(self._a, u, side="right") - 1, 0, self._a.size - 1)
        x = np.clip(u - self._a[i], 0.0, self._l[i])
        out = np.where(self._s[i] == 1, self._c[i] + x, self._c[i] + self._l[i] - x)
        return float(out) if out.ndim == 0 else out

    def _inverse_support(self, v):
        c_sorted = self._c[self._c_order]
        j = np.clip(np.searchsorted(c_sorted, v, side="right") - 1, 0, self._c.size - 1)
        i = self._c_order[j]
        y = np.clip(v - self._c[i], 0.0, self._l[i])
        return np.where(self._s[i] == 1, self._a[i] + y, self._a[i] + self._l[i] - y)

    def _cdf(self, u, v):
        x = np.clip(u[..., None] - self._a, 0.0, self._l)
        y = np.clip(v[..., None] - self._c, 0.0, self._l)
        val = np.where(self._s == 1, np.minimum(x, y), np.maximum(0.0, x + y - self._l))
        return val.sum(axis=-1)

    def _h1(self, u, v):
        return (v >= self.support(u)).astype(float)

    def _h2(self, u, v):
        return (u >= self._inverse_support(v)).astype(float)

    def _rvs(self, n, rng):
        u = rng.random(n)
        return np.column_stack([u, self.support(u)])

    # -- exact measures -----------------------------------------------------
    def kendalls_tau(self, *args, **kwargs) -> float:
        r""":math:`\tau=\sum_i s_i\ell_i^2+2\sum_{i<j}\ell_i\ell_j\,\mathrm{sgn}((a_j-a_i)(c_j-c_i))`."""
        a, c, l, s = self._a, self._c, self._l, self._s
        sgn = np.sign((a[None, :] - a[:, None]) * (c[None, :] - c[:, None]))
        off = np.triu(np.outer(l, l) * sgn, k=1).sum()
        return float(np.sum(s * l**2) + 2.0 * off)

    def spearmans_rho(self, *args, **kwargs) -> float:
        r""":math:`\rho = 12\sum_i\int_{a_i}^{a_i+\ell_i}u\,\varphi(u)\,du - 3`."""
        a, c, l, s = self._a, self._c, self._l, self._s
        b = a + l
        m2 = (b**2 - a**2) / 2.0
        m3 = (b**3 - a**3) / 3.0
        inc = m3 + (c - a) * m2
        dec = (c + l + a) * m2 - m3
        return float(12.0 * np.sum(np.where(s == 1, inc, dec)) - 3.0)

    def _section_integral(self, anti: bool) -> float:
        """Exact :math:`\\int_0^1 C(t,t)dt` (``anti=False``) or :math:`\\int_0^1 C(t,1-t)dt`."""
        a, c, l, s = self._a, self._c, self._l, self._s
        if anti:
            bp = [a, a + l, 1.0 - c, 1.0 - c - l, np.where(s == 1, (1.0 + a - c) / 2.0, 0.0)]
        else:
            bp = [a, a + l, c, c + l, np.where(s == -1, (a + c + l) / 2.0, 0.0)]
        t = np.unique(np.clip(np.concatenate([np.ravel(b) for b in bp] + [[0.0, 1.0]]), 0, 1))
        y = self.cdf_vectorized(t, 1.0 - t if anti else t)
        return float(np.sum(np.diff(t) * (y[1:] + y[:-1]) / 2.0))

    def spearmans_footrule(self, *args, **kwargs) -> float:
        r""":math:`\phi = 6\int_0^1 C(t,t)\,dt - 2` (exact, piecewise linear diagonal)."""
        return 6.0 * self._section_integral(False) - 2.0

    def ginis_gamma(self, *args, **kwargs) -> float:
        r""":math:`\gamma = 4\int_0^1 [C(t,t)+C(t,1-t)]\,dt - 2` (exact)."""
        return 4.0 * (self._section_integral(False) + self._section_integral(True)) - 2.0

    def chatterjees_xi(self, *args, condition_on_y=False, **kwargs) -> float:
        r""":math:`\xi = 1`: :math:`V=\varphi(U)` and :math:`U=\varphi^{-1}(V)` a.s."""
        return 1.0

    def _breakpoints(self) -> np.ndarray:
        a, c, l = self._a, self._c, self._l
        t = np.concatenate([a, a + l, c, c + l, (a + c + l) / 2.0])
        return np.unique(np.clip(t, 0.0, 1.0))

    def lambda_L(self, *args, **kwargs) -> float:
        r""":math:`\lambda_L = \delta'(0^+)` (exact, piecewise linear diagonal)."""
        t = self._breakpoints()
        e = t[t > 0][0] / 2.0
        return float(self.cdf_vectorized(e, e)) / e

    def lambda_U(self, *args, **kwargs) -> float:
        r""":math:`\lambda_U = 2-\delta'(1^-)` (exact, piecewise linear diagonal)."""
        t = self._breakpoints()
        e = (1.0 - t[t < 1][-1]) / 2.0
        return float((1.0 - 2.0 * (1.0 - e) + self.cdf_vectorized(1.0 - e, 1.0 - e)) / e)


# ---------------------------------------------------------------------------
# bounds given one value C(a, b) = theta  (Nelsen 2006, Thm 3.2.3)
# ---------------------------------------------------------------------------


def _check_point(
    a: float, b: float, theta: float, tol: float = 1e-12
) -> tuple[float, float, float]:
    a, b, theta = float(a), float(b), float(theta)
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError(f"(a, b) must lie in [0, 1]^2, got ({a}, {b})")
    lo, hi = max(a + b - 1.0, 0.0), min(a, b)
    if not (lo - tol <= theta <= hi + tol):
        raise ValueError(
            f"theta must satisfy W(a,b) <= theta <= M(a,b), i.e. {lo:.6g} <= theta <= {hi:.6g}"
        )
    return a, b, min(max(theta, lo), hi)


def point_value_lower(u, v, a: float, b: float, theta: float):
    r"""Lower bound :math:`C_L(u,v)=\max\bigl(0,u+v-1,\theta-(a-u)^+-(b-v)^+\bigr)`
    for copulas with :math:`C(a,b)=\theta` (Nelsen 2006, Thm 3.2.3)."""
    a, b, theta = _check_point(a, b, theta)
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    out = np.maximum(
        np.maximum(0.0, u + v - 1.0), theta - np.maximum(a - u, 0.0) - np.maximum(b - v, 0.0)
    )
    return _out(out, u, v)


def point_value_upper(u, v, a: float, b: float, theta: float):
    r"""Upper bound :math:`C_U(u,v)=\min\bigl(u,v,\theta+(u-a)^++(v-b)^+\bigr)`
    for copulas with :math:`C(a,b)=\theta` (Nelsen 2006, Thm 3.2.3)."""
    a, b, theta = _check_point(a, b, theta)
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    out = np.minimum(np.minimum(u, v), theta + np.maximum(u - a, 0.0) + np.maximum(v - b, 0.0))
    return _out(out, u, v)


def _shuffle_upper(a, b, theta) -> ShuffleOfM:
    # strips: [0,θ]->[0,θ], [θ,a]->[b,a+b-θ], [a,a+b-θ]->[θ,b], rest on the diagonal
    return ShuffleOfM(
        [
            (0.0, 0.0, theta, 1),
            (theta, b, a - theta, 1),
            (a, theta, b - theta, 1),
            (a + b - theta, a + b - theta, 1.0 - a - b + theta, 1),
        ]
    )


def _shuffle_lower(a, b, theta) -> ShuffleOfM:
    r = 1.0 - a - b + theta
    return ShuffleOfM(
        [
            (0.0, 1.0 - a + theta, a - theta, -1),
            (a - theta, b - theta, theta, -1),
            (a, b, r, -1),
            (1.0 - b + theta, 0.0, b - theta, -1),
        ]
    )


@dataclass
class BoundsResult:
    r"""Best-possible pointwise bounds on a set of copulas.

    Attributes
    ----------
    lower, upper : copula or NumericQuasiCopula
        The bounds (copul copula objects when they are copulas, otherwise
        :class:`~copul.theory.quasi.NumericQuasiCopula`).
    lower_is_copula, upper_is_copula : bool
        Whether the bounds are copulas.
    description : str
        The set of copulas.
    reference : str
    extra : dict
    """

    lower: Any
    upper: Any
    lower_is_copula: bool
    upper_is_copula: bool
    description: str
    reference: str
    extra: dict = field(default_factory=dict)

    def __iter__(self):
        return iter((self.lower, self.upper))

    def contains(self, C: Any, m: int = 50, tol: float = 1e-9) -> bool:
        """Whether ``lower <= C <= upper`` holds on the grid :math:`\\{k/m\\}^2`."""
        g = np.linspace(0.0, 1.0, int(m) + 1)
        U, V = np.meshgrid(g, g, indexing="ij")
        Z = cdf_function(C)(U, V)
        lo = cdf_function(self.lower)(U, V)
        hi = cdf_function(self.upper)(U, V)
        return bool(np.all(lo - tol <= Z) and np.all(hi + tol >= Z))


def bounds_given_value(a: float, b: float, theta: float) -> BoundsResult:
    r"""Best-possible bounds on the copulas with :math:`C(a,b)=\theta`.

    .. math::

       C_L(u,v) = \max\bigl(0,\,u+v-1,\,\theta-(a-u)^+-(b-v)^+\bigr)
       \le C(u,v) \le
       \min\bigl(u,\,v,\,\theta+(u-a)^++(v-b)^+\bigr) = C_U(u,v),

    for :math:`W(a,b)\le\theta\le M(a,b)` (Nelsen 2006, Thm 3.2.3).  Both
    bounds are shuffles of :math:`M` with :math:`C_L(a,b)=C_U(a,b)=\theta`,
    returned as :class:`ShuffleOfM` objects (exact measures, e.g.
    :math:`\tau(C_U)=1-4(a-\theta)(b-\theta)`).

    Examples
    --------
    >>> from copul.theory.bounds import bounds_given_value
    >>> lo, up = bounds_given_value(0.5, 0.5, 0.3)
    >>> lo.cdf(0.5, 0.5), up.cdf(0.5, 0.5)
    (0.3, 0.3)
    >>> round(up.kendalls_tau(), 12)       # 1 - 4 * 0.2 * 0.2
    0.84
    """
    a, b, theta = _check_point(a, b, theta)
    return BoundsResult(
        lower=_shuffle_lower(a, b, theta),
        upper=_shuffle_upper(a, b, theta),
        lower_is_copula=True,
        upper_is_copula=True,
        description=f"copulas with C({a:.6g}, {b:.6g}) = {theta:.6g}",
        reference="Nelsen (2006), Theorem 3.2.3",
        extra={"a": a, "b": b, "theta": theta},
    )


def blomqvist_beta_bounds(beta: float) -> BoundsResult:
    r"""Best-possible bounds on the copulas with Blomqvist's :math:`\beta(C)=\beta`.

    Since :math:`\beta = 4C(\tfrac12,\tfrac12)-1`, these are the bounds of
    :func:`bounds_given_value` at :math:`a=b=\tfrac12`,
    :math:`\theta=(1+\beta)/4` (Nelsen 2006, Thm 3.2.3).
    """
    beta = float(beta)
    if not -1.0 <= beta <= 1.0:
        raise ValueError("Blomqvist's beta lies in [-1, 1]")
    res = bounds_given_value(0.5, 0.5, (1.0 + beta) / 4.0)
    res.description = f"copulas with Blomqvist's beta = {beta:.6g}"
    res.extra["beta"] = beta
    return res


# ---------------------------------------------------------------------------
# bounds given Kendall's tau / Spearman's rho  (Nelsen et al. 2001)
# ---------------------------------------------------------------------------


def _check_t(t: float) -> float:
    t = float(t)
    if not -1.0 <= t <= 1.0:
        raise ValueError(f"the value of the measure must lie in [-1, 1], got {t}")
    return t


def _tau_lower_parts(u, v, t):
    """Value and partial derivatives of T_t^L."""
    c = 1.0 - t
    d = u - v
    R = np.sqrt(d * d + c)
    f = 0.5 * (u + v - R)
    q = np.divide(d, R, out=np.zeros_like(R), where=R > 0)
    return f, 0.5 * (1.0 - q), 0.5 * (1.0 + q)


def spearman_rho_cubic_root(a, b, theta):
    r"""The function :math:`p(a,b,\theta)` of the :math:`\rho`-bounds.

    :math:`p(a,b,\theta)` is the largest real root :math:`s` of

    .. math::

       s^3 - \tfrac14 (a-b)^2\, s - \tfrac{\theta}{12} = 0
       \qquad(\theta\ge 0),

    i.e. (Cardano) :math:`p = \tfrac16\bigl[(9\theta+\sqrt{D})^{1/3}+
    (9\theta-\sqrt{D})^{1/3}\bigr]` with :math:`D = 81\theta^2-27(a-b)^6\ge0`
    (real cube roots) and
    :math:`p = \tfrac{|a-b|}{\sqrt3}\cos\bigl(\tfrac13\arccos\tfrac{9\sqrt3\,\theta}{|a-b|^3}\bigr)`
    for :math:`D<0`.  It satisfies :math:`p\ge|a-b|/2`.  The value is polished
    by two Newton steps.
    """
    a, b, theta = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(theta, dtype=float)
    )
    d = np.abs(a - b) / 2.0
    k = np.maximum(theta, 0.0) / 12.0
    disc = k * k / 4.0 - d**6 / 27.0
    with np.errstate(all="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        s_card = np.cbrt(k / 2.0 + sq) + np.cbrt(k / 2.0 - sq)
        arg = np.clip(np.divide(3.0 * np.sqrt(3.0) * k, 2.0 * d**3), -1.0, 1.0)
        s_trig = 2.0 * d / np.sqrt(3.0) * np.cos(np.arccos(arg) / 3.0)
    s = np.where(disc >= 0.0, s_card, s_trig)
    s = np.maximum(s, d)
    for _ in range(2):
        f = s**3 - d * d * s - k
        fp = 3.0 * s * s - d * d
        step = np.divide(f, fp, out=np.zeros_like(s), where=fp > 0)
        s = np.maximum(s - step, d)
    return _out(s, a, b, theta)


def _rho_lower_parts(u, v, t):
    """Value and partial derivatives of P_t^L."""
    d = (u - v) / 2.0
    s = np.asarray(spearman_rho_cubic_root(u, v, 1.0 - t), dtype=float)
    den = 3.0 * s * s - d * d
    sp = np.divide(2.0 * d * s, den, out=np.zeros_like(s), where=den > 0)  # s'(d)
    g = (u + v) / 2.0 - s
    return g, 0.5 * (1.0 - sp), 0.5 * (1.0 + sp)


_LOWER_PARTS = {"tau": _tau_lower_parts, "rho": _rho_lower_parts}


def _lower_bound(key, u, v, t):
    """(value, d/du, d/dv) of max(0, u+v-1, f)."""
    f, fu, fv = _LOWER_PARTS[key](u, v, t)
    w = u + v - 1.0
    val = np.maximum(np.maximum(0.0, w), f)
    use_f = f >= np.maximum(0.0, w)
    use_w = ~use_f & (w > 0.0)
    du = np.where(use_f, fu, np.where(use_w, 1.0, 0.0))
    dv = np.where(use_f, fv, np.where(use_w, 1.0, 0.0))
    return val, du, dv


def _upper_bound(key, u, v, t):
    """(value, d/du, d/dv) of the upper bound u - L_{-t}(u, 1 - v)."""
    val, du, dv = _lower_bound(key, u, 1.0 - v, -t)
    return u - val, 1.0 - du, dv


def _bound_value(key, side, u, v, t):
    t = _check_t(t)
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    uc, vc = np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)
    fn = _lower_bound if side == "lower" else _upper_bound
    val = fn(key, uc, vc, t)[0]
    val = np.clip(val, np.maximum(uc + vc - 1.0, 0.0), np.minimum(uc, vc))
    return _out(val, u, v)


def kendall_tau_lower_bound(u, v, t: float):
    r"""Lower bound :math:`T_t^L` for copulas with Kendall's :math:`\tau=t` (or :math:`\ge t`).

    .. math::

       T_t^L(u,v) = \max\Bigl(0,\ u+v-1,\ \tfrac12\bigl[(u+v)-\sqrt{(u-v)^2+1-t}\bigr]\Bigr)

    (Nelsen et al. 2001).  Since :math:`T_t^L` is nondecreasing in
    :math:`t`, it is also a lower bound for :math:`\tau(C)\ge t`.
    """
    return _bound_value("tau", "lower", u, v, t)


def kendall_tau_upper_bound(u, v, t: float):
    r"""Upper bound :math:`T_t^U` for copulas with Kendall's :math:`\tau=t` (or :math:`\le t`).

    .. math::

       T_t^U(u,v) = \min\Bigl(u,\ v,\ \tfrac12\bigl[(u+v-1)+\sqrt{(u+v-1)^2+1+t}\bigr]\Bigr)

    (Nelsen et al. 2001); :math:`T_t^U(u,v) = u - T_{-t}^L(u,1-v)`.
    :math:`T_t^U = M` for :math:`t\ge0`.
    """
    return _bound_value("tau", "upper", u, v, t)


def spearman_rho_lower_bound(u, v, t: float):
    r"""Lower bound :math:`P_t^L` for copulas with Spearman's :math:`\rho=t` (or :math:`\ge t`).

    .. math::

       P_t^L(u,v) = \max\Bigl(0,\ u+v-1,\ \tfrac{u+v}{2} - p(u,v,1-t)\Bigr)

    with :math:`p` from :func:`spearman_rho_cubic_root` (Nelsen et al. 2001).
    """
    return _bound_value("rho", "lower", u, v, t)


def spearman_rho_upper_bound(u, v, t: float):
    r"""Upper bound :math:`P_t^U` for copulas with Spearman's :math:`\rho=t` (or :math:`\le t`).

    .. math::

       P_t^U(u,v) = \min\Bigl(u,\ v,\ \tfrac{u+v-1}{2} + p(1-u,v,1+t)\Bigr)

    (Nelsen et al. 2001); :math:`P_t^U(u,v) = u - P_{-t}^L(u,1-v)`.
    """
    return _bound_value("rho", "upper", u, v, t)


class MeasureBoundCopula(InversionSamplingCopula):
    r"""The pointwise bounds :math:`T_t^{L/U}` / :math:`P_t^{L/U}` as copula objects.

    Each bound is a copula: the smooth part :math:`f` of
    :math:`\max(0,u+v-1,f)` has :math:`\partial_{12}f\ge0`, and the
    singular mass on the interfaces :math:`\{f=0\}`, :math:`\{f=u+v-1\}`
    is nonnegative; the test suite checks 2-increasingness on fine grids.
    Partial derivatives are analytic; sampling uses conditional inversion.

    Parameters
    ----------
    key : {"tau", "rho"}
    t : float
        Value of the measure in :math:`[-1, 1]`.
    side : {"lower", "upper"}
    """

    def __init__(self, key: str, t: float, side: str = "lower"):
        if key not in _LOWER_PARTS:
            raise ValueError("key must be 'tau' or 'rho'")
        if side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'")
        self.key, self.t, self.side = key, _check_t(t), side
        super().__init__()

    def __repr__(self) -> str:
        sym = {"tau": "T", "rho": "P"}[self.key]
        sup = "L" if self.side == "lower" else "U"
        return f"{sym}^{sup}_{{{self.t:.6g}}}"

    __str__ = __repr__

    def _parts(self, u, v):
        fn = _lower_bound if self.side == "lower" else _upper_bound
        return fn(self.key, u, v, self.t)

    def _cdf(self, u, v):
        return self._parts(u, v)[0]

    def _h1(self, u, v):
        return self._parts(u, v)[1]

    def _h2(self, u, v):
        return self._parts(u, v)[2]


def _measure_bounds(key: str, t: float) -> BoundsResult:
    t = _check_t(t)
    name = {"tau": "Kendall's tau", "rho": "Spearman's rho"}[key]
    return BoundsResult(
        lower=MeasureBoundCopula(key, t, "lower"),
        upper=MeasureBoundCopula(key, t, "upper"),
        lower_is_copula=True,
        upper_is_copula=True,
        description=f"copulas with {name} = {t:.6g}",
        reference="Nelsen, Quesada-Molina, Rodríguez-Lallena & Úbeda-Flores (2001)",
        extra={"key": key, "value": t},
    )


def kendall_tau_bounds(t: float) -> BoundsResult:
    r"""Best-possible bounds :math:`T_t^L\le C\le T_t^U` given Kendall's :math:`\tau(C)=t`.

    See :func:`kendall_tau_lower_bound` and :func:`kendall_tau_upper_bound`
    (Nelsen et al. 2001).  Both bounds are copulas.

    Examples
    --------
    >>> from copul.theory.bounds import kendall_tau_bounds
    >>> lo, up = kendall_tau_bounds(0.5)
    >>> round(lo.cdf(0.5, 0.5), 12)          # (1 - sqrt(0.5)) / 2
    0.146446609407
    """
    return _measure_bounds("tau", t)


def spearman_rho_bounds(t: float) -> BoundsResult:
    r"""Best-possible bounds :math:`P_t^L\le C\le P_t^U` given Spearman's :math:`\rho(C)=t`.

    See :func:`spearman_rho_lower_bound` and :func:`spearman_rho_upper_bound`
    (Nelsen et al. 2001).  Both bounds are copulas.
    """
    return _measure_bounds("rho", t)


# ---------------------------------------------------------------------------
# bounds given the diagonal section
# ---------------------------------------------------------------------------


def diagonal_upper_bound(delta: Any, check: bool = True, grid: int = 4096) -> NumericQuasiCopula:
    r"""Best-possible upper bound :math:`A_\delta` on the copulas with diagonal :math:`\delta`.

    .. math::

       A_\delta(u,v) = \min\Bigl(u,\ v,\ \max(u,v) - \max_{t\in[u\wedge v,\,u\vee v]}
       \bigl(t-\delta(t)\bigr)\Bigr)

    (Nelsen, Quesada-Molina, Rodríguez-Lallena & Úbeda-Flores 2004).  That
    it is an upper bound is elementary: for :math:`u\le t\le v`,
    :math:`C(u,v)\le C(u,t)+(v-t)\le\delta(t)+v-t`.  :math:`A_\delta` is a
    quasi-copula, in general not a copula (e.g. for :math:`\delta(t)=t^2`,
    see the tests); among *symmetric* copulas the bound is the copula
    :math:`K_\delta` (:class:`~copul.theory.diagonal.DiagonalCopula`).

    Returns
    -------
    NumericQuasiCopula
    """
    d = as_diagonal(delta)
    if check and not check_diagonal(d).is_diagonal:
        raise ValueError(f"{d!r} is not a diagonal")
    neg = IntervalMin(lambda t: -d.hat(t), n=grid)

    def f(u, v):
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        return np.minimum(lo, hi + neg(lo, hi))

    return NumericQuasiCopula(f, name=f"A_delta({d!r})")


def bounds_given_diagonal(
    delta: Any, symmetric: bool = False, check: bool = True, m: int = 120
) -> BoundsResult:
    r"""Best-possible bounds on the (symmetric) copulas with diagonal section :math:`\delta`.

    * lower bound (all copulas, also symmetric ones): the Bertino copula
      :math:`B_\delta` (Fredricks & Nelsen 1997), a copula;
    * upper bound for ``symmetric=True``: the diagonal copula
      :math:`K_\delta(u,v)=\min(u,v,(\delta(u)+\delta(v))/2)` (Fredricks &
      Nelsen 1997), a copula;
    * upper bound for ``symmetric=False``: :math:`A_\delta` (Nelsen et al.
      2004, :func:`diagonal_upper_bound`), a quasi-copula; whether it is a
      copula is decided numerically on an ``m``-grid (it is in general not).

    All three bounds have diagonal section :math:`\delta`, so they are
    attained everywhere on the diagonal.

    Parameters
    ----------
    delta : Diagonal, callable or copula
        A diagonal (a copula is replaced by its diagonal section).
    symmetric : bool
        Restrict to symmetric copulas.
    check : bool
        Validate the diagonal axioms.
    m : int
        Grid for the numerical copula check of :math:`A_\delta`.
    """
    d = as_diagonal(delta)
    if check and not check_diagonal(d).is_diagonal:
        raise ValueError(f"{d!r} is not a diagonal")
    lower = BertinoCopula(d, check=False)
    if symmetric:
        upper, up_cop = DiagonalCopula(d, check=False), True
        desc, ref = (
            f"symmetric copulas with diagonal {d!r}",
            "Fredricks & Nelsen (1997)",
        )
    else:
        upper = diagonal_upper_bound(d, check=False)
        up_cop = bool(is_copula(upper, m=m))
        desc, ref = (
            f"copulas with diagonal {d!r}",
            "Fredricks & Nelsen (1997); Nelsen, Quesada-Molina, Rodríguez-Lallena & "
            "Úbeda-Flores (2004)",
        )
    return BoundsResult(lower, upper, True, up_cop, desc, ref, extra={"delta": d})


# ---------------------------------------------------------------------------
# front end
# ---------------------------------------------------------------------------

_SUPPORTED = ("tau", "rho", "beta")


def bounds_given_measure(key: str, value: float) -> BoundsResult:
    r"""Best-possible pointwise bounds on the copulas with a given measure value.

    Parameters
    ----------
    key : str
        ``"tau"`` (Kendall), ``"rho"`` (Spearman) or ``"beta"`` (Blomqvist);
        aliases of :func:`copul.measures.resolve_key` are accepted.
    value : float
        Value of the measure.

    Returns
    -------
    BoundsResult
        ``(lower, upper)`` copula objects (iterable), whether each is a
        copula (all three cases: yes), description and reference.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.bounds import bounds_given_measure
    >>> res = bounds_given_measure("rho", 0.4)
    >>> res.lower_is_copula, res.upper_is_copula
    (True, True)
    >>> res.contains(cp.Frank.from_measure("rho", 0.4), m=20)
    True
    """
    from copul.measures.registry import resolve_key

    try:
        k = resolve_key(key)
    except Exception:
        k = str(key).lower()
    if k == "tau":
        return kendall_tau_bounds(value)
    if k == "rho":
        return spearman_rho_bounds(value)
    if k == "beta":
        return blomqvist_beta_bounds(value)
    raise NotImplementedError(
        f"best-possible bounds given {key!r} are not implemented; supported: {_SUPPORTED}"
    )
