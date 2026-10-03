r"""
Quasi-copulas: characterization, 2-increasing defect and lattice operations.

A (bivariate) **quasi-copula** is a function :math:`Q:[0,1]^2\to[0,1]` with

(Q1) the boundary conditions :math:`Q(u,0)=Q(0,v)=0`, :math:`Q(u,1)=u`,
     :math:`Q(1,v)=v`;

(Q2) :math:`Q` is nondecreasing in each argument;

(Q3) :math:`Q` is 1-Lipschitz in each argument,

     .. math::

        |Q(u_2,v)-Q(u_1,v)|\le|u_2-u_1|,\qquad |Q(u,v_2)-Q(u,v_1)|\le|v_2-v_1| .

Quasi-copulas were introduced by Alsina, Nelsen & Schweizer (1993) through a
"track" condition; the characterization (Q1)--(Q3) is due to Genest,
Quesada-Molina, Rodríguez-Lallena & Sempi (1999).  Standard facts used in
this module (Nelsen 2006, Sect. 6.2):

* every copula is a quasi-copula, and a quasi-copula is a copula iff it is
  2-increasing, i.e. every rectangle :math:`R=[u_1,u_2]\times[v_1,v_2]` has
  nonnegative :math:`Q`-volume

  .. math::

     V_Q(R) = Q(u_2,v_2) - Q(u_1,v_2) - Q(u_2,v_1) + Q(u_1,v_1);

* :math:`W\le Q\le M` for every quasi-copula (immediate from (Q1)--(Q3));
* :math:`-\tfrac13\le V_Q(R)\le 1` for every quasi-copula and rectangle
  :math:`R`, and :math:`V_Q(R)=-\tfrac13` forces
  :math:`R=[\tfrac13,\tfrac23]^2` (Nelsen, Quesada-Molina, Rodríguez-Lallena &
  Úbeda-Flores 2002);
* the pointwise supremum and infimum of any nonempty set of quasi-copulas
  (in particular of copulas) are quasi-copulas (Nelsen, Quesada-Molina,
  Rodríguez-Lallena & Úbeda-Flores 2004).  They are in general *not*
  copulas: for the shuffle of :math:`M`
  :math:`C^*(u,v) = \max\{0,\min(u, v-\tfrac13)\} + \max\{0, \min(u-\tfrac23, v)\}`
  the quasi-copula :math:`Q=\max(C^*, C^{*\top})` has
  :math:`V_Q([\tfrac13,\tfrac23]^2) = -\tfrac13`.

Functions
---------
``check_quasi_copula`` / ``is_quasi_copula``
    grid checks of (Q1)--(Q3) (and of :math:`W\le Q\le M`);
``two_increasing_defect`` / ``is_copula``
    the most negative rectangle volume (with its location) on a grid;
``quasi_copula_volume``
    :math:`V_Q(R)` for one or many rectangles;
``copula_max`` / ``copula_min``
    pointwise supremum / infimum of finitely many (quasi-)copulas;
``NumericQuasiCopula``
    a light quasi-copula object (cdf, checks, plotting);
``FunctionCopula``
    a full copul copula from a vectorized cdf (optionally with
    :math:`h`-functions), sampled by numerical conditional inversion.

All functions accept copul copula objects, :class:`NumericQuasiCopula`
instances or plain vectorized callables ``f(u, v)``.

References
----------
Alsina, C., Nelsen, R. B. & Schweizer, B. (1993). On the characteristic
function of a class of binary operations on distribution functions.
*Statistics & Probability Letters* 16, 85–89.

Genest, C., Quesada-Molina, J. J., Rodríguez-Lallena, J. A. & Sempi, C.
(1999). A characterization of quasi-copulas. *Journal of Multivariate
Analysis* 69, 193–205.

Nelsen, R. B., Quesada-Molina, J. J., Rodríguez-Lallena, J. A. &
Úbeda-Flores, M. (2002). Some new properties of quasi-copulas. In Cuadras,
C. M., Fortiana, J. & Rodríguez-Lallena, J. A. (eds.), *Distributions with
Given Marginals and Statistical Modelling*, Kluwer, 187–194.

Nelsen, R. B., Quesada-Molina, J. J., Rodríguez-Lallena, J. A. &
Úbeda-Flores, M. (2004). Best-possible bounds on sets of bivariate
distribution functions. *Journal of Multivariate Analysis* 90, 348–358.

Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
Sect. 6.2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from copul.family.constructions._base import (
    NumericBivCopula,
    _finish,
    _parse_uv,
    component_callables,
)

__all__ = [
    "FunctionCopula",
    "NumericQuasiCopula",
    "QuasiCopulaCheck",
    "VolumeDefect",
    "cdf_function",
    "check_quasi_copula",
    "copula_max",
    "copula_min",
    "is_copula",
    "is_quasi_copula",
    "quasi_copula_volume",
    "two_increasing_defect",
]

#: default number of grid intervals (divisible by 2, 3, 4, 5, 6, 8, 10, 12)
DEFAULT_GRID = 120


# ---------------------------------------------------------------------------
# vectorized access to (quasi-)copulas
# ---------------------------------------------------------------------------


def cdf_function(Q: Any) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    r"""Vectorized function :math:`(u,v)\mapsto Q(u,v)` of a (quasi-)copula.

    Parameters
    ----------
    Q : copula object, NumericQuasiCopula or callable
        A fully specified copul copula (any family, checkerboard or
        construction), a :class:`NumericQuasiCopula`, or a vectorized
        callable ``f(u, v)``.

    Returns
    -------
    callable
        ``f(u, v)`` accepting broadcastable arrays and returning an
        ``ndarray`` of the broadcast shape.
    """
    if isinstance(Q, NumericQuasiCopula):
        return Q.cdf_vectorized
    if hasattr(Q, "cdf") and (hasattr(Q, "dim") or hasattr(Q, "cdf_vectorized")):
        f = component_callables(Q)[0]
    elif callable(Q):
        f = Q
    else:
        raise TypeError(f"expected a copula, quasi-copula or callable, got {type(Q).__name__}")

    def cdf(u, v):
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        with np.errstate(all="ignore"):
            out = np.asarray(f(u, v), dtype=float)
        return np.broadcast_to(out, u.shape).astype(float, copy=True)

    return cdf


def _grid(m: int) -> np.ndarray:
    m = int(m)
    if m < 2:
        raise ValueError("the grid needs at least two intervals (m >= 2)")
    return np.linspace(0.0, 1.0, m + 1)


def _grid_values(Q, m: int) -> tuple[np.ndarray, np.ndarray]:
    g = _grid(m)
    U, V = np.meshgrid(g, g, indexing="ij")
    return g, cdf_function(Q)(U, V)


# ---------------------------------------------------------------------------
# volumes
# ---------------------------------------------------------------------------


def quasi_copula_volume(Q: Any, rect) -> float | np.ndarray:
    r"""Q-volume :math:`V_Q([u_1,u_2]\times[v_1,v_2])` of rectangles.

    .. math::

       V_Q(R) = Q(u_2,v_2) - Q(u_1,v_2) - Q(u_2,v_1) + Q(u_1,v_1).

    Parameters
    ----------
    Q : copula, NumericQuasiCopula or callable
    rect : array_like
        ``(u1, u2, v1, v2)`` for one rectangle or an ``(N, 4)`` array.

    Returns
    -------
    float or numpy.ndarray
        The volume(s); a float for a single rectangle.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.quasi import quasi_copula_volume
    >>> round(quasi_copula_volume(cp.BivIndependenceCopula(), (0.2, 0.5, 0.1, 0.6)), 12)
    0.15
    """
    r = np.asarray(rect, dtype=float)
    single = r.ndim == 1
    r = np.atleast_2d(r)
    if r.shape[-1] != 4:
        raise ValueError("rect must be (u1, u2, v1, v2) or an (N, 4) array")
    u1, u2, v1, v2 = r.T
    f = cdf_function(Q)
    vol = f(u2, v2) - f(u1, v2) - f(u2, v1) + f(u1, v1)
    return float(vol[0]) if single else vol


@dataclass
class VolumeDefect:
    r"""Most negative rectangle volume of a function on a grid.

    Attributes
    ----------
    volume : float
        :math:`\min(0, \min_R V_Q(R))` over all rectangles :math:`R` with
        corners on the grid (``0.0`` if :math:`Q` is 2-increasing there).
    rectangle : tuple of float or None
        ``(u1, u2, v1, v2)`` of a rectangle attaining ``volume`` (``None`` if
        no rectangle has negative volume).
    min_cell_volume : float
        Smallest volume of a single grid cell.
    grid : int
        Number of grid intervals per axis.
    """

    volume: float
    rectangle: tuple[float, float, float, float] | None
    min_cell_volume: float
    grid: int

    def is_two_increasing(self, tol: float = 1e-10) -> bool:
        """Whether no rectangle volume is below ``-tol``."""
        return bool(self.volume >= -tol)


def _min_subrectangle(mass: np.ndarray) -> tuple[float, tuple[int, int, int, int]]:
    """Minimum-sum contiguous sub-rectangle of a matrix (O(m^2 n), vectorized)."""
    m, n = mass.shape
    best = np.inf
    loc = (0, 0, 0, 0)
    for i1 in range(m):
        S = np.cumsum(mass[i1:, :], axis=0)  # row blocks i1..i2
        P = np.concatenate([np.zeros((S.shape[0], 1)), np.cumsum(S, axis=1)], axis=1)
        run = np.maximum.accumulate(P, axis=1)
        diff = P[:, 1:] - run[:, :-1]  # best block ending at column j2
        k = int(np.argmin(diff))
        r, j2 = divmod(k, n)
        val = float(diff[r, j2])
        if val < best:
            j1 = int(np.argmax(P[r, : j2 + 1]))
            best = val
            loc = (i1, i1 + r, j1, j2)
    return best, loc


def two_increasing_defect(Q: Any, m: int = DEFAULT_GRID) -> VolumeDefect:
    r"""Most negative :math:`Q`-volume of a rectangle with corners on a grid.

    The cell volumes of :math:`Q` on the :math:`(m+1)\times(m+1)` grid
    :math:`\{0,\tfrac1m,\dots,1\}^2` are computed and the contiguous block
    of cells with the smallest total volume is found (2-D maximum-subarray
    algorithm).  Since volumes are additive, this is the most negative
    volume over *all* rectangles with corners on the grid.  :math:`Q` is
    2-increasing on the grid iff the result is :math:`0`.

    Parameters
    ----------
    Q : copula, NumericQuasiCopula or callable
    m : int
        Grid intervals per axis (default 120; multiples of 3 contain the
        points :math:`\tfrac13, \tfrac23`).

    Returns
    -------
    VolumeDefect

    Examples
    --------
    >>> from copul.theory.quasi import copula_max, two_increasing_defect
    >>> from copul.theory.symmetry import maximally_nonexchangeable_copula
    >>> C = maximally_nonexchangeable_copula()
    >>> d = two_increasing_defect(copula_max(C, lambda u, v: C.cdf_vectorized(v, u)), m=30)
    >>> round(d.volume, 12), [round(x, 12) for x in d.rectangle]
    (-0.333333333333, [0.333333333333, 0.666666666667, 0.333333333333, 0.666666666667])
    """
    g, Z = _grid_values(Q, m)
    mass = np.diff(np.diff(Z, axis=0), axis=1)
    min_cell = float(mass.min())
    if min_cell >= 0.0:
        return VolumeDefect(0.0, None, min_cell, int(m))
    val, (i1, i2, j1, j2) = _min_subrectangle(mass)
    rect = (float(g[i1]), float(g[i2 + 1]), float(g[j1]), float(g[j2 + 1]))
    return VolumeDefect(min(val, 0.0), rect, min_cell, int(m))


# ---------------------------------------------------------------------------
# quasi-copula checks
# ---------------------------------------------------------------------------


@dataclass
class QuasiCopulaCheck:
    r"""Result of :func:`check_quasi_copula` (grid check of (Q1)--(Q3)).

    Attributes
    ----------
    is_quasi_copula : bool
        All three conditions hold up to ``tol``.
    boundary_ok, increasing_ok, lipschitz_ok : bool
        Conditions (Q1), (Q2), (Q3).
    frechet_ok : bool
        :math:`W\le Q\le M` on the grid (implied by (Q1)--(Q3)).
    max_boundary_error : float
        Largest deviation from the boundary conditions.
    max_decrease : float
        Largest decrease between neighbouring grid points (``0`` if
        nondecreasing).
    max_lipschitz_excess : float
        Largest excess of a difference quotient over 1, times the mesh.
    grid : int
    details : dict
        Location of the worst violations.
    """

    is_quasi_copula: bool
    boundary_ok: bool
    increasing_ok: bool
    lipschitz_ok: bool
    frechet_ok: bool
    max_boundary_error: float
    max_decrease: float
    max_lipschitz_excess: float
    grid: int
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.is_quasi_copula


def check_quasi_copula(Q: Any, m: int = DEFAULT_GRID, tol: float = 1e-9) -> QuasiCopulaCheck:
    r"""Check the quasi-copula axioms (Q1)--(Q3) on a grid.

    Parameters
    ----------
    Q : copula, NumericQuasiCopula or callable
    m : int
        Grid intervals per axis.
    tol : float
        Absolute tolerance.

    Returns
    -------
    QuasiCopulaCheck

    Notes
    -----
    (Q2) and (Q3) are checked between neighbouring grid points, i.e. every
    difference :math:`\Delta = Q(u_{i+1},v_j)-Q(u_i,v_j)` (and the same in
    :math:`v`) must satisfy :math:`-\mathrm{tol}\le\Delta\le h+\mathrm{tol}`
    with mesh :math:`h=1/m` (Genest et al. 1999).
    """
    g, Z = _grid_values(Q, m)
    h = 1.0 / int(m)
    bnd = np.concatenate(
        [np.abs(Z[0, :]), np.abs(Z[:, 0]), np.abs(Z[-1, :] - g), np.abs(Z[:, -1] - g)]
    )
    max_bnd = float(bnd.max())
    du = np.diff(Z, axis=0)
    dv = np.diff(Z, axis=1)
    dec = float(max(0.0, -du.min(), -dv.min()))
    lip = float(max(0.0, du.max() - h, dv.max() - h))
    U, V = np.meshgrid(g, g, indexing="ij")
    W = np.maximum(U + V - 1.0, 0.0)
    M = np.minimum(U, V)
    frechet = bool(np.all(W - tol <= Z) and np.all(M + tol >= Z))
    details = {}
    if dec > tol:
        k = np.unravel_index(np.argmin(du), du.shape) if -du.min() >= -dv.min() else None
        if k is not None:
            details["decrease_at"] = ("u", float(g[k[0]]), float(g[k[0] + 1]), float(g[k[1]]))
        else:
            k = np.unravel_index(np.argmin(dv), dv.shape)
            details["decrease_at"] = ("v", float(g[k[0]]), float(g[k[1]]), float(g[k[1] + 1]))
    if lip > tol:
        if du.max() >= dv.max():
            k = np.unravel_index(np.argmax(du), du.shape)
            details["lipschitz_at"] = ("u", float(g[k[0]]), float(g[k[0] + 1]), float(g[k[1]]))
        else:
            k = np.unravel_index(np.argmax(dv), dv.shape)
            details["lipschitz_at"] = ("v", float(g[k[0]]), float(g[k[1]]), float(g[k[1] + 1]))
    b_ok, i_ok, l_ok = max_bnd <= tol, dec <= tol, lip <= tol
    return QuasiCopulaCheck(
        is_quasi_copula=bool(b_ok and i_ok and l_ok),
        boundary_ok=bool(b_ok),
        increasing_ok=bool(i_ok),
        lipschitz_ok=bool(l_ok),
        frechet_ok=frechet,
        max_boundary_error=max_bnd,
        max_decrease=dec,
        max_lipschitz_excess=lip,
        grid=int(m),
        details=details,
    )


def is_quasi_copula(Q: Any, m: int = DEFAULT_GRID, tol: float = 1e-9) -> bool:
    """Whether ``Q`` satisfies the quasi-copula axioms (Q1)--(Q3) on a grid.

    See :func:`check_quasi_copula`.
    """
    return check_quasi_copula(Q, m=m, tol=tol).is_quasi_copula


def is_copula(Q: Any, m: int = DEFAULT_GRID, tol: float = 1e-9) -> bool:
    r"""Whether ``Q`` is a copula on a grid: a quasi-copula that is 2-increasing.

    Parameters
    ----------
    Q : copula, NumericQuasiCopula or callable
    m : int
        Grid intervals per axis.
    tol : float
        Tolerance for the axioms and for negative rectangle volumes.
    """
    if not is_quasi_copula(Q, m=m, tol=tol):
        return False
    return two_increasing_defect(Q, m=m).is_two_increasing(tol)


# ---------------------------------------------------------------------------
# NumericQuasiCopula
# ---------------------------------------------------------------------------


class NumericQuasiCopula:
    r"""A quasi-copula given by a vectorized function :math:`Q(u,v)`.

    Light-weight object for quasi-copulas that need not be copulas (no
    densities, conditional distributions or sampling): evaluation, the
    axiom checks of this module, volumes and plots.

    Parameters
    ----------
    func : callable
        Vectorized ``func(u, v)``.
    name : str, optional
        Name used in ``repr``.

    Examples
    --------
    >>> import copul as cp
    >>> from copul.theory.quasi import copula_max
    >>> Q = copula_max(cp.Clayton(2), cp.Frank(3))
    >>> Q.is_quasi_copula(m=30)
    True
    """

    def __init__(self, func: Callable, name: str | None = None):
        self._func = cdf_function(func) if not isinstance(func, NumericQuasiCopula) else func._func
        self.name = name or "NumericQuasiCopula"

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__

    # -- evaluation ---------------------------------------------------------
    def cdf_vectorized(self, u, v) -> np.ndarray:
        """Vectorized :math:`Q(u,v)` (arguments clipped to :math:`[0,1]`)."""
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        return self._func(np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0))

    def cdf(self, *args, **kwargs):
        """:math:`Q(u,v)`; scalars give a ``float``, arrays an ``ndarray``."""
        u, v = _parse_uv(args, kwargs, "cdf")
        return _finish(self.cdf_vectorized(u, v), u, v)

    def diagonal(self, t):
        r"""Diagonal section :math:`Q(t,t)`."""
        t = np.asarray(t, dtype=float)
        return _finish(self.cdf_vectorized(t, t), t, t)

    def volume(self, rect):
        """:math:`Q`-volume of rectangle(s), see :func:`quasi_copula_volume`."""
        return quasi_copula_volume(self, rect)

    # -- checks -----------------------------------------------------------------
    def check(self, m: int = DEFAULT_GRID, tol: float = 1e-9) -> QuasiCopulaCheck:
        """Grid check of the quasi-copula axioms, see :func:`check_quasi_copula`."""
        return check_quasi_copula(self, m=m, tol=tol)

    def is_quasi_copula(self, m: int = DEFAULT_GRID, tol: float = 1e-9) -> bool:
        """See :func:`is_quasi_copula`."""
        return is_quasi_copula(self, m=m, tol=tol)

    def is_copula(self, m: int = DEFAULT_GRID, tol: float = 1e-9) -> bool:
        """See :func:`is_copula`."""
        return is_copula(self, m=m, tol=tol)

    def two_increasing_defect(self, m: int = DEFAULT_GRID) -> VolumeDefect:
        """See :func:`two_increasing_defect`."""
        return two_increasing_defect(self, m=m)

    def to_copula(self, check: bool = True, m: int = DEFAULT_GRID, tol: float = 1e-9):
        """Turn a 2-increasing quasi-copula into a :class:`FunctionCopula`.

        Parameters
        ----------
        check : bool
            Verify the copula axioms on a grid first (raise ``ValueError``
            if they fail).
        """
        if check and not self.is_copula(m=m, tol=tol):
            raise ValueError(f"{self.name} is not a copula (not 2-increasing on the grid).")
        return FunctionCopula(self.cdf_vectorized, name=self.name)

    # -- plotting -------------------------------------------------------------
    def plot(self, kind: str = "contour", m: int = 60, ax=None, **kwargs):
        """Plot :math:`Q` (``kind="contour"`` or ``"surface"``) or its cell
        volumes (``kind="mass"``; negative mass in blue).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        g, Z = _grid_values(self, m)
        if kind == "mass":
            mass = np.diff(np.diff(Z, axis=0), axis=1) * m * m
            if ax is None:
                _, ax = plt.subplots()
            lim = float(np.max(np.abs(mass))) or 1.0
            im = ax.imshow(
                mass.T,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=kwargs.pop("cmap", "RdBu_r"),
                vmin=-lim,
                vmax=lim,
                **kwargs,
            )
            plt.colorbar(im, ax=ax, label="cell volume x m^2")
        elif kind == "surface":
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
            U, V = np.meshgrid(g, g, indexing="ij")
            ax.plot_surface(U, V, Z, cmap=kwargs.pop("cmap", "viridis"), **kwargs)
            ax.set_zlabel("Q(u, v)")
        elif kind == "contour":
            if ax is None:
                _, ax = plt.subplots()
            U, V = np.meshgrid(g, g, indexing="ij")
            cs = ax.contour(U, V, Z, levels=kwargs.pop("levels", 10), **kwargs)
            ax.clabel(cs, inline=True, fontsize=8)
            ax.set_aspect("equal")
        else:
            raise ValueError("kind must be 'contour', 'surface' or 'mass'")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(self.name)
        return ax


# ---------------------------------------------------------------------------
# lattice operations
# ---------------------------------------------------------------------------


def _names(Qs: Sequence) -> str:
    return ", ".join(getattr(Q, "name", None) or repr(Q) for Q in Qs)


def copula_max(*Qs: Any) -> NumericQuasiCopula:
    r"""Pointwise maximum :math:`\max_i Q_i(u,v)` of (quasi-)copulas.

    The pointwise supremum of any nonempty set of quasi-copulas is a
    quasi-copula (Nelsen et al. 2004); it is in general not a copula.

    Parameters
    ----------
    *Qs : copula, NumericQuasiCopula or callable
        At least one (quasi-)copula.

    Returns
    -------
    NumericQuasiCopula
    """
    if not Qs:
        raise ValueError("copula_max needs at least one argument")
    fs = [cdf_function(Q) for Q in Qs]

    def f(u, v):
        out = fs[0](u, v)
        for g in fs[1:]:
            out = np.maximum(out, g(u, v))
        return out

    return NumericQuasiCopula(f, name=f"copula_max({_names(Qs)})")


def copula_min(*Qs: Any) -> NumericQuasiCopula:
    r"""Pointwise minimum :math:`\min_i Q_i(u,v)` of (quasi-)copulas.

    The pointwise infimum of any nonempty set of quasi-copulas is a
    quasi-copula (Nelsen et al. 2004); it is in general not a copula.

    Parameters
    ----------
    *Qs : copula, NumericQuasiCopula or callable
        At least one (quasi-)copula.

    Returns
    -------
    NumericQuasiCopula
    """
    if not Qs:
        raise ValueError("copula_min needs at least one argument")
    fs = [cdf_function(Q) for Q in Qs]

    def f(u, v):
        out = fs[0](u, v)
        for g in fs[1:]:
            out = np.minimum(out, g(u, v))
        return out

    return NumericQuasiCopula(f, name=f"copula_min({_names(Qs)})")


# ---------------------------------------------------------------------------
# copulas from a cdf (numerical conditional inversion)
# ---------------------------------------------------------------------------


class InversionSamplingCopula(NumericBivCopula):
    r""":class:`~copul.family.constructions.NumericBivCopula` sampled by
    conditional inversion.

    Draws :math:`U, W` i.i.d. uniform and returns
    :math:`(U, V)` with :math:`V = \inf\{v : \partial_1 C(U,v)\ge W\}`
    (52 vectorized bisection steps on the :math:`h`-function).  Exact up
    to the accuracy of :math:`\partial_1 C` (finite differences of the cdf
    unless a subclass provides it analytically).
    """

    _bisection_steps = 52

    def _rvs(self, n, rng):
        u = rng.random(n)
        w = rng.random(n)
        lo = np.zeros(n)
        hi = np.ones(n)
        for _ in range(self._bisection_steps):
            mid = 0.5 * (lo + hi)
            ge = self._h1_clean(u, mid) >= w
            hi = np.where(ge, mid, hi)
            lo = np.where(ge, lo, mid)
        return np.column_stack([u, hi])


class FunctionCopula(InversionSamplingCopula):
    r"""Copula given by a vectorized cdf (and optionally its partial derivatives).

    The caller is responsible for ``cdf`` being a copula; use
    :func:`is_copula` to check it numerically.  Without ``h1``/``h2`` the
    conditional distributions are central finite differences of the cdf;
    sampling uses numerical conditional inversion.

    Parameters
    ----------
    cdf : callable
        Vectorized :math:`C(u,v)`.
    h1, h2 : callable, optional
        :math:`\partial_1 C` and :math:`\partial_2 C`.
    pdf : callable, optional
        Density (only used with ``absolutely_continuous=True``).
    name : str, optional
    absolutely_continuous : bool
        Whether the copula has a density.

    Examples
    --------
    >>> from copul.theory.quasi import FunctionCopula
    >>> C = FunctionCopula(lambda u, v: u * v, name="Pi")
    >>> round(C.spearmans_rho(), 8)
    0.0
    """

    def __init__(
        self,
        cdf: Callable,
        h1: Callable | None = None,
        h2: Callable | None = None,
        pdf: Callable | None = None,
        *,
        name: str | None = None,
        absolutely_continuous: bool = False,
    ):
        self._cdf_func = cdf
        self._h1_func = h1
        self._h2_func = h2
        self._pdf_func = pdf
        self._ac = bool(absolutely_continuous and pdf is not None)
        self.name = name or "FunctionCopula"
        super().__init__()

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__

    @property
    def is_absolutely_continuous(self) -> bool:
        return self._ac

    def _cdf(self, u, v):
        return self._cdf_func(u, v)

    def _h1(self, u, v):
        if self._h1_func is None:
            return super()._h1(u, v)
        return self._h1_func(u, v)

    def _h2(self, u, v):
        if self._h2_func is None:
            return super()._h2(u, v)
        return self._h2_func(u, v)

    def _pdf(self, u, v):
        if self._pdf_func is None:
            return super()._pdf(u, v)
        return self._pdf_func(u, v)
