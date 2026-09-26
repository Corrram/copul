r"""
The :class:`ExactRegion` abstraction.

An exact region is the set

.. math::

   \mathcal R_{x,y}=\{(\kappa_x(C),\kappa_y(C)) : C \text{ a bivariate copula (of a class)}\}

described by its range :math:`[x_{\min},x_{\max}]` of the first measure and
two boundary functions,

.. math::

   \mathcal R_{x,y}=\{(x,y): x_{\min}\le x\le x_{\max},\;
   \ell(x)\le y\le u(x)\}.

All registered regions are closed and *vertically convex*; they are also
horizontally convex, which allows :meth:`ExactRegion.swap` to describe the
same set with the roles of the axes exchanged (boundaries are then inverted
numerically to machine precision).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from copul.regions.measures import MEASURES, resolve, symbol

__all__ = ["ExactRegion", "KeyPoint", "SwappedRegion"]


@dataclass(frozen=True)
class KeyPoint:
    """A distinguished point of a region, e.g. the value of :math:`M`.

    Attributes
    ----------
    label : str
        LaTeX label (without dollar signs), e.g. ``"M"`` or ``r"C_1"``.
    x, y : float
        Coordinates.
    copula : callable, optional
        Zero-argument factory returning a copula attaining the point.
    """

    label: str
    x: float
    y: float
    copula: Callable[[], Any] | None = None

    def swap(self) -> KeyPoint:
        return KeyPoint(self.label, self.y, self.x, self.copula)


BoundaryFn = Callable[[np.ndarray], np.ndarray]
CopulaFactory = Callable[[float], Any]


class ExactRegion:
    r"""An exact attainable region of two dependence measures.

    Parameters
    ----------
    x, y : str
        Measure keys of the horizontal and vertical axis.
    lower, upper : callable
        Vectorised boundary functions :math:`\ell, u` on ``x_range``.
    x_range : pair of float
        Range of the first measure over the region.
    key_points : sequence of KeyPoint
        Distinguished points (:math:`M`, :math:`W`, :math:`\Pi`, ...).
    reference : str
        Citation of the result.
    source : str
        File(s) in this repository from which the formulas were taken.
    boundary_family : dict, optional
        ``{"upper": f, "lower": g}`` with ``f(x)`` returning a copula attaining
        the respective boundary at ``x`` (or ``None`` where no family is
        implemented).
    copula_class : str
        ``"all"`` (all bivariate copulas) or a subclass such as ``"si"``.
    status : str
        ``"published"``, ``"preprint"``, ``"package"`` (formula shipped in the
        package, publication status unknown) ...
    notes : str
        Free-form remarks.
    """

    def __init__(
        self,
        x: str,
        y: str,
        lower: BoundaryFn,
        upper: BoundaryFn,
        x_range: tuple[float, float],
        key_points=(),
        reference: str = "",
        source: str = "",
        boundary_family: dict[str, CopulaFactory] | None = None,
        copula_class: str = "all",
        status: str = "published",
        notes: str = "",
    ) -> None:
        self.x = resolve(x)
        self.y = resolve(y)
        self._lower = lower
        self._upper = upper
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.key_points = tuple(key_points)
        self.reference = reference
        self.source = source
        self._family = dict(boundary_family or {})
        self.copula_class = copula_class
        self.status = status
        self.notes = notes

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        cls = "" if self.copula_class == "all" else f", class={self.copula_class!r}"
        return f"ExactRegion(x={self.x!r}, y={self.y!r}{cls})"

    @property
    def key(self) -> tuple[str, str, str]:
        """Registry key ``(x, y, copula_class)``."""
        return (self.x, self.y, self.copula_class)

    @property
    def title(self) -> str:
        cls = "" if self.copula_class == "all" else f" ({self.copula_class.upper()})"
        return f"$({symbol(self.x)},{symbol(self.y)})${cls}"

    # ------------------------------------------------------------------
    def _eval(self, fn: BoundaryFn, x) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        out = np.asarray(fn(np.clip(xa, *self.x_range)), dtype=float)
        if out.size == xa.size:
            out = out.reshape(xa.shape).copy()
        else:
            out = np.broadcast_to(out, xa.shape).copy()
        bad = (xa < self.x_range[0]) | (xa > self.x_range[1])
        if np.any(bad):
            out[bad] = np.nan
        return out if out.ndim else out[()]

    def lower(self, x) -> np.ndarray:
        """Lower boundary :math:`\\ell(x)` (``nan`` outside :attr:`x_range`)."""
        return self._eval(self._lower, x)

    def upper(self, x) -> np.ndarray:
        """Upper boundary :math:`u(x)` (``nan`` outside :attr:`x_range`)."""
        return self._eval(self._upper, x)

    def margin(self, x, y) -> np.ndarray:
        r"""Signed violation: :math:`\max(\ell(x)-y,\;y-u(x),\;\text{range excess})`.

        Non-positive values mean ``(x, y)`` lies in the region; the value is a
        vertical distance to the boundary for points above/below it.
        """
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        xc = np.clip(xa, *self.x_range)
        lo = np.asarray(self._lower(xc), dtype=float)
        up = np.asarray(self._upper(xc), dtype=float)
        rng = np.maximum(self.x_range[0] - xa, xa - self.x_range[1])
        return np.maximum(np.maximum(lo - ya, ya - up), rng)

    def contains(self, x, y, tol: float = 1e-9):
        """Whether ``(x, y)`` lies in the region up to ``tol`` (vectorised)."""
        res = self.margin(x, y) <= tol
        return bool(res) if np.ndim(res) == 0 else res

    # ------------------------------------------------------------------
    def boundary_copula(self, x: float, side: str = "upper"):
        """A copula attaining the ``side`` boundary at ``x``.

        Returns
        -------
        copula or None
            ``None`` if no boundary family is implemented for this part of the
            boundary.

        Raises
        ------
        ValueError
            If ``x`` is outside :attr:`x_range` or ``side`` is invalid.
        """
        if side not in ("upper", "lower"):
            raise ValueError("side must be 'upper' or 'lower'")
        x = float(x)
        if not (self.x_range[0] - 1e-12 <= x <= self.x_range[1] + 1e-12):
            raise ValueError(f"x={x} outside the range {self.x_range}")
        fam = self._family.get(side)
        return None if fam is None else fam(min(max(x, self.x_range[0]), self.x_range[1]))

    @property
    def has_boundary_family(self) -> dict[str, bool]:
        return {s: s in self._family for s in ("upper", "lower")}

    # ------------------------------------------------------------------
    def sample_boundary(self, n: int = 400) -> np.ndarray:
        """Points on the boundary as a closed polygon (counter-clockwise).

        The lower boundary is traversed from left to right, then the upper
        one from right to left; the first point is repeated at the end.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2 n + 1, 2)``.
        """
        xs = np.linspace(*self.x_range, n)
        lo = np.column_stack([xs, self.lower(xs)])
        up = np.column_stack([xs[::-1], self.upper(xs[::-1])])
        return np.vstack([lo, up, lo[:1]])

    def swap(self) -> SwappedRegion:
        """The same region with the axes exchanged."""
        return SwappedRegion(self)

    # ------------------------------------------------------------------
    def plot(
        self,
        ax=None,
        fill: bool = True,
        mark: bool = True,
        n: int = 801,
        color: str | None = None,
        fill_color: str | None = None,
        label: str | None = None,
        paper_axes: bool = True,
        **style,
    ):
        """Plot the region.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes (a new figure is created otherwise).
        fill : bool
            Shade the region.
        mark : bool
            Mark :attr:`key_points`.
        n : int
            Number of evaluation points of the boundary.
        color, fill_color : str, optional
            Boundary and fill colour (defaults from :mod:`copul.regions.style`).
        label : str, optional
            Legend label of the boundary.
        paper_axes : bool
            Apply :func:`copul.regions.style.apply_paper_axes`.
        **style
            Passed to :meth:`~matplotlib.axes.Axes.plot` for the boundary.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        from copul.regions.style import BLUE, FILL, apply_paper_axes, mark_points

        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 5.5))
        color = color or BLUE
        fill_color = fill_color or FILL
        if paper_axes:
            yr = MEASURES[self.y].range
            apply_paper_axes(ax, self.x, self.y, xlim=self.x_range, ylim=yr)
        xs = np.linspace(*self.x_range, n)
        lo, up = self.lower(xs), self.upper(xs)
        if fill:
            ax.fill_between(xs, lo, up, color=fill_color, lw=0, zorder=0)
        kw = {"lw": 2.0, "color": color}
        kw.update(style)
        poly = self.sample_boundary(n)
        ax.plot(poly[:, 0], poly[:, 1], label=label, zorder=3, **kw)
        if mark and self.key_points:
            mark_points(ax, self.key_points)
        return ax


class SwappedRegion(ExactRegion):
    """An :class:`ExactRegion` with exchanged axes.

    :meth:`contains` and :meth:`margin` are evaluated on the original region
    (exact); the boundary functions are obtained by vectorised bisection on
    the original boundaries (assuming horizontal convexity, which holds for
    all registered regions).
    """

    def __init__(self, base: ExactRegion, n_grid: int = 4001) -> None:
        self.base = base
        self._grid = np.linspace(*base.x_range, n_grid)
        self._glo = np.asarray(base.lower(self._grid), dtype=float)
        self._gup = np.asarray(base.upper(self._grid), dtype=float)
        yr = (float(np.nanmin(self._glo)), float(np.nanmax(self._gup)))
        super().__init__(
            base.y,
            base.x,
            self._inv_lower,
            self._inv_upper,
            yr,
            key_points=[p.swap() for p in base.key_points],
            reference=base.reference,
            source=base.source,
            copula_class=base.copula_class,
            status=base.status,
            notes=base.notes,
        )

    def __repr__(self) -> str:
        return f"SwappedRegion(x={self.x!r}, y={self.y!r}, base={self.base!r})"

    def margin(self, x, y):
        return self.base.margin(y, x)

    def boundary_copula(self, x: float, side: str = "upper"):
        """Boundary copulas of swapped regions are not tracked; always ``None``."""
        if side not in ("upper", "lower"):
            raise ValueError("side must be 'upper' or 'lower'")

    def _inside(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return self.base.margin(p, q) <= 0.0

    def _invert(self, q, which: str) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, dtype=float))
        g = self._grid
        inside = (self._glo[None, :] <= q[:, None] + 1e-15) & (
            q[:, None] <= self._gup[None, :] + 1e-15
        )
        any_in = inside.any(axis=1)
        idx_first = np.argmax(inside, axis=1)
        idx_last = inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1)
        out = np.full(q.shape, np.nan)
        if which == "lower":
            a = g[np.maximum(idx_first - 1, 0)]  # outside (or boundary)
            b = g[idx_first]  # inside
        else:
            a = g[np.minimum(idx_last + 1, g.size - 1)]
            b = g[idx_last]
        edge = (idx_first == 0) if which == "lower" else (idx_last == g.size - 1)
        for _ in range(60):
            mid = 0.5 * (a + b)
            ins = self._inside(mid, q)
            b = np.where(ins, mid, b)
            a = np.where(ins, a, mid)
        res = np.where(edge, g[idx_first] if which == "lower" else g[idx_last], b)
        out[any_in] = res[any_in]
        return out

    def _inv_lower(self, q):
        return self._invert(q, "lower")

    def _inv_upper(self, q):
        return self._invert(q, "upper")
