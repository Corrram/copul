r"""
Matplotlib helpers for publication-style region figures.

The helpers reproduce the look of the region plots in ``notes/`` (blue
boundary, light-blue fill, dotted grid, equal aspect, labelled key copulas
:math:`M`, :math:`W`, :math:`\Pi`) without touching global state unless
:func:`paper_style` is used as a context manager.

Examples
--------
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import matplotlib.pyplot as plt
>>> from copul.regions.style import apply_paper_axes, mark_points
>>> fig, ax = plt.subplots()
>>> _ = apply_paper_axes(ax, "xi", "rho")
>>> _ = mark_points(ax, [("M", 1.0, 1.0), (r"\Pi", 0.0, 0.0)])
>>> plt.close(fig)
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence

from copul.regions.measures import MEASURES, label, resolve

__all__ = [
    "BLUE",
    "FILL",
    "apply_paper_axes",
    "mark_points",
    "paper_style",
]

#: Boundary colour used throughout the notes.
BLUE = "#00529B"
#: Fill colour of attainable regions.
FILL = "#D6EAF8"

_PAPER_RC = {
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 100,
}


@contextlib.contextmanager
def paper_style(**overrides):
    """Context manager applying paper rcParams (serif fonts, CM math, 300 dpi).

    Examples
    --------
    >>> from copul.regions.style import paper_style
    >>> with paper_style():  # doctest: +SKIP
    ...     copul.regions.get("xi", "rho").plot()
    """
    import matplotlib as mpl

    rc = dict(_PAPER_RC)
    rc.update(overrides)
    with mpl.rc_context(rc):
        yield


def _padded(lim: Sequence[float], pad: float) -> tuple[float, float]:
    lo, hi = float(lim[0]), float(lim[1])
    d = pad * (hi - lo)
    return lo - d, hi + d


def apply_paper_axes(
    ax,
    x: str,
    y: str,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    aspect: str | float = "equal",
    major: float | None = 0.25,
    pad: float = 0.025,
    axis_lines: bool = True,
    grid: bool = True,
):
    """Label and style ``ax`` for a region plot of measure ``y`` against ``x``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : str
        Measure keys (labels and default limits are taken from
        :data:`copul.regions.measures.MEASURES`).
    xlim, ylim : pair of float, optional
        Axis limits before padding (default: the measures' ranges).
    aspect : "equal", "auto" or float
        Passed to :meth:`~matplotlib.axes.Axes.set_aspect`.
    major : float or None
        Major tick spacing.
    pad : float
        Relative padding added to the limits.
    """
    from matplotlib.ticker import MultipleLocator

    xk, yk = resolve(x), resolve(y)
    ax.set_xlabel(label(xk))
    ax.set_ylabel(label(yk))
    ax.set_xlim(*_padded(xlim if xlim is not None else MEASURES[xk].range, pad))
    ax.set_ylim(*_padded(ylim if ylim is not None else MEASURES[yk].range, pad))
    if aspect is not None:
        ax.set_aspect(aspect, adjustable="box")
    if major:
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.yaxis.set_major_locator(MultipleLocator(major))
    if grid:
        ax.grid(True, linestyle=":", alpha=0.6)
    if axis_lines:
        ax.axhline(0.0, color="black", lw=0.8, zorder=1)
        ax.axvline(0.0, color="black", lw=0.8, zorder=1)
    return ax


def mark_points(
    ax,
    points: Iterable,
    color: str = "black",
    size: float = 36,
    fontsize: float = 13,
    offset: float = 8.0,
    zorder: int = 5,
):
    r"""Scatter and label key copulas.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    points : iterable
        Items ``(label, x, y)`` or objects with ``label``, ``x``, ``y``
        attributes (e.g. :class:`copul.regions.KeyPoint`).  Labels are
        wrapped in ``$...$`` unless they already contain a dollar sign.
    offset : float
        Label offset in points; labels are pushed towards the centre of the
        axes so that they stay inside the region.
    """
    xl, yl = ax.get_xlim(), ax.get_ylim()
    cx, cy = 0.5 * (xl[0] + xl[1]), 0.5 * (yl[0] + yl[1])
    for p in points:
        lab, px, py = (p.label, p.x, p.y) if hasattr(p, "label") else p
        ax.scatter([px], [py], s=size, color=color, zorder=zorder)
        txt = lab if "$" in lab else f"${lab}$"
        dx = -offset if px > cx else offset
        dy = -offset if py > cy else offset
        ax.annotate(
            txt,
            (px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            va="top" if dy < 0 else "bottom",
            fontsize=fontsize,
            color=color,
            zorder=zorder,
        )
    return ax
