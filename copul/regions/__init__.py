r"""
Registry of known exact regions between dependence measures.

An :class:`ExactRegion` describes the set of attainable pairs
:math:`(\kappa_x(C),\kappa_y(C))` over all bivariate copulas (or a subclass
such as SI copulas) by closed-form lower/upper boundary functions, together
with key points, boundary-attaining copula families, a reference and the
source file in this repository.

Every registered region is validated numerically in ``tests/regions`` (the
boundary copulas attain the boundary, 2000 random checkerboards lie inside,
and optimal checkerboards from :mod:`copul.optim` approach the boundary from
inside).

Worked example
--------------
>>> import copul.regions as cr
>>> sorted(cr.available())[:3]
[('rho', 'nu'), ('xi', 'nu'), ('xi', 'rho')]
>>> reg = cr.get("xi", "rho")
>>> float(reg.upper(0.3))
0.7
>>> bool(reg.contains(0.3, 0.69)), bool(reg.contains(0.3, 0.71))
(True, False)
>>> round(float(cr.get("rho", "xi").lower(0.7)), 12)  # axes swapped automatically
0.3
>>> C = reg.boundary_copula(0.3, side="upper")  # XiRhoBoundaryCopula(b=1)
>>> ax = reg.plot()  # doctest: +SKIP
>>> fig = cr.plot_grid([("xi", "rho"), ("xi", "nu"), ("rho", "nu")])  # doctest: +SKIP

Measure keys: ``"xi"``, ``"rho"``, ``"tau"``, ``"footrule"``, ``"gamma"``,
``"beta"``, ``"nu"`` (aliases such as ``"spearman"`` or ``"psi"`` work too; see
:mod:`copul.regions.measures`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from copul.regions.base import ExactRegion, KeyPoint, SwappedRegion
from copul.regions.measures import MEASURES, evaluate, label, resolve
from copul.regions.style import BLUE, FILL, apply_paper_axes, mark_points, paper_style

__all__ = [
    "BLUE",
    "FILL",
    "MEASURES",
    "ExactRegion",
    "KeyPoint",
    "SwappedRegion",
    "apply_paper_axes",
    "available",
    "evaluate",
    "get",
    "label",
    "mark_points",
    "paper_style",
    "plot_grid",
    "register",
    "resolve",
]

_REGISTRY: dict[tuple[str, str, str], ExactRegion] = {}
_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if not _LOADED:
        from copul.regions.catalog import build_default_regions

        for reg in build_default_regions():
            _REGISTRY.setdefault(reg.key, reg)
        _LOADED = True


def register(region: ExactRegion, overwrite: bool = False) -> ExactRegion:
    """Register an additional :class:`ExactRegion`.

    Parameters
    ----------
    region : ExactRegion
    overwrite : bool
        Replace an existing region with the same ``(x, y, class)`` key.
    """
    _ensure_loaded()
    if region.key in _REGISTRY and not overwrite:
        raise KeyError(f"region {region.key} already registered")
    _REGISTRY[region.key] = region
    return region


def available(copula_class: str | None = "all") -> list[tuple[str, str]]:
    """Registered ``(x, y)`` pairs (in their natural orientation).

    Parameters
    ----------
    copula_class : str or None
        Restrict to a class (``"all"``, ``"si"``, ...); ``None`` lists every
        region as ``(x, y, class)`` triples.
    """
    _ensure_loaded()
    if copula_class is None:
        return sorted(_REGISTRY)  # type: ignore[return-value]
    return sorted((x, y) for (x, y, c) in _REGISTRY if c == copula_class)


def get(x: str, y: str, copula_class: str = "all") -> ExactRegion:
    """Look up the exact region of ``(x, y)``.

    The lookup is order-insensitive: if only ``(y, x)`` is registered, the
    region is returned with swapped axes (:class:`SwappedRegion`).

    Raises
    ------
    KeyError
        If no such region is registered.
    """
    _ensure_loaded()
    xk, yk = resolve(x), resolve(y)
    c = copula_class.lower()
    if (xk, yk, c) in _REGISTRY:
        return _REGISTRY[(xk, yk, c)]
    if (yk, xk, c) in _REGISTRY:
        key = ("swap", xk, yk, c)
        if key not in _SWAPPED:
            _SWAPPED[key] = _REGISTRY[(yk, xk, c)].swap()
        return _SWAPPED[key]
    raise KeyError(
        f"No exact region registered for ({xk}, {yk}) and class {c!r}. Available: {available(None)}"
    )


_SWAPPED: dict = {}


def plot_grid(
    pairs: Iterable[Sequence[str]] | None = None,
    ncols: int = 3,
    size: float = 3.6,
    fill: bool = True,
    mark: bool = True,
    titles: bool = True,
):
    """Plot several regions side by side.

    Parameters
    ----------
    pairs : iterable of (x, y) or (x, y, class)
        Regions to draw (default: all registered ones).
    ncols : int
        Number of columns.
    size : float
        Size of each panel in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    _ensure_loaded()
    if pairs is None:
        pairs = [(x, y, c) for (x, y, c) in sorted(_REGISTRY)]
    regs = [get(*p) for p in pairs]
    nrows = max(1, -(-len(regs) // ncols))
    ncols = min(ncols, max(1, len(regs)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    for ax, reg in zip(axes.ravel(), regs):
        reg.plot(ax=ax, fill=fill, mark=mark)
        if titles:
            ax.set_title(reg.title)
    for ax in axes.ravel()[len(regs) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
