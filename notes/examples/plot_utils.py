"""
plot_utils.py – shared utilities for rank-correlation region / families plots.

Import from any script in this tree with:

    # scripts in a sub-folder (e.g. blest-regions/, xi-rho/, …):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from plot_utils import ...

    # scripts directly in notes/examples/:
    from plot_utils import ...
"""

import pathlib
import pickle
import importlib.resources as pkg_resources
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── Colour palette ───────────────────────────────────────────────────────────

BLUE = "#00529B"   # boundary / envelope line colour
FILL = "#D6EAF8"   # attainable-region fill colour

# ── Standard axes styling ────────────────────────────────────────────────────


def apply_standard_axes(
    ax,
    xlabel: str,
    ylabel: str,
    xlim: Tuple[float, float] = (-1.05, 1.05),
    ylim: Tuple[float, float] = (-1.05, 1.05),
    major_x: float = 0.25,
    major_y: float = 0.25,
    fontsize: int = 16,
    tick_labelsize: int = 13,
) -> None:
    """Apply the standard cosmetic treatment shared by all region plots."""
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(major_x))
    ax.yaxis.set_major_locator(MultipleLocator(major_y))
    ax.tick_params(labelsize=tick_labelsize)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)


# ── Key-copula point annotations ─────────────────────────────────────────────


def scatter_annotate(
    ax,
    x: float,
    y: float,
    label: str,
    xytext: Tuple[int, int],
    ha: str = "center",
    va: str = "bottom",
    fontsize: int = 18,
    scatter_size: int = 60,
    color: str = "black",
    zorder: int = 5,
) -> None:
    """Plot a single dot and its text annotation (dot and text share *color*)."""
    ax.scatter([x], [y], s=scatter_size, color=color, zorder=zorder)
    ax.annotate(
        label,
        (x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=fontsize,
        ha=ha,
        va=va,
        color=color,
    )


def mark_key_copulas(
    ax,
    M:  Optional[Tuple[float, float]] = (1.0,  1.0),
    W:  Optional[Tuple[float, float]] = (-1.0, -1.0),
    Pi: Optional[Tuple[float, float]] = (0.0,  0.0),
    M_xytext:  Tuple[int, int] = (-10,  0),
    W_xytext:  Tuple[int, int] = ( 10,  0),
    Pi_xytext: Tuple[int, int] = (  0, 18),
    M_ha:  str = "right",  M_va:  str = "top",
    W_ha:  str = "left",   W_va:  str = "bottom",
    Pi_ha: str = "center", Pi_va: str = "bottom",
    fontsize: int = 18,
) -> None:
    """
    Mark and label the three Fréchet-Hoeffding bounds M, W, and Π.

    Pass ``M=None``, ``W=None``, or ``Pi=None`` to skip individual points.
    Custom offset (*_xytext), alignment (*_ha / *_va) and *fontsize* can all
    be overridden per-point.
    """
    if M is not None:
        scatter_annotate(ax, M[0],  M[1],  r"$M$",   M_xytext,  M_ha,  M_va,  fontsize)
    if W is not None:
        scatter_annotate(ax, W[0],  W[1],  r"$W$",   W_xytext,  W_ha,  W_va,  fontsize)
    if Pi is not None:
        scatter_annotate(ax, Pi[0], Pi[1], r"$\Pi$", Pi_xytext, Pi_ha, Pi_va, fontsize)


# ── Figure save helper ───────────────────────────────────────────────────────


def save_region_plot(
    fig,
    path: str | pathlib.Path,
    dpi: int = 300,
    show: bool = True,
) -> None:
    """Tighten layout, create parent dirs, save, and optionally show the figure."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=dpi)
    if show:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Data infrastructure shared by "families combined" plots
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CorrelationData:
    params: np.ndarray
    values: Dict[str, np.ndarray]


def find_data_dir(script_file: Optional[str] = None) -> pathlib.Path:
    """
    Locate the ``rank_correlation_estimates`` pickle directory.

    Search order:
    1. The installed ``copul`` package's ``docs/`` folder.
    2. Three levels up from *script_file* (or this file) then into
       ``docs/rank_correlation_estimates``.
    3. ``rank_correlation_estimates/`` in the current working directory
       (fallback, prints a warning).

    Pass ``__file__`` from the calling script so that the relative path
    is computed relative to that script, not to this module.
    """
    default = pathlib.Path("rank_correlation_estimates")

    # 1. Installed package
    try:
        with pkg_resources.path("copul", "docs") as docs_path:
            candidate = pathlib.Path(docs_path) / "rank_correlation_estimates"
            if candidate.exists():
                print(f"Loading data from: {candidate.resolve()}")
                return candidate
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    # 2. Relative path from the calling script
    anchor = pathlib.Path(script_file) if script_file else pathlib.Path(__file__)
    relative = (anchor.parent / "../../../docs/rank_correlation_estimates").resolve()
    if relative.exists():
        print(f"Loading data from: {relative}")
        return relative

    print("Warning: Could not find data directory. Empirical curves may be missing.")
    return default


def load_family_data(
    family: str,
    data_dir: pathlib.Path,
    key1: str = "chatterjees_xi",
    key2: str = "spearmans_rho",
    extra_keys: Optional[List[str]] = None,
    filter_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
) -> Tuple:
    """
    Load a CorrelationData pickle and return masked arrays for *key1*, *key2*,
    and any *extra_keys*.

    Returns ``(None, None, ...)`` on missing file or load failure.

    Parameters
    ----------
    family      : copula family name (used to find the pickle file).
    data_dir    : directory containing the pickle files.
    key1, key2  : primary keys to extract (default: ξ and ρ).
    extra_keys  : additional keys (e.g. ``["kendalls_tau"]``); missing keys
                  yield arrays of NaN.
    filter_fn   : optional callable ``(arr1, arr2) -> bool-mask`` for extra
                  filtering (e.g. restricting to positive dependence).
    """
    n_out = 2 + len(extra_keys or [])
    none_result: Tuple = (None,) * n_out

    candidates = [
        data_dir / f"{family}_data.pkl",
        data_dir / f"{family}.pkl",
        data_dir / f"{family} Copula_data.pkl",
    ]
    file_path = next((c for c in candidates if c.exists()), None)
    if file_path is None:
        return none_result

    try:
        data = pickle.loads(file_path.read_bytes())
    except Exception as exc:
        print(f"Error loading {family}: {exc}")
        return none_result

    try:
        arr1 = data.values.get(key1)
        arr2 = data.values.get(key2)
        if arr1 is None or arr2 is None:
            return none_result
        extras = [
            data.values.get(k, np.full_like(arr1, np.nan))
            for k in (extra_keys or [])
        ]
    except AttributeError:
        return none_result

    mask = np.isfinite(arr1) & np.isfinite(arr2)
    if filter_fn is not None:
        mask &= filter_fn(arr1, arr2)
    if not np.any(mask):
        return none_result

    return (arr1[mask], arr2[mask]) + tuple(e[mask] for e in extras)


# ═══════════════════════════════════════════════════════════════════════════
#  Shared analytic copula curves
# ═══════════════════════════════════════════════════════════════════════════


def get_gaussian_xi_rho_tau(n_points: int = 300, r_max: float = 1.0):
    """
    Gaussian copula: (ξ, ρ, τ) as a function of the Pearson-r parameter r ∈ [0, r_max].
    """
    r = np.linspace(0, r_max, n_points)
    rho = (6 / np.pi) * np.arcsin(r / 2)
    tau = (2 / np.pi) * np.arcsin(r)
    xi  = (3 / np.pi) * np.arcsin((1 + r**2) / 2) - 0.5
    return xi, rho, tau


def get_cb_xi_rho_tau(n_points: int = 1000):
    """
    C_b (clamped-bilinear / uniform-strip) family: (ξ, ρ, τ) for b > 0.
    """
    b = np.concatenate([
        np.linspace(0, 1, n_points),
        np.linspace(1, 100, n_points)[1:],
    ])
    with np.errstate(divide="ignore", invalid="ignore"):
        xi  = np.where(b <= 1, (b**2 / 10) * (5 - 2*b), 1 - 1/b + 3/(10*b**2))
        rho = np.where(b <= 1, b - 3*b**2/10,            1 - 1/(2*b**2) + 1/(5*b**3))
        tau = np.where(b <= 1, 2*b/3 - b**2/6,           1 - 2/(3*b) + 1/(6*b**2))
    mask = np.isfinite(xi) & np.isfinite(rho)
    return xi[mask], rho[mask], tau[mask]


def get_marshall_olkin_a1eq1_xi_rho_tau(n_points: int = 1000):
    """
    Marshall-Olkin with α₁ = 1 fixed, α₂ ∈ [0, 1] varying: (ξ, ρ, τ).

    Formulas (substituting α₁ = 1, α₂ = a2):
        ρ = 3·a2 / (2 + a2),   τ = a2,   ξ = 2·a2 / (3 − a2).
    """
    a2  = np.linspace(0, 1, n_points)
    rho = 3 * a2 / (2 + a2)
    tau = a2
    xi  = 2 * a2 / (3 - a2)
    return xi, rho, tau
