#!/usr/bin/env python3
"""
Plot the attainable region for (Blest’s ν, Chatterjee’s ξ)  — axes swapped.

- Right boundary:  ( +ν(μ), ξ(μ) )   for μ > 0
- Left  boundary:  ( -ν(μ), ξ(μ) )   by survival symmetry
- Marks key copulas M (1,1), Π (0,0), W (-1,1)
- Optionally exports the boundary samples as CSV and SVG/PNG.

Notes:
* Uses the closed forms from the clamped–parabola optimiser.
* Samples μ on both sides of μ=1 with dense geometric spacing.
* Numerically careful near μ→0⁺ (uses a small positive lower bound).
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
# Closed-form ξ(μ), ν(μ)
# ----------------------------------------------------------------------
def xi_mu_scalar(mu: float) -> float:
    """ξ(μ) with closed forms for μ<1 and μ≥1; continuous at μ=1."""
    if mu <= 0.0:
        return 1.0
    if mu >= 1.0:
        return (8.0 * (7.0 * mu - 3.0)) / (105.0 * mu**3)

    s = np.sqrt(mu)
    t = np.sqrt(1.0 - mu)
    A = np.arcsinh(t / s)
    num = (
        -105.0 * s**8 * A
        + 183.0 * s**6 * t
        - 38.0 * s**4 * t
        - 88.0 * s**2 * t
        + 112.0 * s**2
        + 48.0 * t
        - 48.0
    )
    den = 210.0 * s**6
    return num / den


def sqrt1pm1(x):
    """
    Numerically stable version of sqrt(1 + x) - 1.
    Avoids catastrophic cancellation for small |x|.
    """
    x = np.asarray(x)
    # For small x, use series: sqrt(1+x) - 1 ≈ x / (sqrt(1+x)+1)
    return np.where(np.abs(x) < 1e-8, x / (2 + 0.5 * x), np.sqrt(1 + x) - 1)


def nu_mu_scalar(mu: float) -> float:
    """ν(μ) with high-precision evaluation for μ<1 to avoid cancellation."""
    m = np.longdouble(mu)
    if m <= 0:
        return 1.0
    if m >= 1:
        val = (4.0 * (28.0 * m - 9.0)) / (105.0 * m**2)
        return float(np.clip(val, -1.0, 1.0))

    s = np.sqrt(m, dtype=np.longdouble)
    tau = sqrt1pm1(-m)  # t - 1
    t = 1.0 + tau
    A = np.arcsinh(t / s)

    num = (
        -105.0 * s**8 * A
        + 87.0 * s**6 * t
        + 250.0 * s**4 * t
        - 376.0 * s**2 * t
        + 448.0 * s**2
        + 144.0 * t
        - 144.0
    )
    den = 420.0 * s**4
    val = num / den
    return float(np.clip(val, -1.0, 1.0))


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PlotCfg:
    mu_min: float = 1e-8
    mu_left_pts: int = 1000
    mu_right_max: float = 1e3
    mu_right_pts: int = 1000
    show_full_box: bool = False  # if True: x,y ∈ [-1,1]^2; else x∈[-1,1], y∈[0,1]
    figsize: tuple[int, int] = (8, 8)
    line_width: float = 2.2
    palette_blue: str = "#0B66C3"
    palette_fill: str = "#DCEEFF"
    out_dir: Path = Path("images")
    out_basename: str = "nu-xi-region"  # updated default name
    export_csv: bool = True
    export_svg: bool = True
    export_png: bool = True
    dpi: int = 300


# ----------------------------------------------------------------------
# Sampling μ and computing the boundary (ξ increasing)
# ----------------------------------------------------------------------
def sample_boundary(cfg: PlotCfg):
    # Dense sampling on both sides of μ=1
    mu_left = np.geomspace(0.001, 0.999, cfg.mu_left_pts, endpoint=False)
    mu_right = np.geomspace(1.0, cfg.mu_right_max, cfg.mu_right_pts, endpoint=True)
    mu = np.concatenate([mu_left, mu_right])

    with np.errstate(all="ignore"):
        xi_vals = np.array([xi_mu_scalar(m) for m in mu])
        nu_vals = np.array([nu_mu_scalar(m) for m in mu])

    # Sort by ξ ascending (strictly monotone in μ)
    idx = np.argsort(xi_vals)
    x = xi_vals[idx]  # ξ(μ)
    y = nu_vals[idx]  # ν(μ)

    # Collapse near-duplicate ξ to make it strictly increasing
    xr = np.round(x, 12)
    starts = np.flatnonzero(np.r_[True, xr[1:] != xr[:-1]])
    sizes = np.diff(np.r_[starts, len(x)])
    x_u = xr[starts].astype(float)
    y_u = np.add.reduceat(y, starts) / sizes

    # Clip and ensure endpoints
    x_u = np.clip(x_u, 0.0, 1.0)
    y_u = np.clip(y_u, -1.0, 1.0)
    if x_u[0] > 0.0:
        x_u = np.r_[0.0, x_u]
        y_u = np.r_[0.0, y_u]
    if x_u[-1] < 1.0:
        x_u = np.r_[x_u, 1.0]
        y_u = np.r_[y_u, 1.0]

    # Return ξ grid and corresponding ν_max(ξ)
    return x_u, y_u


# ----------------------------------------------------------------------
# Plot (axes swapped: x-axis = ν, y-axis = ξ)
# ----------------------------------------------------------------------
def plot_region(cfg: PlotCfg, xi_grid: np.ndarray, nu_max: np.ndarray):
    BLUE, FILL = cfg.palette_blue, cfg.palette_fill
    fig, ax = plt.subplots(figsize=cfg.figsize)

    # Right/left boundary curves as functions of y = ξ
    x_right = +nu_max
    x_left = -nu_max
    y_vals = xi_grid

    # Boundaries
    ax.plot(x_right, y_vals, color=BLUE, lw=cfg.line_width, label=r"$\nu_{\max}(\xi)$")
    ax.plot(x_left, y_vals, color=BLUE, lw=cfg.line_width, label=r"$\nu_{\min}(\xi)$")

    # Shade region between left and right at fixed ξ
    ax.fill_betweenx(
        y_vals,
        x_left,
        x_right,
        color=FILL,
        alpha=0.9,
        zorder=0,
        label="Attainable region",
    )

    # Key copulas in (ν, ξ) coordinates
    ax.scatter([0, 1, -1], [0, 1, 1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=15,
        ha="center",
        va="bottom",
    )
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(-8, -6),
        textcoords="offset points",
        fontsize=15,
        ha="right",
        va="top",
    )
    ax.annotate(
        r"$W$",
        (-1, 1),
        xytext=(8, -6),
        textcoords="offset points",
        fontsize=15,
        ha="left",
        va="top",
    )

    # Example optimiser: μ=1 → (ξ,ν)=(32/105, 76/105)
    xi1 = 32.0 / 105.0
    nu1 = 76.0 / 105.0
    ax.scatter([+nu1], [xi1], s=60, color="black", zorder=6)
    ax.annotate(
        r"$C_{1}$",
        (nu1, xi1),
        xytext=(4, -4),
        textcoords="offset points",
        fontsize=13,
        ha="left",
        va="top",
    )
    # symmetric point
    ax.scatter([-nu1], [xi1], s=60, color="black", zorder=6)
    ax.annotate(
        r"$C_{-1}$",
        (-nu1, xi1),
        xytext=(-8, -4),
        textcoords="offset points",
        fontsize=13,
        ha="right",
        va="bottom",
    )

    # Axes & styling (now: x=ν ∈ [-1,1], y=ξ ∈ [0,1])
    ax.set_xlabel(r"Blest's $\nu$", fontsize=15)
    ax.set_ylabel(r"Chatterjee's $\xi$", fontsize=15)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    # ax.legend(loc="lower right", fontsize=11, frameon=True)

    fig.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
# Export helpers (kept same columns: xi, nu_max, nu_min)
# ----------------------------------------------------------------------
def export_outputs(
    cfg: PlotCfg, xi_grid: np.ndarray, nu_max: np.ndarray, fig: plt.Figure
):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.export_csv:
        data = np.column_stack([xi_grid, nu_max, -nu_max])
        header = "xi,nu_max,nu_min"
        np.savetxt(
            cfg.out_dir / f"{cfg.out_basename}.csv",
            data,
            delimiter=",",
            header=header,
            comments="",
        )

    if cfg.export_svg:
        fig.savefig(cfg.out_dir / f"{cfg.out_basename}.svg")
    if cfg.export_png:
        fig.savefig(
            cfg.out_dir / f"{cfg.out_basename}.png", dpi=config.dpi
        )  # fallback if cfg.export_png is True


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> PlotCfg:
    p = argparse.ArgumentParser(
        description="Plot the attainable region with axes swapped (ν on x, ξ on y)."
    )
    p.add_argument(
        "--full-box",
        action="store_true",
        help="Show full [-1,1]×[-1,1] axes (y will still be clipped to [0,1]).",
    )
    p.add_argument("--no-csv", action="store_true", help="Do not export CSV.")
    p.add_argument("--no-svg", action="store_true", help="Do not export SVG.")
    p.add_argument("--no-png", action="store_true", help="Do not export PNG.")
    p.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300).")
    p.add_argument("--out", type=str, default="nu-xi-region", help="Output basename.")
    args = p.parse_args()

    return PlotCfg(
        show_full_box=args.full_box,
        export_csv=not args.no_csv,
        export_svg=not args.no_svg,
        export_png=not args.no_png,
        dpi=args.dpi,
        out_basename=args.out,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    cfg = parse_args()
    xi_grid, nu_max = sample_boundary(cfg)
    fig, _ = plot_region(cfg, xi_grid, nu_max)
    # quick guard for export_png typo: use the same flag as before
    if cfg.export_png:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.out_dir / f"{cfg.out_basename}.png", dpi=cfg.dpi)
    if cfg.export_svg:
        fig.savefig(cfg.out_dir / f"{cfg.out_basename}.svg")
    if cfg.export_csv:
        data = np.column_stack([xi_grid, nu_max, -nu_max])
        np.savetxt(
            cfg.out_dir / f"{cfg.out_basename}.csv",
            data,
            delimiter=",",
            header="xi,nu_max,nu_min",
            comments="",
        )
    plt.show()


if __name__ == "__main__":
    main()
