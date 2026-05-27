#!/usr/bin/env python3
"""
Visualization of the TP2 checkerboard lower-bound proof at density level.

The figure visualizes deviations from independence,

    c - 1  ->  c_bar_m - 1  ->  c_Pi^Delta - 1,

where:
    c             is the original TP2 copula density,
    c_bar_m       averages c over vertical strips only,
    c_Pi^Delta    averages c over full m x n checkerboard cells.

The example uses the FGM copula with positive parameter theta:
    c(u,v) = 1 + theta (1 - 2u)(1 - 2v),  0 <= theta <= 1.
For theta > 0 this density is TP2 and has uniform marginals.

Output:
    images/tp2_checkerboard_proof_densities.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

THETA = 1.0
M = 3
N = 3
GRID = 700

OUT_DIR = "images"
OUT_FILE = "tp2_checkerboard_proof_densities.png"


# ─── TP2 DENSITY ──────────────────────────────────────────────────────────────


def fgm_density(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """FGM copula density.

    c(u,v) = 1 + theta(1 - 2u)(1 - 2v).

    For 0 <= theta <= 1 this is nonnegative and has uniform marginals.
    For theta >= 0 it is TP2 because
        c(u1,v1)c(u2,v2)-c(u1,v2)c(u2,v1)
        = 4 theta (u2-u1)(v2-v1) >= 0
    whenever u1 <= u2 and v1 <= v2.
    """
    return 1.0 + theta * (1.0 - 2.0 * u) * (1.0 - 2.0 * v)


def h_from_density(c: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
    """Compute h(u,v)=partial_1 C(u,v)=int_0^v c(u,s) ds.

    Input shape: c[v_index, u_index].
    Output shape: same.
    """
    return cumulative_trapezoid(c, v_grid, axis=0, initial=0.0)


def xi_from_h(h: np.ndarray, u_grid: np.ndarray, v_grid: np.ndarray) -> float:
    """Numerical xi from h=partial_1 C."""
    return 6.0 * np.trapezoid(np.trapezoid(h**2, u_grid, axis=1), v_grid) - 2.0


def make_cell_indices(grid: np.ndarray, num_cells: int) -> list[np.ndarray]:
    """Return grid indices for equal-width cells on [0,1]."""
    indices: list[np.ndarray] = []

    for i in range(num_cells):
        left = i / num_cells
        right = (i + 1) / num_cells

        if i < num_cells - 1:
            idx = np.where((grid >= left) & (grid < right))[0]
        else:
            idx = np.where((grid >= left) & (grid <= right))[0]

        indices.append(idx)

    return indices


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    u_grid = np.linspace(0.0, 1.0, GRID)
    v_grid = np.linspace(0.0, 1.0, GRID)
    U, V = np.meshgrid(u_grid, v_grid)

    # Original TP2 density.
    c = fgm_density(U, V, THETA)

    # Original h and xi.
    h = h_from_density(c, v_grid)
    xi_C = xi_from_h(h, u_grid, v_grid)

    # Grid-cell indices.
    strip_indices = make_cell_indices(u_grid, M)
    vcell_indices = make_cell_indices(v_grid, N)

    # c_bar_m: average c over vertical strips only.
    c_bar_m = np.zeros_like(c)
    for u_idx in strip_indices:
        # q_i(v) = m int_{I_i} c(u,v) du
        row_avg_density = M * np.trapezoid(c[:, u_idx], u_grid[u_idx], axis=1)
        c_bar_m[:, u_idx] = row_avg_density[:, None]

    h_bar_m = h_from_density(c_bar_m, v_grid)
    Q_m = xi_from_h(h_bar_m, u_grid, v_grid)

    # c_delta: average c over full m x n cells.
    # Equivalently, average c_bar_m over v-cells after averaging over u-strips.
    c_delta = np.zeros_like(c)
    for u_idx in strip_indices:
        for v_idx in vcell_indices:
            cell = c[np.ix_(v_idx, u_idx)]
            cell_mean = (
                M
                * N
                * np.trapezoid(
                    np.trapezoid(cell, u_grid[u_idx], axis=1),
                    v_grid[v_idx],
                )
            )
            c_delta[np.ix_(v_idx, u_idx)] = cell_mean

    h_delta = h_from_density(c_delta, v_grid)
    xi_delta = xi_from_h(h_delta, u_grid, v_grid)

    print("Approximate proof quantities")
    print(f"  xi(C)              = {xi_C:.5f}")
    print(f"  Q_m                = {Q_m:.5f}")
    print(f"  xi(C_Pi^Delta)     = {xi_delta:.5f}")

    print("Density ranges")
    print(f"  c           in [{c.min():.3f}, {c.max():.3f}]")
    print(f"  c_bar_m     in [{c_bar_m.min():.3f}, {c_bar_m.max():.3f}]")
    print(f"  c_Pi^Delta  in [{c_delta.min():.3f}, {c_delta.max():.3f}]")

    # ─── PLOTTING ─────────────────────────────────────────────────────────────

    # Plot deviations from independence. This makes the smoothing effect visible.
    Z_c = c - 1.0
    Z_bar_m = c_bar_m - 1.0
    Z_delta = c_delta - 1.0

    absmax = max(
        float(np.max(np.abs(Z_c))),
        float(np.max(np.abs(Z_bar_m))),
        float(np.max(np.abs(Z_delta))),
    )

    fig = plt.figure(figsize=(12.6, 4.1), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.04],
        left=0.055,
        right=0.93,
        bottom=0.17,
        top=0.68,
        wspace=0.32,
    )

    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    panels = [
        (Z_c, r"$c-1$", rf"$\xi(C)\approx {xi_C:.3f}$"),
        (
            Z_bar_m,
            r"$\bar c_m-1$",
            rf"$Q_m\approx {Q_m:.3f}$",
        ),
        (
            Z_delta,
            r"$c^\Delta_\Pi-1$",
            rf"$\xi(C^\Delta_\Pi)\approx {xi_delta:.3f}$",
        ),
    ]

    im = None
    for ax, (Z, title, subtitle) in zip(axes, panels):
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-absmax,
            vmax=absmax,
        )

        # Draw m x n grid.
        for k in range(1, M):
            ax.axvline(k / M, color="black", linewidth=0.8, alpha=0.35)
        for ll in range(1, N):
            ax.axhline(ll / N, color="black", linewidth=0.8, alpha=0.35)

        ax.set_title(title + "\n" + subtitle, fontsize=13, pad=8)
        ax.set_xlabel(r"$u$", fontsize=12)
        ax.set_ylabel(r"$v$", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

    if im is None:
        raise RuntimeError("No image was created for the colorbar.")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("density minus 1", rotation=270, labelpad=14, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        r"TP2 checkerboard lower-bound proof: density smoothing steps "
        r"($m=n=3$)",
        fontsize=15,
        y=0.97,
    )

    out_path = os.path.join(OUT_DIR, OUT_FILE)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
