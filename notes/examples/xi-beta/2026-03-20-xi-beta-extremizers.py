#!/usr/bin/env python3
"""
Generate a 2x2 contour plot of the conditional distribution C(v|u) = \partial_u C(u,v)
for the XiBetaBoundaryCopula at b = -0.7, -0.3, 0.3, and 0.7.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker


def cond_cdf_u(u, v, b):
    """
    Computes the conditional distribution P(V <= v | U = u) = \partial C(u,v) / \partial u
    for the 2x2 checkerboard boundary copula.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    res = np.zeros_like(u)

    # Define the 4 quadrants
    q1 = (u <= 0.5) & (v <= 0.5)
    q2 = (u <= 0.5) & (v > 0.5)
    q3 = (u > 0.5) & (v <= 0.5)
    q4 = (u > 0.5) & (v > 0.5)

    # Partial derivatives wrt u based on the exact region paper
    res[q1] = (1.0 + b) * v[q1]
    res[q2] = v[q2] + b * (1.0 - v[q2])
    res[q3] = (1.0 - b) * v[q3]
    res[q4] = v[q4] - b * (1.0 - v[q4])

    return res


def main():
    # Grid for contour plots
    u_vals = np.linspace(0, 1, 500)
    v_vals = np.linspace(0, 1, 500)
    U, V = np.meshgrid(u_vals, v_vals)

    b_values = [-0.7, -0.3, 0.3, 0.7]

    # Configure publication-ready plot settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'text.usetex': False  # Set to True if you have a full LaTeX engine installed
    })

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Create the contour plots
    contour_levels = np.linspace(0, 1, 50)  # 50 levels looks very smooth

    for ax, b in zip(axes, b_values):
        Z = cond_cdf_u(U, V, b)

        # Use a clean colormap like 'viridis' or 'Blues'
        cf = ax.contourf(U, V, Z, levels=contour_levels, cmap='viridis', vmin=0, vmax=1)

        # Add a subtle grid and crosshairs at the quadrant boundaries (0.5, 0.5)
        ax.axhline(0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(0.5, color='white', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_title(rf"$b = {b}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        # Clean up tick marks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    # Add axis labels to the outer plots
    for ax in axes[2:]:
        ax.set_xlabel(r"$u$")
    for ax in axes[0::2]:
        ax.set_ylabel(r"$v$")

    # Add a single colorbar for the whole figure
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label(r"$P(V \leq v \mid U = u)$", rotation=270, labelpad=20, fontsize=14)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.update_ticks()

    fig.suptitle(r"Conditional CDF of the $C_b^\#$ Boundary Family", y=0.95, fontsize=16)

    # Save to file
    Path("figs").mkdir(exist_ok=True)
    out_path = "figs/cond_cdf_boundary.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()