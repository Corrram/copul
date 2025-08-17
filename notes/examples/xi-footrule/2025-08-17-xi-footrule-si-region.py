#!/usr/bin/env python3
"""
Plots the attainable region for Stochastically Increasing (SI) copulas.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def main() -> None:
    # --- Boundary Definition for SI Region ---
    xi_vals = np.linspace(0, 1, 500)
    psi_upper_bound = np.sqrt(xi_vals)  # Upper bound is psi = sqrt(xi)
    psi_lower_bound = xi_vals  # Lower bound is psi = xi

    # --- Plotting ---
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the boundary lines
    ax.plot(xi_vals, psi_upper_bound, color=BLUE, lw=2.5, zorder=3, label="Upper Bound")
    ax.plot(
        xi_vals,
        psi_lower_bound,
        color=BLUE,
        lw=2.5,
        # ls="--",
        zorder=2,
        label="Lower Bound (SI)",
    )

    # Fill the SI region with a dotted hatch
    ax.fill_between(
        xi_vals,
        psi_lower_bound,
        psi_upper_bound,
        facecolor=FILL,
        # hatch="..",
        edgecolor=BLUE,
        linewidth=0,
        zorder=1,
        label="SI region",
    )

    # --- Highlight key points relevant to the SI region ---
    ax.scatter([0, 1, 0.25], [0, 1, 0.5], s=70, color="black", zorder=5)
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(15, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="center",
    )
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )
    ax.annotate(
        r"$C^{\mathrm{Fr}}_{1/2}$",
        (0.25, 0.5),
        xytext=(-15, 15),
        textcoords="offset points",
        fontsize=18,
        ha="center",
    )

    # --- Axes, grid, legend ---
    # ax.set_title("Attainable Region for SI Copulas", fontsize=16)
    ax.set_xlabel(r"Chatterjee's $\xi$", fontsize=16)
    # ax.set_ylabel(r"Spearman's footrule $\psi$", fontsize=16)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.55, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    # ax.legend(loc="upper left", fontsize=12, frameon=True)
    fig.tight_layout()
    plt.savefig("attainable_region_si.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
