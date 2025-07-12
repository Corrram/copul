#!/usr/bin/env python3
"""
Plots the exact attainable region for
Chatterjee's ξ and Spearman's Footrule ψ,
with the axes interchanged.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def main() -> None:
    # ---------------- Boundary Definition ----------------
    # The boundary is given by |ψ| = √ξ. Parameterize using ψ.
    psi_boundary = np.linspace(0, 1, 500)
    xi_boundary = psi_boundary**2

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the boundary envelope with axes swapped: x = ξ, y = ψ
    ax.plot(xi_boundary, psi_boundary, color=BLUE, lw=2.5,
            label=r"Boundary: $|\psi| = \sqrt{\xi}$")
    ax.plot(xi_boundary, -psi_boundary, color=BLUE, lw=2.5)

    # Fill the attainable region: between y = -ψ and y = ψ for each ξ
    ax.fill_between(
        xi_boundary, -psi_boundary, psi_boundary,
        color=FILL, alpha=0.7, zorder=0,
        label="Attainable Region"
    )

    # ---------------------- Highlight key points ----------------------
    psi_max_diff = 0.5
    xi_max_diff = 0.25

    key_psi = [0, 1, -1, psi_max_diff, -psi_max_diff]
    key_xi  = [0, 1, 1, xi_max_diff, xi_max_diff]
    # Swap positions: (ξ, ψ)
    ax.scatter(key_xi, key_psi, s=70, color="black", zorder=5, clip_on=False)

    # Annotations at swapped positions
    ax.annotate(r"$\Pi$", (0, 0), xytext=(10, 0), textcoords="offset points",
                fontsize=18, va="center")
    ax.annotate(r"$M$", (1, 1), xytext=(-15, -15), textcoords="offset points",
                fontsize=18, ha="left", va="center")
    ax.annotate(r"$W$", (1, -1), xytext=(-15, 15), textcoords="offset points",
                fontsize=18, ha="left", va="center")
    ax.annotate(r"$C^{\text{Fr}}_{0.5}$", (xi_max_diff, psi_max_diff),
                xytext=(5, -15), textcoords="offset points", fontsize=18, ha="left")
    ax.annotate(r"$C^{\text{Fr}}_{-0.5}$", (xi_max_diff, -psi_max_diff),
                xytext=(5, 5), textcoords="offset points", fontsize=18, ha="left")

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Chatterjee's $\xi$", fontsize=16)
    ax.set_ylabel(r"Spearman's Footrule $\psi$", fontsize=16)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    ax.legend(loc="center", fontsize=12, frameon=True)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()