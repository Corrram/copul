#!/usr/bin/env python3
"""
Plots the exact attainable region for
Chatterjee's ξ and Spearman's Footrule ψ,
with dotted hatches on the side wings beyond the diagonal lines ψ = ±ξ.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def main() -> None:
    # ---------------- Boundary Definition ----------------
    # The boundary is given by |ψ| = √ξ. We can parameterize this
    # easily using ψ as the parameter.
    psi_boundary = np.linspace(0, 1, 500)
    xi_boundary  = psi_boundary**2

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the boundary envelope: ξ = ψ²
    ax.plot(psi_boundary,  xi_boundary, color=BLUE, lw=2.5,
            label=r"Boundary: $|\psi| = \pm\sqrt{\xi}$")
    ax.plot(-psi_boundary, xi_boundary, color=BLUE, lw=2.5)

    # Add thin diagonal lines from (0,0) to (±1,1)
    ax.plot([0, 1], [0, 1], color=BLUE, lw=1, linestyle="--")
    ax.plot([0, -1], [0, 1], color=BLUE, lw=1, linestyle="--")

    # Fill the full attainable region (solid)
    ax.fill_betweenx(
        xi_boundary,
        -psi_boundary, psi_boundary,
        color=FILL, alpha=0.7, zorder=0,
        label="Attainable region"
    )

    # ---------------- Dotted hatch on the side wings ----------------
    # We want to hatch only where |ψ| > ξ, i.e. to the right of ψ=ξ and left of ψ=-ξ.
    mask = psi_boundary >= xi_boundary  # this is True for all interior xi

    # Right wing: ψ from ξ to √ξ
    ax.fill_betweenx(
        xi_boundary,
        xi_boundary,       # left boundary of hatch
        psi_boundary,      # right boundary of hatch
        where=mask,
        facecolor='none',
        hatch='..',
        edgecolor=BLUE,
        linewidth=0,
        zorder=1,
        label="SI/SD region"
    )
    # Left wing: ψ from -√ξ to -ξ
    ax.fill_betweenx(
        xi_boundary,
        -psi_boundary,     # left boundary
        -xi_boundary,      # right boundary
        where=mask,
        facecolor='none',
        hatch='..',
        edgecolor=BLUE,
        linewidth=0,
        zorder=1
    )

    # ---------------------- Highlight key points ----------------------
    psi_max_diff = 0.5
    xi_max_diff  = psi_max_diff**2  # 0.25

    key_psi = [0,  1,  -1,  psi_max_diff, -psi_max_diff]
    key_xi  = [0,  1,   1,  xi_max_diff,  xi_max_diff]
    ax.scatter(key_psi, key_xi,  s=70, color="black", zorder=5, clip_on=False)

    # Annotations
    ax.annotate(r"$\Pi$",           ( 0, 0),  xytext=(  0, 15),
                textcoords="offset points", fontsize=18, ha="center")
    ax.annotate(r"$M$",             ( 1, 1),  xytext=(  -10, -15),
                textcoords="offset points", fontsize=18, ha="right", va="bottom")
    ax.annotate(r"$W$",             (-1, 1),  xytext=( 10, -15),
                textcoords="offset points", fontsize=18, ha="left",  va="bottom")
    ax.annotate(r"$C^{\mathrm{Fr}}_{1/2}$",  ( psi_max_diff, xi_max_diff),
                xytext=( 5, -15), textcoords="offset points",
                fontsize=18, ha="left")
    ax.annotate(r"$C^{\mathrm{Fr}}_{-1/2}$", (-psi_max_diff, xi_max_diff),
                xytext=(-45, -15), textcoords="offset points",
                fontsize=18, ha="left")

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Spearman's footrule $\psi$", fontsize=16)
    ax.set_ylabel(r"Chatterjee's $\xi$",         fontsize=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    ax.legend(loc="upper center", fontsize=12, frameon=True)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
