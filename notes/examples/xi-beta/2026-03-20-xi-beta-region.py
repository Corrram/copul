#!/usr/bin/env python3
"""
Plot Chatterjee's ξ (x-axis) versus Blomqvist's β (y-axis)
with the attainable region correctly shaded
and key copulas (M, Π, W) and boundary copulas (C#) marked.

The exact region is given by β^2 <= 2ξ, intersected with [0,1] x [-1,1].
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator


def main() -> None:
    # Generate a dense grid for ξ (x-axis)
    xi = np.linspace(0, 1.0, 2000)

    # Calculate the upper and lower boundaries for β (y-axis)
    beta_max = np.clip(np.sqrt(2 * xi), -1.0, 1.0)
    beta_min = np.clip(-np.sqrt(2 * xi), -1.0, 1.0)

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the boundary envelopes
    ax.plot(xi, beta_max, color=BLUE, lw=2.5, label=r"$\beta_{\max}(\xi)$")
    ax.plot(xi, beta_min, color=BLUE, lw=2.5, label=r"$\beta_{\min}(\xi)$")

    # Shade the attainable region
    ax.fill_between(
        xi,
        beta_min,
        beta_max,
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # ---------------------- Highlight key points ----------------------
    # Independence Π: (ξ, β) = (0, 0)
    # Moved to the right, deeper into the parabola
    ax.scatter([0], [0], s=60, color="black", zorder=5)
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="center",
    )

    # Comonotonicity M: (ξ, β) = (1, 1)
    # Pushed diagonally down and left into the blue area
    ax.scatter([1], [1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(-10, -10),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )

    # Countermonotonicity W: (ξ, β) = (1, -1)
    # Pushed diagonally up and left into the blue area
    ax.scatter([1], [-1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$W$",
        (1, -1),
        xytext=(-10, 10),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="bottom",
    )

    # Mark the extreme bounds of the parabola
    # Pushed straight down into the blue area
    ax.scatter([0.5], [1.0], s=60, color="black", zorder=6)
    ax.annotate(
        r"$C_{1}^{\#}$",
        (0.5, 1.0),
        xytext=(0, -10),
        textcoords="offset points",
        fontsize=16,
        ha="center",
        va="top",
    )

    # Pushed straight up into the blue area
    ax.scatter([0.5], [-1.0], s=60, color="black", zorder=6)
    ax.annotate(
        r"$C_{-1}^{\#}$",
        (0.5, -1.0),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=16,
        ha="center",
        va="bottom",
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Chatterjee's $\xi$", fontsize=16)
    ax.set_ylabel(r"Blomqvist's $\beta$", fontsize=16)

    # Restored to tight slim limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    ax.set_aspect(0.5, adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    fig.tight_layout()
    Path("images").mkdir(exist_ok=True)
    plt.savefig("images/xi-beta-region.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()