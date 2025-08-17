#!/usr/bin/env python3
"""
Plots the exact attainable region for
Chatterjee's ξ and Spearman's Footrule ψ, with ξ on the x-axis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd


def calculate_lower_boundary(mu_values):
    """
    Calculates the (xi, psi) coordinates for the lower boundary based on the
    simplified parametric formulas in mu.
    Returns (xi_vals, psi_vals).
    """
    epsilon = 1e-12
    mu_values = np.maximum(mu_values, 0.5)

    v1 = 2.0 * mu_values / (2.0 * mu_values + 1.0)
    B = 1.0 / (2.0 * mu_values)
    safe_v1 = np.maximum(v1, epsilon)

    psi_vals = -2 * safe_v1**2 + 6 * safe_v1 - 5 + 1.0 / safe_v1

    B_sq = B**2
    poly_part = 4 * safe_v1**3 - 18 * safe_v1**2 + 36 * safe_v1 - 22
    log_part = -12 * np.log(safe_v1)
    b_term_poly = -4 * safe_v1**3 + 6 * safe_v1**2 - 1
    xi_vals = poly_part + log_part + B_sq * b_term_poly

    return xi_vals, psi_vals


def main() -> None:
    # ---------------- Boundary Definition ----------------
    # Upper boundary (y = sqrt(x))
    xi_upper_bound = np.linspace(0, 1, 500)
    psi_upper_bound = np.sqrt(xi_upper_bound)

    # Lower parametric boundary
    mu_vals = np.logspace(4, -4, 2000) + 0.5
    xi_lower_bound, psi_lower_bound = calculate_lower_boundary(mu_vals)
    xi_endpoint = 12 * np.log(2) - 8

    # Load the data for the conjectured lower boundary from your CSV file.
    conjecture_df = pd.read_csv("lower_boundary_final_smooth.csv")
    # ----------------------------------------------------

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the boundary lines
    ax.plot(
        xi_upper_bound, psi_upper_bound, color=BLUE, lw=2.5, zorder=3, label="Boundary"
    )
    # The solid blue theoretical lower boundary is plotted for reference
    ax.plot(xi_lower_bound, psi_lower_bound, color=BLUE, lw=2.5, zorder=3)
    ax.plot([xi_endpoint, 1], [-0.5, -0.5], color=BLUE, lw=2.5, zorder=3)

    # Plot the SI region boundary line
    ax.plot([0, 1], [0, 1], color=BLUE, lw=1, ls="--", zorder=2)

    # MODIFIED: The fill polygon is now constructed using your conjecture data
    # -----------------------------------------------------------------------------------------
    # The lower boundary of the fill is your conjecture curve, plus the horizontal line to xi=1
    fill_lower_x = np.concatenate([conjecture_df["xi"].values, [1]])
    fill_lower_y = np.concatenate([conjecture_df["psi"].values, [-0.5]])

    # Create the full polygon for the fill area
    # It goes forward along the top boundary, and then backward along our new lower boundary
    fill_poly_x = np.concatenate([xi_upper_bound, fill_lower_x[::-1]])
    fill_poly_y = np.concatenate([psi_upper_bound, fill_lower_y[::-1]])

    ax.fill(
        fill_poly_x,
        fill_poly_y,
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )
    # -----------------------------------------------------------------------------------------

    # Dotted hatch on the SI region
    xi_hatch = np.linspace(0, 1, 100)
    ax.fill_between(
        xi_hatch,
        xi_hatch,
        np.sqrt(xi_hatch),
        facecolor="none",
        hatch="..",
        edgecolor=BLUE,
        linewidth=0,
        zorder=1,
        label="SI region",
    )

    # Plot the loaded data as a distinct dashed line.
    ax.plot(
        conjecture_df["xi"],
        conjecture_df["psi"],
        # color="crimson",  # Color restored to crimson
        ls="--",
        lw=2.5,
        zorder=4,  # A higher zorder to draw it on top of the fill and blue line
        label="$(\\xi(C^*_{\mu}), \psi(C^*_{\mu}))$ for $\mu\geq 0$",
    )

    # ---------------------- Highlight key points ----------------------
    ax.scatter(
        [0, 1, 1, 0.25, 0.5], [0, 1, -0.5, 0.5, -0.5], s=70, color="black", zorder=5
    )
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
        xytext=(-2, -10),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )
    ax.annotate(
        r"$W$",
        (1, -0.5),
        xytext=(-2, 5),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="bottom",
    )
    ax.annotate(
        r"$C_{\#}$",
        (0.5, -0.5),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="bottom",
    )
    ax.annotate(
        r"$C_{\searrow}$",
        (xi_endpoint, -0.5),
        xytext=(-25, -15),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="bottom",
    )
    ax.annotate(
        r"$C^{\mathrm{Fr}}_{1/2}$",
        (0.25, 0.5),
        xytext=(-15, 15),
        textcoords="offset points",
        fontsize=18,
        ha="center",
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Chatterjee's $\xi$", fontsize=16)
    ax.set_ylabel(r"Spearman's footrule $\psi$", fontsize=16)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.55, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    ax.legend(loc="upper left", fontsize=12, frameon=True)
    fig.tight_layout()
    plt.savefig("attainable_xi_psi_region_final.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
