#!/usr/bin/env python3
"""
Plots the exact attainable region for
Chatterjee's ξ and Spearman's Footrule ψ.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def calculate_lower_boundary(mu_values):
    """
    Calculates the (xi, psi) coordinates for the lower boundary based on the
    simplified parametric formulas in mu.

    NOTE: In this script, psi is the x-axis and xi is the y-axis.
    This function returns (psi_vals, xi_vals).
    """
    epsilon = 1e-12
    # Ensure mu is positive to avoid division by zero or log of zero.
    mu_values = np.maximum(mu_values, epsilon)

    # The simplified formulas depend primarily on v1 and B.
    v1 = 2.0 * mu_values / (2.0 * mu_values + 1.0)
    B = 1.0 / (2.0 * mu_values)

    # Use a safe version of v1 to prevent division by zero or log(0).
    safe_v1 = np.maximum(v1, epsilon)

    # --- Calculate psi(mu) using the simplified formula ---
    psi_vals = -2 * safe_v1 ** 2 + 6 * safe_v1 - 5 + 1.0 / safe_v1

    # --- Calculate xi(mu) using the simplified formula ---
    B_sq = B ** 2

    polynomial_part = 4 * safe_v1 ** 3 - 18 * safe_v1 ** 2 + 36 * safe_v1 - 22
    log_part = -12 * np.log(safe_v1)
    b_term_polynomial = -4 * safe_v1 ** 3 + 6 * safe_v1 ** 2 - 1

    xi_vals = polynomial_part + log_part + B_sq * b_term_polynomial

    return psi_vals, xi_vals


def main() -> None:
    # ---------------- Boundary Definition ----------------
    # Right-side boundary (parabola)
    psi_pos_bound = np.linspace(0, 1, 500)
    xi_pos_bound = psi_pos_bound ** 2

    # Left-side boundary (parametric curve)
    mu_vals = np.logspace(4, -4, 2000) + 1/2
    # mu_vals = mu_vals[mu_vals > 0.002]
    psi_lower_raw, xi_lower_raw = calculate_lower_boundary(mu_vals)

    # Filter to only include psi >= -0.5
    mask = (0 >= psi_lower_raw) & (psi_lower_raw >= -0.5)
    psi_neg_bound = psi_lower_raw[mask]
    xi_neg_bound = xi_lower_raw[mask]
    # # Sort by psi for correct plotting
    # sort_indices = np.argsort(psi_neg_bound)
    # psi_neg_bound = psi_neg_bound[sort_indices]
    # xi_neg_bound = xi_neg_bound[sort_indices]

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the boundary envelope
    ax.plot(psi_pos_bound, xi_pos_bound, color=BLUE, lw=2.5, zorder=3, label=r"Boundary")
    ax.plot(psi_neg_bound, xi_neg_bound, color=BLUE, lw=2.5, zorder=3)

    # Add the diagonal line for the SI region
    ax.plot([0, 1], [0, 1], color=BLUE, lw=1, ls="--", zorder=2)

    # Fill the attainable region (solid)
    # Right Lobe (SI region): Between ξ=ψ² and ξ=ψ
    ax.fill_between(psi_pos_bound, psi_pos_bound ** 2, 1,
                    color=FILL, alpha=0.7, zorder=0, label="Attainable region")
    # Left Lobe: Between ξ=0 and the new upper boundary
    ax.fill_between(psi_neg_bound, xi_neg_bound, 1, color=FILL, alpha=0.7, zorder=0)

    # Dotted hatch on the SI wing
    ax.fill_between(psi_pos_bound, psi_pos_bound ** 2, psi_pos_bound,
                    facecolor='none', hatch='..', edgecolor=BLUE,
                    linewidth=0, zorder=1, label="SI region")

    # ---------------------- Highlight key points ----------------------
    xi_endpoint = 12 * np.log(2) - 8
    ax.scatter([-0.5], [xi_endpoint], s=80, color="black", zorder=5)
    ax.scatter([0, 1, -0.5, 0.5], [0, 1, 1, 0.25], s=70, color="black", zorder=5)

    # Annotations
    ax.annotate(r"$\Pi$", (0, 0), xytext=(0, 15), textcoords="offset points",
                fontsize=18, ha="center")
    ax.annotate(r"$M$", (1, 1), xytext=(-10, -2), textcoords="offset points",
                fontsize=18, ha="right", va="top")
    ax.annotate(r"$W$", (-0.5, 1), xytext=(10, -2), textcoords="offset points",
                fontsize=18, ha="left", va="top")
    ax.annotate(r"$C_{*}$", (-0.5, xi_endpoint), xytext=(10, 5),
                textcoords="offset points", fontsize=18, ha="left", va="bottom", clip_on=False)
    ax.annotate(r"$C^{\mathrm{Fr}}_{1/2}$", (0.5, 0.25),
                xytext=(5, -15), textcoords="offset points",
                fontsize=18, ha="left")

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Spearman's footrule $\psi$", fontsize=16)
    ax.set_ylabel(r"Chatterjee's $\xi$", fontsize=16)
    ax.set_xlim(-0.55, 1.05)
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
    plt.savefig("attainable_xi_psi_region.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()