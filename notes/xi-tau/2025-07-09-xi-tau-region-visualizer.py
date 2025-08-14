#!/usr/bin/env python3
"""
Plot Chatterjee’s ξ versus Kendall’s τ
with the attainable region correctly shaded
and the extrema C_{±b_0} marked.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from math import sqrt


# ----------------------------------------------------------------------
#  Closed-form measures (Prop. 1)
# ----------------------------------------------------------------------
def xi_from_b(b: float) -> float:
    """Chatterjee’s ξ(C_b) – even in b."""
    ab = abs(b)
    return (b * b / 10) * (5 - 2 * ab) if ab <= 1 else 1 - 1 / ab + 3 / (10 * ab * ab)


def tau_from_b(b: float) -> float:
    """Kendall’s τ(C_b) – odd in b."""
    if b >= 0:
        return (b * (4 - b)) / 6 if b <= 1 else (6 * b * b - 4 * b + 1) / (6 * b * b)
    return -tau_from_b(-b)


xi_vec, tau_vec = map(np.vectorize, (xi_from_b, tau_from_b))


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # ---------------- Parameter grid (positive branch) ----------------
    b_pos = np.hstack(
        [
            np.linspace(0.0, 1.0, 600, endpoint=False)[1:],  # fine mesh 0<b≤1
            np.linspace(1.0, 20.0, 1400),  # stretch to large b
        ]
    )
    xi_pos, tau_pos = xi_vec(b_pos), tau_vec(b_pos)

    # Sort the positive branch by ξ so fill_betweenx works monotonically
    sort_idx = np.argsort(xi_pos)
    xi_sorted = xi_pos[sort_idx]
    tau_sorted_pos = tau_pos[sort_idx]

    # ------------- Extend envelope to ξ=1 with |τ| = 1 ----------------
    xi_ceiling = np.linspace(xi_sorted[-1], 1.0, 200)
    tau_ceiling = np.ones_like(xi_ceiling)  # full width |τ|=1

    # Complete envelope for  ξ∈[0,1]
    np.concatenate([xi_sorted, xi_ceiling[1:]])
    np.concatenate([tau_sorted_pos, tau_ceiling[1:]])

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Envelope curve ±τ(ξ) including the ceiling segments out to |τ|=1
    # smooth‐closing envelope:
    # take just the sorted parametric branch and tack on the corner (1,1)
    xi_plot = np.concatenate([xi_sorted, [1.0]])
    tau_plot_pos = np.concatenate([tau_sorted_pos, [1.0]])
    ax.plot(tau_plot_pos, xi_plot, color=BLUE, lw=2.5, label=r"$\pm\tau(\xi)$")
    ax.plot(-tau_plot_pos, xi_plot, color=BLUE, lw=2.5)

    # Fill attainable region (up to |τ| = 1 when ξ ≥ ξ_max(param))
    ax.fill_betweenx(
        xi_plot,
        -tau_plot_pos,
        tau_plot_pos,
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # Hatch |τ| > ξ sub-region
    # mask = tau_env > xi_env
    # ax.fill_betweenx(
    #     xi_env[mask],  xi_env[mask],  tau_env[mask],
    #     facecolor="none", hatch="..", edgecolor=BLUE, linewidth=0
    # )
    # ax.fill_betweenx(
    #     xi_env[mask], -tau_env[mask], -xi_env[mask],
    #     facecolor="none", hatch="..", edgecolor=BLUE, linewidth=0
    # )

    # ---------------------- Highlight key points ----------------------
    # Extremal difference b0:
    b0 = (10 - sqrt(10)) / 9  # ≈ 0.759
    xi0 = xi_from_b(b0)
    tau0 = tau_from_b(b0)

    key_tau = [0, 1, -1, tau0, -tau0]
    key_xi = [0, 1, 1, xi0, xi0]
    ax.scatter(key_tau, key_xi, s=60, color="black", zorder=5)

    # Labels
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(0, 20),
        textcoords="offset points",
        fontsize=18,
        ha="center",
        va="top",
    )
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(-10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )
    ax.annotate(
        r"$W$",
        (-1, 1),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="top",
    )
    ax.annotate(
        r"$C_{b_0}$",
        (tau0, xi0),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="top",
    )
    ax.annotate(
        r"$C_{-b_0}$",
        (-tau0, xi0),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Kendall's $\tau$", fontsize=16)
    ax.set_ylabel(r"Chatterjee's $\xi$", fontsize=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)

    ax.legend(loc="center", fontsize=12, frameon=True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
