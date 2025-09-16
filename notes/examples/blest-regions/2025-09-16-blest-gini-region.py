#!/usr/bin/env python3
"""
Plot Blest’s ν versus Gini’s γ.

What’s exact here (per the paper):
- Lower boundary on γ ∈ [0, 1]:
    ν_min(γ) = γ - (2/9) * (1 - γ)^2, attained by the median–swap family C_δ
    with γ(C_δ) = 1 - 6 δ^2 and ν(C_δ) = 1 - 6 δ^2 - 8 δ^4, δ ∈ [0, 1/2].
- Upper boundary on γ ∈ [-1, 0]:
    ν_max(γ) = γ + (2/9) * (1 + γ)^2, attained by the survival copulas \hat C_δ.

The opposite halves (upper on γ∈[0,1] and lower on γ∈[-1,0]) are not fully
characterised in closed form in the article, so we do NOT shade the full region.
We plot the proven boundary arcs, the family curves, and key copulas (M, Π, W).

Usage:
  python3 plot_nu_vs_gini.py
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
#  Closed-form boundary pieces and family (from the paper)
# ----------------------------------------------------------------------
def nu_lower_pos(gamma: float) -> float:
    """Exact lower boundary ν_min(γ) for γ ∈ [0, 1]."""
    d = 1.0 - gamma
    return gamma - (2.0 / 9.0) * d * d


def nu_upper_neg(gamma: float) -> float:
    """Exact upper boundary ν_max(γ) for γ ∈ [-1, 0] (by survival symmetry)."""
    d = 1.0 + gamma
    return gamma + (2.0 / 9.0) * d * d


def gamma_from_delta(delta: float) -> float:
    """γ(C_δ) = 1 - 6 δ^2,   δ ∈ [0, 1/2]."""
    return 1.0 - 6.0 * delta * delta


def nu_from_delta(delta: float) -> float:
    """ν(C_δ) = 1 - 6 δ^2 - 8 δ^4,   δ ∈ [0, 1/2]."""
    d2 = delta * delta
    return 1.0 - 6.0 * d2 - 8.0 * d2 * d2


nu_lower_pos_vec = np.vectorize(nu_lower_pos)
nu_upper_neg_vec = np.vectorize(nu_upper_neg)
gamma_from_delta_vec = np.vectorize(gamma_from_delta)
nu_from_delta_vec = np.vectorize(nu_from_delta)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # γ grids for the proven boundary arcs
    gamma_pos = np.linspace(0.0, 1.0, 1201)
    gamma_neg = np.linspace(-1.0, 0.0, 1201)
    nu_low = nu_lower_pos_vec(gamma_pos)  # exact ν_min on [0,1]
    nu_up = nu_upper_neg_vec(gamma_neg)  # exact ν_max on [-1,0]

    # Median-swap family C_δ (traces the exact lower boundary on [0,1])
    delta = np.linspace(0.0, 0.5, 500)
    gamma_family = gamma_from_delta_vec(delta)
    nu_family = nu_from_delta_vec(delta)

    # Survival family \hat C_δ (upper boundary on [-1,0])
    gamma_surv = -gamma_family
    nu_surv = -nu_family

    # -------------------------- Plotting ------------------------------
    BLUE, RED, FILL = "#00529B", "#B03A2E", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Exact boundary arcs
    ax.plot(
        gamma_pos, nu_low, color=BLUE, lw=2.5, label=r"Exact lower boundary on $[0,1]$"
    )
    ax.plot(
        gamma_neg, nu_up, color=BLUE, lw=2.5, label=r"Exact upper boundary on $[-1,0]$"
    )

    # Family curves (same arcs, shown dashed)
    ax.plot(
        gamma_family,
        nu_family,
        color="black",
        lw=1.4,
        ls="--",
        label=r"Median-swap family $\{C_\delta\}$",
    )
    ax.plot(
        gamma_surv,
        nu_surv,
        color="black",
        lw=1.4,
        ls="--",
        label=r"Survival family $\{\widehat{C}_\delta\}$",
    )

    # ---------------------- Highlight key points ----------------------
    # Π: independence (γ=0, ν≈±2/9 are the proven extreme values at γ=0)
    ax.scatter([0], [0], s=60, color="black", zorder=5)
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(0, 20),
        textcoords="offset points",
        fontsize=18,
        ha="center",
        va="top",
    )

    # M (γ=1, ν=1) and W (γ=-1, ν=-1)
    ax.scatter([1], [1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(-10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )
    ax.scatter([-1], [-1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$W$",
        (-1, -1),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="bottom",
    )

    # Example point on the family: say γ=0 => δ = sqrt((1-0)/6) = 1/sqrt(6)
    delta0 = 1.0 / np.sqrt(6.0)
    gamma0 = gamma_from_delta(delta0)
    nu0 = nu_from_delta(delta0)  # = -2/9
    ax.scatter([gamma0], [nu0], s=60, color="black", zorder=5)
    ax.annotate(
        r"$C_{\delta=1/\sqrt{6}}$",
        (gamma0, nu0),
        xytext=(8, -5),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="top",
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Gini's $\gamma$", fontsize=16)
    ax.set_ylabel(r"Blest's $\nu$", fontsize=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    ax.legend(loc="lower right", fontsize=12, frameon=True)
    fig.tight_layout()
    pathlib.Path("images").mkdir(exist_ok=True)
    plt.savefig("images/blest-gini-boundary-arcs.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
