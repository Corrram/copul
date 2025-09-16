#!/usr/bin/env python3
"""
Plot Blest’s ν versus Kendall’s τ.

What’s exact here (per the τ–ν article):
- Lower boundary on τ ∈ [0, 1] is the lower envelope of two closed-form arcs:
    (i) Chevron arc (median–swap optimisers C_δ):
        τ(C_δ) = 1 - 4 δ^2,
        ν(C_δ) = 1 - 12 δ^3,
        so ν = 1 - (3/2) * (1 - τ)^{3/2}.
    (ii) Corner arc (end–swap optimisers C_d):
        τ(C_d) = 1 - 8 d (1 - d),
        ν(C_d) = 1 - 12 d + 24 d^2 - 16 d^3,
        so ν = 2 * ((1 + τ)/2)^{3/2} - 1.
  The two arcs intersect once at τ* ≈ 0.20333699, where ν_min(τ*) ≈ -0.0666041.

- Upper boundary on τ ∈ [-1, 0] follows by survival symmetry:
    ν_max(τ) = - ν_min(-τ).

The opposite halves (upper on τ∈[0,1] and lower on τ∈[-1,0]) are not fully
characterised in closed form here, so we do NOT shade the full region.
We plot the proven boundary arcs, the family curves, and key copulas (M, Π, W).

Usage:
  python3 plot_nu_vs_kendall.py
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
#  Closed-form boundary pieces and families
# ----------------------------------------------------------------------
def nu_chevron_of_tau(tau: float) -> float:
    """Chevron arc: ν = 1 - (3/2) * (1 - τ)^{3/2}, valid for τ ∈ [0, 1]."""
    return 1.0 - 1.5 * np.power(1.0 - tau, 1.5)


def nu_corner_of_tau(tau: float) -> float:
    """Corner arc: ν = 2 * ((1 + τ)/2)^{3/2} - 1, valid for τ ∈ [0, 1]."""
    return 2.0 * np.power((1.0 + tau) / 2.0, 1.5) - 1.0


def nu_lower_pos(tau: float) -> float:
    """Exact lower boundary ν_min(τ) for τ ∈ [0, 1]: lower envelope of the two arcs."""
    return min(nu_chevron_of_tau(tau), nu_corner_of_tau(tau))


def nu_upper_neg(tau: float) -> float:
    """Exact upper boundary ν_max(τ) for τ ∈ [-1, 0] by survival symmetry."""
    return -nu_lower_pos(-tau)


# Families (to overlay as dashed curves)
def tau_from_delta(delta: float) -> float:
    """Chevron family: τ(C_δ) = 1 - 4 δ^2,   δ ∈ [0, 1/2]."""
    return 1.0 - 4.0 * delta * delta


def nu_from_delta(delta: float) -> float:
    """Chevron family: ν(C_δ) = 1 - 12 δ^3,   δ ∈ [0, 1/2]."""
    return 1.0 - 12.0 * delta * delta * delta


def tau_from_d(d: float) -> float:
    """Corner family: τ(C_d) = 1 - 8 d (1 - d),   d ∈ [0, 1/2]."""
    return 1.0 - 8.0 * d * (1.0 - d)


def nu_from_d(d: float) -> float:
    """Corner family: ν(C_d) = 1 - 12 d + 24 d^2 - 16 d^3,   d ∈ [0, 1/2]."""
    return 1.0 - 12.0 * d + 24.0 * d * d - 16.0 * d * d * d


nu_chevron_vec = np.vectorize(nu_chevron_of_tau)
nu_corner_vec = np.vectorize(nu_corner_of_tau)
nu_lower_pos_vec = np.vectorize(nu_lower_pos)
nu_upper_neg_vec = np.vectorize(nu_upper_neg)
tau_from_delta_vec = np.vectorize(tau_from_delta)
nu_from_delta_vec = np.vectorize(nu_from_delta)
tau_from_d_vec = np.vectorize(tau_from_d)
nu_from_d_vec = np.vectorize(nu_from_d)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # τ grids for the proven boundary arcs
    tau_pos = np.linspace(0.0, 1.0, 1601)
    tau_neg = np.linspace(-1.0, 0.0, 1601)

    # Exact boundaries
    nu_low = nu_lower_pos_vec(tau_pos)  # exact ν_min on [0,1]
    nu_up = nu_upper_neg_vec(tau_neg)  # exact ν_max on [-1,0]

    # Individual arcs on [0,1] (to show the envelope composition)
    nu_chev = nu_chevron_vec(tau_pos)
    nu_corn = nu_corner_vec(tau_pos)

    # Families:
    # Chevron family (δ in [0,1/2]) traces the chevron arc on [0,1]
    delta = np.linspace(0.0, 0.5, 500)
    tau_chev_family = tau_from_delta_vec(delta)
    nu_chev_family = nu_from_delta_vec(delta)

    # Corner family (d in [0,1/2]) traces the corner arc on [0,1]
    d = np.linspace(0.0, 0.5, 500)
    tau_corner_family = tau_from_d_vec(d)
    nu_corner_family = nu_from_d_vec(d)

    # Survival images of the families (mirror to negative τ)
    tau_chev_surv = -tau_chev_family
    nu_chev_surv = -nu_chev_family
    tau_corner_surv = -tau_corner_family
    nu_corner_surv = -nu_corner_family

    # Locate the switch point τ* on [0,1] where the two arcs meet (for annotation)
    diff = np.abs(nu_chev - nu_corn)
    idx = int(np.argmin(diff))
    tau_star = tau_pos[idx]
    nu_star = nu_low[idx]

    # -------------------------- Plotting ------------------------------
    BLUE, RED, GREY = "#00529B", "#B03A2E", "#6C757D"
    fig, ax = plt.subplots(figsize=(8.2, 6.3))

    # Exact boundary arcs
    ax.plot(
        tau_pos, nu_low, color=BLUE, lw=2.6, label=r"Exact lower boundary on $[0,1]$"
    )
    ax.plot(
        tau_neg, nu_up, color=BLUE, lw=2.6, label=r"Exact upper boundary on $[-1,0]$"
    )

    # Show each arc composing the lower envelope (thin)
    ax.plot(tau_pos, nu_chev, color=GREY, lw=1.2, ls=":", label="Chevron arc")
    ax.plot(tau_pos, nu_corn, color=GREY, lw=1.2, ls="--", label="Corner arc")

    # Families (dashed)
    ax.plot(
        tau_chev_family,
        nu_chev_family,
        color="black",
        lw=1.4,
        ls="--",
        label=r"Median-swap family $\{C_\delta\}$",
    )
    ax.plot(
        tau_corner_family,
        nu_corner_family,
        color="black",
        lw=1.4,
        ls="--",
        label=r"End-swap family $\{C_d\}$",
    )

    # Survival family overlays (negative side, dashed)
    ax.plot(tau_chev_surv, nu_chev_surv, color="black", lw=1.0, ls="--")
    ax.plot(tau_corner_surv, nu_corner_surv, color="black", lw=1.0, ls="--")

    # ---------------------- Highlight key points ----------------------
    # Π: independence (τ=0, ν=0)
    ax.scatter([0], [0], s=60, color="black", zorder=5)
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(0, 18),
        textcoords="offset points",
        fontsize=16,
        ha="center",
        va="bottom",
    )

    # M (τ=1, ν=1) and W (τ=-1, ν=-1)
    ax.scatter([1], [1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(-10, -4),
        textcoords="offset points",
        fontsize=16,
        ha="right",
        va="top",
    )
    ax.scatter([-1], [-1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$W$",
        (-1, -1),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="bottom",
    )

    # Switch point annotation
    ax.scatter([tau_star], [nu_star], s=55, color=RED, zorder=6)
    ax.annotate(
        r"$\tau_\ast\!\approx\!{:.4f}$".format(tau_star),
        (tau_star, nu_star),
        xytext=(10, -12),
        textcoords="offset points",
        fontsize=12,
        ha="left",
        va="top",
        color=RED,
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Kendall's $\tau$", fontsize=16)
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

    ax.legend(loc="lower right", fontsize=11, frameon=True)
    fig.tight_layout()
    pathlib.Path("images").mkdir(exist_ok=True)
    plt.savefig("images/blest-vs-kendall-boundary-arcs.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
