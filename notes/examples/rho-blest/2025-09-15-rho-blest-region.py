#!/usr/bin/env python3
"""
Plot Blest’s ν versus Spearman’s ρ
with the attainable region correctly shaded,
and key copulas (M, Π, W) and the cusp at μ=1 marked.

Upper boundary is traced by the V-threshold family C_μ (μ∈[0,2]):
  ρ(μ) = { 1 - μ^3,                               0 ≤ μ ≤ 1
         { -μ^3 + 6μ^2 - 12μ + 7,                 1 ≤ μ ≤ 2
  ν(μ) = { 1 - (3/4) μ^4,                         0 ≤ μ ≤ 1
         { -(3/4) μ^4 + 4μ^3 - 6μ^2 + 3,          1 ≤ μ ≤ 2

For ρ ∈ [0,1] the boundary admits the explicit law
  ν_max(ρ) = 1 - (3/4) * (1 - ρ)^(4/3).

The lower boundary is centrally symmetric:
  ν_min(ρ) = -ν_max(-ρ).
"""
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
#  Closed-form boundary (C_μ) and explicit ν(ρ) on [0,1]
# ----------------------------------------------------------------------
def rho_from_mu(mu: float) -> float:
    """Spearman's ρ for C_μ (piecewise)."""
    return 1.0 - mu**3 if mu <= 1.0 else (-(mu**3) + 6 * mu**2 - 12 * mu + 7)


def nu_from_mu(mu: float) -> float:
    """Blest's ν for C_μ (piecewise)."""
    return (
        1.0 - 0.75 * mu**4
        if mu <= 1.0
        else (-(0.75) * mu**4 + 4 * mu**3 - 6 * mu**2 + 3)
    )


rho_vec, nu_vec = np.vectorize(rho_from_mu), np.vectorize(nu_from_mu)


def nu_max_pos_from_rho_pos(rho: float) -> float:
    """Explicit boundary for ρ ∈ [0,1]: ν_max(ρ) = 1 - (3/4)*(1-ρ)^(4/3)."""
    return 1.0 - 0.75 * (1.0 - rho) ** (4.0 / 3.0)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # --------- Parameterize the upper boundary via μ ∈ [0,2] ----------
    mu = np.linspace(0.0, 2.0, 5001)
    rho_mu = rho_vec(mu)
    nu_mu = nu_vec(mu)

    # Sort by ρ so we can regard the boundary as a function ρ ↦ ν_max
    sort_idx = np.argsort(rho_mu)
    rho_sorted = rho_mu[sort_idx]
    nu_sorted = nu_mu[sort_idx]  # this is ν_max(ρ) on [-1, 1]

    # Lower boundary by central symmetry: ν_min(ρ) = -ν_max(-ρ)
    nu_lower = -np.interp(-rho_sorted, rho_sorted, nu_sorted)

    # For visual confirmation on [0,1], overlay the explicit law
    rho_pos = np.linspace(0.0, 1.0, 600)
    nu_pos_explicit = nu_max_pos_from_rho_pos(rho_pos)

    # ------------------------------ Plot ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Upper and lower boundary curves
    ax.plot(
        rho_sorted,
        nu_sorted,
        color=BLUE,
        lw=2.5,
        label=r"Upper boundary $\nu_{\max}(\rho)$",
    )
    ax.plot(rho_sorted, nu_lower, color=BLUE, lw=2.0)

    # Fill attainable region
    ax.fill_between(
        rho_sorted,
        nu_lower,
        nu_sorted,
        color=FILL,
        alpha=0.8,
        zorder=0,
        label="Attainable region",
    )

    # Overlay explicit ν_max(ρ) for ρ∈[0,1] (should coincide with the upper curve on the right half)
    ax.plot(
        rho_pos,
        nu_pos_explicit,
        linestyle="--",
        lw=2.0,
        color="black",
        label=r"$\nu_{\max}(\rho)=1-\frac{3}{4}(1-\rho)^{4/3}\ \ (0\leq \rho\leq 1)$",
    )

    # -------------------------- Highlight points ----------------------
    # M (comonotone), Π (independence), W (countermonotone)
    key_rho = [1, 0, -1, 0]
    key_nu = [1, 0, -1, 0.25]  # last one is cusp at μ=1
    ax.scatter(key_rho, key_nu, s=60, color="black", zorder=5)

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
        r"$\Pi$",
        (0, 0),
        xytext=(0, 18),
        textcoords="offset points",
        fontsize=18,
        ha="center",
        va="bottom",
    )
    ax.annotate(
        r"$W$",
        (-1, -1),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="bottom",
    )
    ax.annotate(
        r"$\mu=1$",
        (0, 0.25),
        xytext=(8, -2),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="center",
    )

    # ------------------------ Axes & cosmetics ------------------------
    ax.set_xlabel(r"Spearman's $\rho$", fontsize=16)
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
    plt.savefig("images/rho-blest-region.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
