#!/usr/bin/env python3
"""
Plot Spearman’s ρ versus Blest’s ν (axes swapped vs. the original script).

Same boundary data (V-threshold family C_μ), but we now place
  x-axis: ν,  y-axis: ρ.

Upper boundary is traced by the family C_μ (μ∈[0,2]):
  ρ(μ) = { 1 - μ^3,                               0 ≤ μ ≤ 1
         { -μ^3 + 6μ^2 - 12μ + 7,                 1 ≤ μ ≤ 2
  ν(μ) = { 1 - (3/4) μ^4,                         0 ≤ μ ≤ 1
         { -(3/4) μ^4 + 4μ^3 - 6μ^2 + 3,          1 ≤ μ ≤ 2

For ρ ∈ [0,1]:
  ν_max(ρ) = 1 - (3/4) * (1 - ρ)^(4/3)
Invert to overlay ρ(ν) on ν ∈ [0,1]:
  ρ(ν) = 1 - ((4/3) * (1 - ν))^(3/4)

Lower boundary is centrally symmetric:
  ν_min(ρ) = -ν_max(-ρ)  ⇒  ρ_max(ν) = -ρ_min(ν) when plotting ρ vs ν.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
#  Closed-form boundary (C_μ)
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


def rho_of_nu_pos(nu: float) -> float:
    """
    Inversion of the explicit upper boundary on ρ ∈ [0,1]:
      ν = 1 - (3/4)*(1-ρ)^(4/3)
    ⇒  1-ρ = ((4/3)*(1-ν))^(3/4)
    ⇒  ρ = 1 - ((4/3)*(1-ν))^(3/4)
    Valid for ν ∈ [0,1], gives the upper curve (ρ≥0).
    """
    return 1.0 - np.power((4.0 / 3.0) * (1.0 - nu), 3.0 / 4.0)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # --------- Parameterize the upper boundary via μ ∈ [0,2] ----------
    mu = np.linspace(0.0, 2.0, 5001)
    rho_mu = rho_vec(mu)
    nu_mu = nu_vec(mu)

    # Sort by ρ so we can also build the symmetric lower boundary there
    sort_idx = np.argsort(rho_mu)
    rho_sorted = rho_mu[sort_idx]  # y-values if we swap axes
    nu_sorted = nu_mu[sort_idx]  # x-values if we swap axes (upper boundary)

    # Lower boundary by central symmetry in (ρ,ν):
    #   ν_min(ρ) = -ν_max(-ρ).
    # We want ρ as function of ν, but for shading it's simpler to keep (y=ρ, x=ν)
    # and use fill_betweenx with x1=ν_lower(ρ), x2=ν_upper(ρ).
    nu_lower = -np.interp(-rho_sorted, rho_sorted, nu_sorted)

    # Explicit overlay (now as ρ(ν) for ν ∈ [0,1])
    nu_pos = np.linspace(0.0, 1.0, 600)
    rho_pos_explicit = rho_of_nu_pos(nu_pos)

    # ------------------------------ Plot ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Upper and lower boundary curves (now plotted as x=ν, y=ρ)
    ax.plot(
        nu_sorted,
        rho_sorted,
        color=BLUE,
        lw=2.5,
        label=r"Upper boundary (as $\rho(\nu)$)",
    )
    ax.plot(nu_lower, rho_sorted, color=BLUE, lw=2.0)

    # Fill attainable region between lower/upper (in x for each y)
    ax.fill_betweenx(
        rho_sorted,  # y
        nu_lower,  # x1 (lower)
        nu_sorted,  # x2 (upper)
        color=FILL,
        alpha=0.8,
        zorder=0,
        label="Attainable region",
    )

    # Overlay explicit ρ(ν) for ν∈[0,1] (should match the upper curve for ρ≥0)
    ax.plot(
        nu_pos,
        rho_pos_explicit,
        linestyle="--",
        lw=2.0,
        color="black",
        label=r"$\rho(\nu)=1-(\frac{4}{3}(1-\nu))^{3/4}\ \ (0\leq \nu\leq 1)$",
    )

    # -------------------------- Highlight points ----------------------
    # M (1,1) -> (ν,ρ)=(1,1); Π (0,0) -> (0,0); W (-1,-1) -> (-1,-1)
    # cusp at μ=1: (ρ,ν)=(0,0.25) -> (ν,ρ)=(0.25,0)
    key_nu = [1, 0, -1, 0.25]
    key_rho = [1, 0, -1, 0]
    ax.scatter(key_nu, key_rho, s=60, color="black", zorder=5)

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
        (0.25, 0),
        xytext=(8, -2),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="center",
    )

    # ------------------------ Axes & cosmetics ------------------------
    ax.set_xlabel(r"Blest's $\nu$", fontsize=16)
    ax.set_ylabel(r"Spearman's $\rho$", fontsize=16)
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
    plt.savefig("images/blest-vs-rho_axes-swapped.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
