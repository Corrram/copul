#!/usr/bin/env python3
"""
Plot Spearman's ρ (y-axis) versus Blest's ν (x-axis).

Upper boundary is traced by the V-threshold family C_μ (μ ∈ [0, 2]):
    ρ(μ) = 1 − μ³  (μ≤1),   −μ³ + 6μ² − 12μ + 7  (μ>1)
    ν(μ) = 1 − ¾μ⁴ (μ≤1),   −¾μ⁴ + 4μ³ − 6μ² + 3  (μ>1)
Explicit closed form on ρ ∈ [0,1]:  ν_max(ρ) = 1 − (3/4)·(1−ρ)^(4/3)
Lower boundary: central symmetry ν_min(ρ) = −ν_max(−ρ).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import BLUE, FILL, apply_standard_axes, mark_key_copulas, save_region_plot, scatter_annotate


# ── Boundary and family formulas ─────────────────────────────────────────────

def rho_from_mu(mu: float) -> float:
    return 1.0 - mu**3 if mu <= 1.0 else -(mu**3) + 6*mu**2 - 12*mu + 7

def nu_from_mu(mu: float) -> float:
    return 1.0 - 0.75*mu**4 if mu <= 1.0 else -0.75*mu**4 + 4*mu**3 - 6*mu**2 + 3

rho_vec = np.vectorize(rho_from_mu)
nu_vec  = np.vectorize(nu_from_mu)

def rho_of_nu_pos(nu: float) -> float:
    """Inversion of ν = 1 − (3/4)·(1−ρ)^(4/3) for ν ∈ [0,1]."""
    return 1.0 - np.power((4.0 / 3.0) * (1.0 - nu), 3.0 / 4.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mu = np.linspace(0.0, 2.0, 5001)
    sort_idx   = np.argsort(rho_vec(mu))
    rho_sorted = rho_vec(mu)[sort_idx]
    nu_sorted  = nu_vec(mu)[sort_idx]
    nu_lower   = -np.interp(-rho_sorted, rho_sorted, nu_sorted)

    # Explicit closed-form overlay on [0,1]
    nu_pos = np.linspace(0.0, 1.0, 600)
    rho_pos_explicit = rho_of_nu_pos(nu_pos)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(nu_sorted, rho_sorted, color=BLUE, lw=2.5,
            label=r"Upper boundary (as $\rho(\nu)$)")
    ax.plot(nu_lower, rho_sorted, color=BLUE, lw=2.0)
    ax.fill_betweenx(rho_sorted, nu_lower, nu_sorted,
                     color=FILL, alpha=0.8, zorder=0, label="Attainable region")
    ax.plot(nu_pos, rho_pos_explicit, ls="--", lw=2.0, color="black",
            label=r"$\rho(\nu)=1-(\frac{4}{3}(1-\nu))^{3/4}\ \ (0\leq \nu\leq 1)$")

    mark_key_copulas(ax)
    # Cusp at μ=1: (ρ,ν)=(0,0.25) → x=0.25, y=0
    scatter_annotate(ax, 0.25, 0.0, r"$\mu=1$", xytext=(8, -2), ha="left", va="center", fontsize=16)

    apply_standard_axes(ax, r"Blest's $\nu$", r"Spearman's $\rho$")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    save_region_plot(fig, "images/blest-vs-rho_axes-swapped.png")


if __name__ == "__main__":
    main()
