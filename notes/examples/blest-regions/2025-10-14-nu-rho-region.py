#!/usr/bin/env python3
"""
Plot Spearman's ρ (x-axis) versus Blest's ν (y-axis)  [clean version].

Upper boundary traced by the V-threshold family C_μ (μ ∈ [0, 2]).
Lower boundary: central symmetry ν_min(ρ) = −ν_max(−ρ).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import BLUE, FILL, apply_standard_axes, save_region_plot


# ── Boundary formulas ─────────────────────────────────────────────────────────

def rho_from_mu(mu: float) -> float:
    return 1.0 - mu**3 if mu <= 1.0 else -(mu**3) + 6*mu**2 - 12*mu + 7

def nu_from_mu(mu: float) -> float:
    return 1.0 - 0.75*mu**4 if mu <= 1.0 else -0.75*mu**4 + 4*mu**3 - 6*mu**2 + 3

rho_vec = np.vectorize(rho_from_mu)
nu_vec  = np.vectorize(nu_from_mu)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mu        = np.linspace(0.0, 2.0, 5001)
    sort_idx  = np.argsort(rho_vec(mu))
    rho_sorted = rho_vec(mu)[sort_idx]
    nu_sorted  = nu_vec(mu)[sort_idx]
    nu_lower   = -np.interp(-rho_sorted, rho_sorted, nu_sorted)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(rho_sorted, nu_sorted, color=BLUE, lw=2.0, label=r"Upper boundary ($\nu(\rho)$)")
    ax.plot(rho_sorted, nu_lower,  color=BLUE, lw=2.0)
    ax.fill_between(rho_sorted, nu_lower, nu_sorted,
                    color=FILL, alpha=0.8, zorder=0, label="Attainable region")

    apply_standard_axes(ax, r"Spearman's $\rho$", r"Blest's $\nu$")
    save_region_plot(fig, "images/nu-rho-region.png")


if __name__ == "__main__":
    main()
