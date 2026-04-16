#!/usr/bin/env python3
"""
Plot Gini's γ (x-axis) versus Blest's ν (y-axis).

Exact boundaries:
  ν_min(γ) = γ − (2/9)·(1−γ)²   for γ ∈ [0, 1]   (attained by C_δ family)
  ν_max(γ) = γ + (2/9)·(1+γ)²   for γ ∈ [−1, 0]  (attained by survival Ĉ_δ)
Median-swap family C_δ (δ ∈ [0, 1/2]):
    γ(δ) = 1 − 6δ²,   ν(δ) = 1 − 6δ² − 8δ⁴.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import BLUE, FILL, apply_standard_axes, mark_key_copulas, save_region_plot, scatter_annotate


# ── Boundary and family formulas ─────────────────────────────────────────────

def nu_lower_pos(gamma: float) -> float:
    return gamma - (2.0 / 9.0) * (1.0 - gamma)**2

def nu_upper_neg(gamma: float) -> float:
    return gamma + (2.0 / 9.0) * (1.0 + gamma)**2

def gamma_from_delta(delta: float) -> float:
    return 1.0 - 6.0 * delta**2

def nu_from_delta(delta: float) -> float:
    d2 = delta**2
    return 1.0 - 6.0 * d2 - 8.0 * d2**2

nu_lp_v    = np.vectorize(nu_lower_pos)
nu_un_v    = np.vectorize(nu_upper_neg)
gamma_dv   = np.vectorize(gamma_from_delta)
nu_dv      = np.vectorize(nu_from_delta)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gamma_pos = np.linspace(0.0,  1.0, 1201)
    gamma_neg = np.linspace(-1.0, 0.0, 1201)
    nu_low = nu_lp_v(gamma_pos)
    nu_up  = nu_un_v(gamma_neg)

    delta        = np.linspace(0.0, 0.5, 500)
    gamma_family = gamma_dv(delta)
    nu_family    = nu_dv(delta)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(gamma_pos, nu_low, color=BLUE, lw=2.5, label=r"Exact lower boundary on $[0,1]$")
    ax.plot(gamma_neg, nu_up,  color=BLUE, lw=2.5, label=r"Exact upper boundary on $[-1,0]$")
    ax.plot(gamma_family,  nu_family,  color="black", lw=1.4, ls="--",
            label=r"Median-swap family $\{C_\delta\}$")
    ax.plot(-gamma_family, -nu_family, color="black", lw=1.4, ls="--",
            label=r"Survival family $\{\widehat{C}_\delta\}$")

    mark_key_copulas(ax, Pi_xytext=(0, 20), Pi_va="top")

    # Example: γ=0 ↔ δ=1/√6
    delta0 = 1.0 / np.sqrt(6.0)
    scatter_annotate(ax, gamma_from_delta(delta0), nu_from_delta(delta0),
                     r"$C_{\delta=1/\sqrt{6}}$", xytext=(8, -5), ha="left", va="top", fontsize=16)

    apply_standard_axes(ax, r"Gini's $\gamma$", r"Blest's $\nu$")
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    save_region_plot(fig, "images/blest-gini-boundary-arcs.png")


if __name__ == "__main__":
    main()
