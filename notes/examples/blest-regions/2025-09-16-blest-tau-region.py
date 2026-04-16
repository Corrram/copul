#!/usr/bin/env python3
"""
Plot Kendall's τ (x-axis) versus Blest's ν (y-axis).

Exact boundary arcs:
  Lower boundary on τ ∈ [0,1] is the lower envelope of:
    (i) Chevron arc  C_δ:  ν = 1 − (3/2)·(1−τ)^(3/2)
    (ii) Corner arc  C_d:  ν = 2·((1+τ)/2)^(3/2) − 1
  The two arcs meet at τ* ≈ 0.20334.
  Upper boundary on τ ∈ [−1,0] by survival symmetry: ν_max(τ) = −ν_min(−τ).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import BLUE, apply_standard_axes, mark_key_copulas, save_region_plot, scatter_annotate


# ── Boundary formulas ─────────────────────────────────────────────────────────

def nu_chevron(tau: float) -> float:
    return 1.0 - 1.5 * (1.0 - tau)**1.5

def nu_corner(tau: float) -> float:
    return 2.0 * ((1.0 + tau) / 2.0)**1.5 - 1.0

def nu_lower_pos(tau: float) -> float:
    return min(nu_chevron(tau), nu_corner(tau))

def nu_upper_neg(tau: float) -> float:
    return -nu_lower_pos(-tau)

def tau_from_delta(delta: float) -> float:
    return 1.0 - 4.0 * delta**2

def nu_from_delta(delta: float) -> float:
    return 1.0 - 12.0 * delta**3

def tau_from_d(d: float) -> float:
    return 1.0 - 8.0 * d * (1.0 - d)

def nu_from_d(d: float) -> float:
    return 1.0 - 12.0*d + 24.0*d**2 - 16.0*d**3

nu_chev_v    = np.vectorize(nu_chevron)
nu_corn_v    = np.vectorize(nu_corner)
nu_low_p_v   = np.vectorize(nu_lower_pos)
nu_up_n_v    = np.vectorize(nu_upper_neg)
tau_delta_v  = np.vectorize(tau_from_delta)
nu_delta_v   = np.vectorize(nu_from_delta)
tau_d_v      = np.vectorize(tau_from_d)
nu_d_v       = np.vectorize(nu_from_d)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    GREY = "#6C757D"
    RED  = "#B03A2E"

    tau_pos = np.linspace(0.0,  1.0, 1601)
    tau_neg = np.linspace(-1.0, 0.0, 1601)
    nu_low  = nu_low_p_v(tau_pos)
    nu_up   = nu_up_n_v(tau_neg)
    nu_chev = nu_chev_v(tau_pos)
    nu_corn = nu_corn_v(tau_pos)

    delta = np.linspace(0.0, 0.5, 500)
    d     = np.linspace(0.0, 0.5, 500)
    tau_chev_fam  = tau_delta_v(delta);  nu_chev_fam  = nu_delta_v(delta)
    tau_corn_fam  = tau_d_v(d);          nu_corn_fam  = nu_d_v(d)

    # Intersection τ* of the two arcs
    idx     = int(np.argmin(np.abs(nu_chev - nu_corn)))
    tau_star, nu_star = tau_pos[idx], nu_low[idx]

    fig, ax = plt.subplots(figsize=(8.2, 6.3))

    ax.plot(tau_pos, nu_low,  color=BLUE, lw=2.6, label=r"Exact lower boundary on $[0,1]$")
    ax.plot(tau_neg, nu_up,   color=BLUE, lw=2.6, label=r"Exact upper boundary on $[-1,0]$")
    ax.plot(tau_pos, nu_chev, color=GREY, lw=1.2, ls=":",  label="Chevron arc")
    ax.plot(tau_pos, nu_corn, color=GREY, lw=1.2, ls="--", label="Corner arc")
    ax.plot(tau_chev_fam,  nu_chev_fam,  color="black", lw=1.4, ls="--",
            label=r"Median-swap family $\{C_\delta\}$")
    ax.plot(tau_corn_fam,  nu_corn_fam,  color="black", lw=1.4, ls="--",
            label=r"End-swap family $\{C_d\}$")
    ax.plot(-tau_chev_fam, -nu_chev_fam, color="black", lw=1.0, ls="--")
    ax.plot(-tau_corn_fam, -nu_corn_fam, color="black", lw=1.0, ls="--")

    mark_key_copulas(ax, Pi_xytext=(0, 18), Pi_va="bottom")

    scatter_annotate(ax, tau_star, nu_star,
                     r"$\tau_\ast\!\approx\!{:.4f}$".format(tau_star),
                     xytext=(10, -12), ha="left", va="top", fontsize=12,
                     color=RED, scatter_size=55)

    apply_standard_axes(ax, r"Kendall's $\tau$", r"Blest's $\nu$")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    save_region_plot(fig, "images/blest-vs-kendall-boundary-arcs.png")


if __name__ == "__main__":
    main()
