#!/usr/bin/env python3
"""
Plot Blomqvist's β (y-axis) versus Blest's ν (x-axis)  [clean version].

Boundaries (closed-form):
    ν_max(β) = 1 − (3/8)·(1−β)² − (1/32)·(1−β)⁴
    ν_min(β) = −ν_max(−β)
Median-swap family C_δ (δ ∈ [0, 1/2]): β(δ) = 1−4δ, ν(δ) = 1−6δ²−8δ⁴.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import BLUE, FILL, apply_standard_axes, save_region_plot


# ── Boundary and family formulas ─────────────────────────────────────────────

def nu_max(beta: float) -> float:
    d = 1.0 - beta
    return 1.0 - (3.0/8.0)*d**2 - (1.0/32.0)*d**4

def nu_min(beta: float) -> float:
    return -nu_max(-beta)

def beta_from_delta(delta: float) -> float:
    return 1.0 - 4.0 * delta

def nu_from_delta(delta: float) -> float:
    d2 = delta**2
    return 1.0 - 6.0*d2 - 8.0*d2**2

nu_max_v = np.vectorize(nu_max)
nu_min_v = np.vectorize(nu_min)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    beta     = np.linspace(-1.0, 1.0, 2001)
    nu_upper = nu_max_v(beta)
    nu_lower = nu_min_v(beta)

    delta      = np.linspace(0.0, 0.5, 400)
    beta_fam   = np.vectorize(beta_from_delta)(delta)
    nu_fam     = np.vectorize(nu_from_delta)(delta)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(nu_upper, beta, color=BLUE, lw=2.5, label=r"$\nu_{\max}(\beta)$")
    ax.plot(nu_lower, beta, color=BLUE, lw=2.5, label=r"$\nu_{\min}(\beta)$")
    ax.fill_betweenx(beta, nu_lower, nu_upper,
                     color=FILL, alpha=0.7, zorder=0, label="Attainable region")
    ax.plot(nu_fam, beta_fam, color="black", lw=1.2, ls="--",
            label=r"$\{C_\delta\}_{\delta\in[0,1/2]}$")

    apply_standard_axes(ax, r"Blest's $\nu$", r"Blomqvist's $\beta$")
    save_region_plot(fig, "images/nu-beta-region.png")


if __name__ == "__main__":
    main()
