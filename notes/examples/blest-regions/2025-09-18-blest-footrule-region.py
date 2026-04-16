#!/usr/bin/env python3
"""
Plot Spearman's footrule F (y-axis) versus Blest's ν (x-axis),
full [-1,1]² canvas shown.

Normalisation: F(C) = 6·∫₀¹ C(t,t) dt − 2,  so F ∈ [−1/2, 1].

Boundary formulas:
    ν_max(F) = 1 − (√6/3)·(1−F)^(3/2)
    ν_min(F) = (2/(3√3))·(1+2F)^(3/2) − 1
Upper boundary is traced by the median-swap family C_δ (δ ∈ [0, 1/2]):
    F(δ) = 1 − 6δ²,   ν(δ) = 1 − 12δ³.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import BLUE, FILL, apply_standard_axes, save_region_plot

SQRT3 = np.sqrt(3.0)


# ── Boundary and family formulas ─────────────────────────────────────────────

def nu_max(F: float) -> float:
    return 1.0 - (np.sqrt(6.0) / 3.0) * (1.0 - F)**1.5

def nu_min(F: float) -> float:
    return (2.0 / (3.0 * SQRT3)) * (1.0 + 2.0 * F)**1.5 - 1.0

def F_from_delta(delta: float) -> float:
    return 1.0 - 6.0 * delta**2

def nu_from_delta(delta: float) -> float:
    return 1.0 - 12.0 * delta**3

nu_max_v = np.vectorize(nu_max)
nu_min_v = np.vectorize(nu_min)
F_dv     = np.vectorize(F_from_delta)
nu_dv    = np.vectorize(nu_from_delta)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    F        = np.linspace(-0.5, 1.0, 2001)
    nu_upper = nu_max_v(F)
    nu_lower = nu_min_v(F)

    delta  = np.linspace(0.0, 0.5, 400)
    F_fam  = F_dv(delta)
    nu_fam = nu_dv(delta)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.fill_betweenx(F, nu_lower, nu_upper,
                     color=FILL, alpha=0.7, zorder=0, label="Attainable region")
    ax.plot(nu_upper, F, color=BLUE, lw=2.5, zorder=5, label=r"$\nu_{\max}(F)$")
    ax.plot(nu_lower, F, color=BLUE, lw=2.5, zorder=5, label=r"$\nu_{\min}(F)$")
    ax.plot([-1.0, -0.5], [-0.5, -0.5], color=BLUE, lw=2.5, zorder=6)
    ax.plot(nu_fam, F_fam, color="black", lw=1.2, ls="--",
            label=r"$\{C_\delta\}_{\delta\in[0,1/2]}$ (upper boundary)")

    apply_standard_axes(ax, r"Blest's $\nu$", r"Spearman's footrule",
                        ylim=(-1.05, 1.05), fontsize=15)
    save_region_plot(fig, "images/nu-psi-region.png")


if __name__ == "__main__":
    main()
