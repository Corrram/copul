#!/usr/bin/env python3
"""
Plot Chatterjee's ξ (x-axis) versus Blest's ν (y-axis).

Upper boundary ν_max(ξ) is traced by the clamped-parabola family C_μ (μ > 0).
Lower boundary by survival symmetry: ν_min(ξ) = −ν_max(ξ).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from plot_utils import BLUE, FILL, apply_standard_axes, mark_key_copulas, save_region_plot, scatter_annotate


# ── Clamped-parabola family C_μ ───────────────────────────────────────────────

def xi_mu(mu: float) -> float:
    if mu <= 0: return 1.0
    if mu >= 1:
        return (8.0 * (7.0*mu - 3.0)) / (105.0 * mu**3)
    s, t = np.sqrt(mu), np.sqrt(1.0 - mu)
    A    = np.arcsinh(t / s)
    num  = (-105*s**8*A + 183*s**6*t - 38*s**4*t - 88*s**2*t
            + 112*s**2 + 48*t - 48)
    return num / (210.0 * s**6)

def nu_mu(mu: float) -> float:
    if mu <= 0: return 1.0
    if mu >= 1:
        return (4.0 * (28.0*mu - 9.0)) / (105.0 * mu**2)
    s, t = np.sqrt(mu), np.sqrt(1.0 - mu)
    A    = np.arcsinh(t / s)
    num  = (-105*s**8*A + 87*s**6*t + 250*s**4*t - 376*s**2*t
            + 448*s**2 + 144*t - 144)
    return num / (420.0 * s**4)

xi_vec = np.vectorize(xi_mu)
nu_vec = np.vectorize(nu_mu)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    mu = np.concatenate([
        np.geomspace(1e-6, 1.0, 1200, endpoint=False),
        np.geomspace(1.0, 1e3, 1000),
    ])
    sort_idx  = np.argsort(xi_vec(mu))
    xi_sorted = np.clip(xi_vec(mu)[sort_idx], 0.0, 1.0)
    nu_sorted = np.clip(nu_vec(mu)[sort_idx], -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(xi_sorted,  nu_sorted, color=BLUE, lw=2.5, label=r"$\nu_{\max}(\xi)$")
    ax.plot(xi_sorted, -nu_sorted, color=BLUE, lw=2.5, label=r"$\nu_{\min}(\xi)$")
    ax.fill_between(xi_sorted, -nu_sorted, nu_sorted,
                    color=FILL, alpha=0.7, zorder=0, label="Attainable region")

    mark_key_copulas(ax, W=(1.0, -1.0), W_xytext=(-10, 0), W_ha="right", W_va="bottom",
                     Pi_xytext=(0, 20), Pi_va="top")

    scatter_annotate(ax, xi_mu(1.0), nu_mu(1.0), r"$C_{\mu=1}$",
                     xytext=(8, -5), ha="left", va="top", fontsize=16)

    apply_standard_axes(ax, r"Chatterjee's $\xi$", r"Blest's $\nu$",
                        xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    ax.legend(loc="lower right", fontsize=12, frameon=True)
    save_region_plot(fig, "images/xi-blest-region_axes-swapped.png")


if __name__ == "__main__":
    main()
