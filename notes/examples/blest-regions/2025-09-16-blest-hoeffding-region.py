#!/usr/bin/env python3
"""
Plot Hoeffding's D (x-axis) versus Blest's ν (y-axis).

Upper boundary has two regimes:
  (A) Mixture regime, 0 ≤ D ≤ D★ = 1/120:
      C_α = (1−α)Π + α·C_{μ=1},   ν(α) = α/4,
      D(α) = α²·[(1−α)·B₁ + α·D★]  (B₁ computed numerically)
  (B) Chevron regime, D★ ≤ D ≤ 13/120:
      Pure chevrons C_μ with μ ∈ [0,1].
Lower boundary: vertical reflection ν_min(D) = −ν_max(D).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from plot_utils import BLUE, FILL, mark_key_copulas, save_region_plot, scatter_annotate


# ── Chevron family C_μ ────────────────────────────────────────────────────────

def t_star(mu: float) -> float:
    return 1.0 - 0.5 * mu

def C_mu_on_grid(mu: float, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    t = t_star(mu)
    g = 0.5 * (1.0 - V[0, :])
    a = np.maximum(0.0, t - g)
    s = np.minimum(1.0, t + g)
    return np.minimum(U, a) + np.maximum(0.0, U - s)

def nu_from_mu(mu: float) -> float:
    if mu <= 1.0:
        return 1.0 - 0.75 * mu**4
    return -0.75*mu**4 + 4.0*mu**3 - 6.0*mu**2 + 3.0

def D_from_mu(mu: float) -> float:
    if mu <= 1.0:
        return mu**5/240 - mu**3/24 + mu**2/6 - (11/48)*mu + 13/120
    return -mu**5/240 + (5/48)*mu**4 - 0.5*mu**3 + mu**2 - (43/48)*mu + 73/240

def compute_BKR_for_mu(mu: float, n: int = 600) -> float:
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    C    = C_mu_on_grid(mu, U, V)
    F    = (C - U * V) ** 2
    return float(np.trapz(np.trapz(F, v, axis=1), u))


# ── Upper boundary construction ───────────────────────────────────────────────

def build_upper_boundary(num_alpha: int = 601, n_grid: int = 600):
    mu_cusp = 1.0
    D_star  = D_from_mu(mu_cusp)
    nu_star = nu_from_mu(mu_cusp)
    B1      = compute_BKR_for_mu(mu_cusp, n=n_grid)

    alpha  = np.linspace(0.0, 1.0, num_alpha)
    D_mix  = alpha**2 * ((1.0 - alpha) * B1 + alpha * D_star)
    nu_mix = alpha * nu_star

    mu        = np.linspace(0.0, 1.0, 1201)
    D_chev    = np.array([D_from_mu(x) for x in mu])
    nu_chev   = np.array([nu_from_mu(x) for x in mu])

    D_all  = np.concatenate([D_mix,  D_chev])
    nu_all = np.concatenate([nu_mix, nu_chev])
    order  = np.argsort(D_all)
    D_sorted  = D_all[order]
    nu_sorted = nu_all[order]

    D_grid    = np.linspace(0.0, D_from_mu(0.0), 1600)
    nu_upper  = np.full_like(D_grid, -np.inf)
    bins  = np.searchsorted(D_sorted, D_grid)
    left  = np.maximum(0, bins - 2)
    right = np.minimum(len(D_sorted), bins + 2)
    for i in range(len(D_grid)):
        nu_upper[i] = np.max(nu_sorted[left[i]:right[i]]) if right[i] > left[i] else -np.inf
    nu_upper[0]  = 0.0
    nu_upper[-1] = nu_from_mu(0.0)
    return D_grid, nu_upper


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    D_grid, nu_upper = build_upper_boundary()
    nu_lower = -nu_upper

    D_M, nu_M   = D_from_mu(0.0), nu_from_mu(0.0)
    D_cusp, nu_cusp = D_from_mu(1.0), nu_from_mu(1.0)
    D_W, nu_W   = D_from_mu(2.0), nu_from_mu(2.0)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(D_grid, nu_upper, color=BLUE, lw=2.5, label=r"Upper boundary $\nu_{\max}(D)$")
    ax.plot(D_grid, nu_lower, color=BLUE, lw=2.0)
    ax.fill_between(D_grid, nu_lower, nu_upper,
                    color=FILL, alpha=0.9, zorder=0, label="Attainable region")

    scatter_annotate(ax, 0.0,    0.0,    r"$\Pi$",   xytext=(0, 18),   ha="center", va="bottom")
    scatter_annotate(ax, D_cusp, nu_cusp, r"$\mu=1$", xytext=(8, -2),   ha="left",  va="center", fontsize=16)
    scatter_annotate(ax, D_M,   nu_M,   r"$M$",    xytext=(-10, 0), ha="right",  va="top")
    scatter_annotate(ax, D_W,   nu_W,   r"$W$",    xytext=(8, 0),   ha="left",   va="bottom")

    ax.set_xlabel(r"Hoeffding's $D$", fontsize=16)
    ax.set_ylabel(r"Blest's $\nu$",   fontsize=16)
    ax.set_xlim(-0.001, max(0.115, float(D_M)))
    ax.set_ylim(-1.05, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    save_region_plot(fig, "images/nu-hoeffding-region_corrected.png")


if __name__ == "__main__":
    main()
