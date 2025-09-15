#!/usr/bin/env python3
"""
Plot Blest’s ν (x-axis) versus Chatterjee’s ξ (y-axis)
with the attainable region correctly shaded
and key copulas (M, Π, W) and a sample C_μ marked.

Upper boundary is traced by the optimiser family C_μ (μ>0).
Lower boundary follows by survival symmetry: (ξ, -ν).
"""

import numpy as np
import matplotlib.pyplot as plt
from _pytest import pathlib
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
#  Closed-form ξ(μ), ν(μ) from the clamped–parabola family
# ----------------------------------------------------------------------
def xi_mu(mu: float) -> float:
    if mu <= 0:
        return 1.0
    if mu >= 1:
        return (8.0 * (7.0 * mu - 3.0)) / (105.0 * mu**3)
    s = np.sqrt(mu)
    t = np.sqrt(1.0 - mu)
    A = np.arcsinh(t / s)
    num = (
        -105.0 * s**8 * A
        + 183.0 * s**6 * t
        - 38.0 * s**4 * t
        - 88.0 * s**2 * t
        + 112.0 * s**2
        + 48.0 * t
        - 48.0
    )
    den = 210.0 * s**6
    return num / den


def nu_mu(mu: float) -> float:
    if mu <= 0:
        return 1.0
    if mu >= 1:
        return (4.0 * (28.0 * mu - 9.0)) / (105.0 * mu**2)
    s = np.sqrt(mu)
    t = np.sqrt(1.0 - mu)
    A = np.arcsinh(t / s)
    num = (
        -105.0 * s**8 * A
        + 87.0 * s**6 * t
        + 250.0 * s**4 * t
        - 376.0 * s**2 * t
        + 448.0 * s**2
        + 144.0 * t
        - 144.0
    )
    den = 420.0 * s**4
    return num / den


xi_vec = np.vectorize(xi_mu)
nu_vec = np.vectorize(nu_mu)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # Parameter grid for μ>0 covering both branches densely near μ=1
    mu_left = np.geomspace(1e-6, 1.0, 1200, endpoint=False)  # (0,1)
    mu_right = np.geomspace(1.0, 1e3, 1000)  # [1,∞)
    mu = np.concatenate([mu_left, mu_right])

    xi_vals = xi_vec(mu)
    nu_vals = nu_vec(mu)

    # Sort by ξ (monotone in μ, but sort for safety)
    sort_idx = np.argsort(xi_vals)
    xi_sorted = np.clip(xi_vals[sort_idx], 0.0, 1.0)
    nu_sorted = np.clip(nu_vals[sort_idx], -1.0, 1.0)

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the envelopes as x(ν) vs y(ξ)
    ax.plot(nu_sorted, xi_sorted, color=BLUE, lw=2.5, label=r"$\nu_{\max}(\xi)$")
    ax.plot(-nu_sorted, xi_sorted, color=BLUE, lw=2.5, label=r"$\nu_{\min}(\xi)$")

    # Shade attainable region between ±ν_max(ξ) at each ξ
    ax.fill_betweenx(
        xi_sorted,
        -nu_sorted,
        nu_sorted,
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # ---------------------- Highlight key points ----------------------
    # Independence Π: (ν, ξ) = (0, 0)
    ax.scatter([0], [0], s=60, color="black", zorder=5)
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(0, 20),
        textcoords="offset points",
        fontsize=18,
        ha="center",
        va="top",
    )

    # Comonotonicity M: (ν, ξ) = (1, 1)
    ax.scatter([1], [1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(-10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )

    # Countermonotonicity W: (ν, ξ) = (-1, 1)
    ax.scatter([-1], [1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$W$",
        (-1, 1),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="top",
    )

    # Mark a representative interior optimiser, e.g. μ=1
    mu0 = 1.0
    xi0 = xi_mu(mu0)  # 32/105
    nu0 = nu_mu(mu0)  # 76/105
    ax.scatter([nu0], [xi0], s=60, color="black", zorder=6)
    ax.annotate(
        r"$C_{\mu=1}$",
        (nu0, xi0),
        xytext=(8, -5),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="top",
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Blest's $\nu$", fontsize=16)
    ax.set_ylabel(r"Chatterjee's $\xi$", fontsize=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    ax.legend(loc="lower right", fontsize=12, frameon=True)
    fig.tight_layout()
    pathlib.Path("images").mkdir(exist_ok=True)
    plt.savefig("images/xi-blest-region.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
