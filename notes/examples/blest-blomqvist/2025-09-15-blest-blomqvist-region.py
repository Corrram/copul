#!/usr/bin/env python3
"""
Plot Blest’s ν versus Blomqvist’s β
with the attainable region correctly shaded
and key copulas (M, Π, W) and a sample C_δ marked.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------------------------------------------------------------
#  Closed-form boundaries and family (from the paper)
# ----------------------------------------------------------------------
def nu_max(beta: float) -> float:
    """Upper boundary ν_max(β) on [-1,1]."""
    d = 1.0 - beta
    return 1.0 - (3.0 / 8.0) * d * d - (1.0 / 32.0) * d**4


def nu_min(beta: float) -> float:
    """Lower boundary ν_min(β) = -ν_max(-β)."""
    return -nu_max(-beta)


def beta_from_delta(delta: float) -> float:
    """β(C_δ) = 1 - 4δ, δ∈[0,1/2]."""
    return 1.0 - 4.0 * delta


def nu_from_delta(delta: float) -> float:
    """ν(C_δ) = 1 - 6δ² - 8δ⁴, δ∈[0,1/2]."""
    return 1.0 - 6.0 * delta * delta - 8.0 * delta**4


nu_max_vec = np.vectorize(nu_max)
nu_min_vec = np.vectorize(nu_min)
beta_from_delta_vec = np.vectorize(beta_from_delta)
nu_from_delta_vec = np.vectorize(nu_from_delta)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # β grid for the envelope
    beta = np.linspace(-1.0, 1.0, 2001)
    nu_upper = nu_max_vec(beta)
    nu_lower = nu_min_vec(beta)

    # Median-swap family curve (upper boundary, traced by δ)
    delta = np.linspace(0.0, 0.5, 400)
    beta_family = beta_from_delta_vec(delta)
    nu_family = nu_from_delta_vec(delta)

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Envelope curves
    ax.plot(beta, nu_upper, color=BLUE, lw=2.5, label=r"$\nu_{\max}(\beta)$")
    ax.plot(beta, nu_lower, color=BLUE, lw=2.5, label=r"$\nu_{\min}(\beta)$")

    # Shade attainable region
    ax.fill_between(
        beta,
        nu_lower,
        nu_upper,
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # Overlay the median-swap family (coincides with upper boundary)
    ax.plot(
        beta_family,
        nu_family,
        color="black",
        lw=1.2,
        ls="--",
        label=r"$\{C_\delta\}_{\delta\in[0,1/2]}$",
    )

    # ---------------------- Highlight key points ----------------------
    # Independence Π
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

    # Comonotonicity M (β=1, ν=1)
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

    # Countermonotonicity W (β=-1, ν=-1)
    ax.scatter([-1], [-1], s=60, color="black", zorder=5)
    ax.annotate(
        r"$W$",
        (-1, -1),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="bottom",
    )

    # Mark an interior extremal on the upper boundary: e.g. β=0 ↔ δ=1/4
    delta0 = 0.25
    beta0 = beta_from_delta(delta0)
    nu0 = nu_from_delta(delta0)  # = 19/32
    ax.scatter([beta0], [nu0], s=60, color="black", zorder=5)
    ax.annotate(
        r"$C_{\delta=1/4}$",
        (beta0, nu0),
        xytext=(8, -5),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="top",
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Blomqvist's $\beta$", fontsize=16)
    ax.set_ylabel(r"Blest's $\nu$", fontsize=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    ax.legend(loc="lower right", fontsize=12, frameon=True)
    fig.tight_layout()
    pathlib.Path("images").mkdir(exist_ok=True)
    plt.savefig("images/blest-blomqvist-region.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
