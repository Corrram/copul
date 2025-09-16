#!/usr/bin/env python3
"""
Plot Blomqvist’s β versus Blest’s ν (axes swapped vs. the original script).

We now place:
  x-axis: ν,  y-axis: β

Boundaries (same formulas), but plotted as x=ν(β), y=β.
The attainable region is shaded using fill_betweenx over β.
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
    nu_upper = nu_max_vec(beta)  # x-values for the upper boundary
    nu_lower = nu_min_vec(beta)  # x-values for the lower boundary

    # Median-swap family curve (coincides with upper boundary)
    delta = np.linspace(0.0, 0.5, 400)
    beta_family = beta_from_delta_vec(delta)
    nu_family = nu_from_delta_vec(delta)

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Envelope curves (now plotted as x=ν, y=β)
    ax.plot(nu_upper, beta, color=BLUE, lw=2.5, label=r"$\nu_{\max}(\beta)$")
    ax.plot(nu_lower, beta, color=BLUE, lw=2.5, label=r"$\nu_{\min}(\beta)$")

    # Shade attainable region between lower/upper in x for each y=β
    ax.fill_betweenx(
        beta,
        nu_lower,  # x1
        nu_upper,  # x2
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # Overlay the median-swap family (same arc, dashed) as (x=ν, y=β)
    ax.plot(
        nu_family,
        beta_family,
        color="black",
        lw=1.2,
        ls="--",
        label=r"$\{C_\delta\}_{\delta\in[0,1/2]}$",
    )

    # ---------------------- Highlight key points ----------------------
    # Π: (β,ν)=(0,0) -> (ν,β)=(0,0)
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

    # M: (β,ν)=(1,1) -> (ν,β)=(1,1)
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

    # W: (β,ν)=(-1,-1) -> (ν,β)=(-1,-1)
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
    ax.scatter([nu0], [beta0], s=60, color="black", zorder=5)
    ax.annotate(
        r"$C_{\delta=1/4}$",
        (nu0, beta0),
        xytext=(8, -5),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="top",
    )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Blest's $\nu$", fontsize=16)
    ax.set_ylabel(r"Blomqvist's $\beta$", fontsize=16)
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
    plt.savefig("images/blomqvist-vs-blest_axes-swapped.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
