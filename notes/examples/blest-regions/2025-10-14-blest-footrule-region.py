#!/usr/bin/env python3
"""
Plot Spearman’s footrule F versus Blest’s ν with axes swapped
(x-axis: ν, y-axis: F) and the full [-1,1]^2 canvas shown.

Normalisation:
    F(C) = 6 ∫_0^1 C(t,t) dt - 2

Boundary formulas:
  ν_max(F) = 1 - (√6 / 3) * (1 - F)^(3/2)
  ν_min(F) = (2 / (3√3)) * (1 + 2F)^(3/2) - 1

Upper boundary is traced by the median-swap family C_δ:
  F(C_δ) = 1 - 6 δ^2,  ν(C_δ) = 1 - 12 δ^3,  δ ∈ [0, 1/2]
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------
#  Closed-form boundaries and median-swap family
# ----------------------------------------------------------------------
SQRT3 = np.sqrt(3.0)


def nu_max(F: float) -> float:
    """Upper boundary ν_max(F) for F ∈ [-1/2, 1]."""
    d = 1.0 - F
    return 1.0 - (np.sqrt(6.0) / 3.0) * (d**1.5)


def nu_min(F: float) -> float:
    """Lower boundary ν_min(F) for F ∈ [-1/2, 1]."""
    return (2.0 / (3.0 * SQRT3)) * ((1.0 + 2.0 * F) ** 1.5) - 1.0


def F_from_delta(delta: float) -> float:
    """F(C_δ) = 1 - 6 δ^2,  δ∈[0,1/2]."""
    return 1.0 - 6.0 * (delta**2)


def nu_from_delta(delta: float) -> float:
    """ν(C_δ) = 1 - 12 δ^3,  δ∈[0,1/2]."""
    return 1.0 - 12.0 * (delta**3)


nu_max_vec = np.vectorize(nu_max)
nu_min_vec = np.vectorize(nu_min)
F_from_delta_vec = np.vectorize(F_from_delta)
nu_from_delta_vec = np.vectorize(nu_from_delta)


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------
def main() -> None:
    # F grid for the envelope (true domain [-1/2, 1])
    F = np.linspace(-0.5, 1.0, 2001)
    nu_upper = nu_max_vec(F)  # x-values (upper boundary as ν(F))
    nu_lower = nu_min_vec(F)  # x-values (lower boundary as ν(F))

    # Median-swap family curve (upper boundary, traced by δ)
    delta = np.linspace(0.0, 0.5, 400)
    F_family = F_from_delta_vec(delta)  # y-values
    nu_family = nu_from_delta_vec(delta)  # x-values

    # -------------------------- Plotting ------------------------------
    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 8))

    # After the fill_betweenx
    ax.fill_betweenx(
        F,
        nu_lower,
        nu_upper,
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # Draw both boundary curves clearly on top of the shading
    ax.plot(nu_upper, F, color=BLUE, lw=2.5, zorder=5, label=r"$\nu_{\max}(F)$")
    ax.plot(nu_lower, F, color=BLUE, lw=2.5, zorder=5, label=r"$\nu_{\min}(F)$")
    ax.plot([-1.0, -0.5], [-0.5, -0.5], color=BLUE, lw=2.5, zorder=6)
    # Shade attainable region between lower/upper in x for each y=F
    ax.fill_betweenx(
        F,
        nu_lower,  # x1
        nu_upper,  # x2
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # Overlay the median-swap family (coincides with upper boundary)
    ax.plot(
        nu_family,
        F_family,
        color="black",
        lw=1.2,
        ls="--",
        label=r"$\{C_\delta\}_{\delta\in[0,1/2]}$ (upper boundary)",
    )

    # ---------------------- Highlight key points ----------------------
    # Π: (F,ν)=(0,0) -> (ν,F)=(0,0)
    # ax.scatter([0], [0], s=60, color="black", zorder=5)
    # ax.annotate(
    #     r"$\Pi$",
    #     (0, 0),
    #     xytext=(0, 20),
    #     textcoords="offset points",
    #     fontsize=18,
    #     ha="center",
    #     va="top",
    # )
    #
    # # M: (F,ν)=(1,1) -> (ν,F)=(1,1)
    # ax.scatter([1], [1], s=60, color="black", zorder=5)
    # ax.annotate(
    #     r"$M$",
    #     (1, 1),
    #     xytext=(-10, 0),
    #     textcoords="offset points",
    #     fontsize=18,
    #     ha="right",
    #     va="top",
    # )
    #
    # # W: (F,ν)=(-1/2,-1) -> (ν,F)=(-1,-1/2)
    # ax.scatter([-1.0], [-0.5], s=60, color="black", zorder=5)
    # ax.annotate(
    #     r"$W$",
    #     (-1.0, -0.5),
    #     xytext=(10, 0),
    #     textcoords="offset points",
    #     fontsize=18,
    #     ha="left",
    #     va="bottom",
    # )

    # Mark an interior point on the upper boundary: δ=1/4 ⇒ F=0.625, ν=0.8125
    # delta0 = 0.25
    # F0 = F_from_delta(delta0)
    # nu0 = nu_from_delta(delta0)
    # ax.scatter([nu0], [F0], s=60, color="black", zorder=5)
    # ax.annotate(
    #     r"$C_{\delta=1/4}$",
    #     (nu0, F0),
    #     xytext=(8, -5),
    #     textcoords="offset points",
    #     fontsize=16,
    #     ha="left",
    #     va="top",
    # )

    # -------------------- Axes, grid, legend --------------------------
    ax.set_xlabel(r"Blest's $\nu$", fontsize=16)
    ax.set_ylabel(r"Spearman's footrule", fontsize=15)

    # Full square [-1,1]^2
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)

    # ax.legend(loc="lower right", fontsize=12, frameon=True)
    fig.tight_layout()
    Path("images").mkdir(exist_ok=True)
    plt.savefig("images/nu-psi-region.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
