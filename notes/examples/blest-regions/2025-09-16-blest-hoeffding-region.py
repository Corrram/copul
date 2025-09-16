#!/usr/bin/env python3
"""
Blest’s ν vs Hoeffding’s D: corrected attainable region.

Upper boundary has two regimes:
  (A) Mixture regime, 0 ≤ D ≤ D★ = 1/120:
      C_α = (1-α) Π + α C_{μ=1}.
      ν(C_α) = α * (1/4),
      D(C_α) = α^2 * [ (1-α) * B_1 + α * D★ ],
      where B_1 = ∫(C_{μ=1}-uv)^2 du dv (BKR integral of the cusp copula).

  (B) Chevron regime, D★ ≤ D ≤ 13/120:
      Pure chevrons C_μ with μ∈[0,1] (closed forms below).

Lower boundary: vertical reflection ν_min(D) = -ν_max(D).

This script computes B_1 numerically (deterministic quadrature), constructs both
branches, and plots the union as the upper boundary, then mirrors it.
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# --------------------------- Chevron family C_μ ---------------------------
def t_star(mu: float) -> float:
    return 1.0 - 0.5 * mu


def T_mu_of_u(mu: float, u: np.ndarray) -> np.ndarray:
    """Tent map T_μ(u) = max(0, 1 - 2|u - t*|)."""
    t = t_star(mu)
    return np.maximum(0.0, 1.0 - 2.0 * np.abs(u - t))


def a_s_from_v(mu: float, v: np.ndarray):
    """For given v, return a(v), s(v) defining the central gap."""
    t = t_star(mu)
    g = 0.5 * (1.0 - v)  # half-width of the central gap
    a = np.maximum(0.0, t - g)  # left edge
    s = np.minimum(1.0, t + g)  # right edge
    return a, s


def C_mu_on_grid(mu: float, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    C_μ(u,v) = ∫_0^u 1{|t - t*| ≥ (1-v)/2} dt
             = min(u, a(v)) + max(0, u - s(v)).
    Vectorized on a meshgrid (U,V) with indexing='ij'.
    """
    a, s = a_s_from_v(mu, V[0, :])  # shape (m,)
    A = np.minimum(U, a)  # broadcast to (n,m)
    B = np.maximum(0.0, U - s)  # broadcast to (n,m)
    return A + B


def C_mu_line_on_graph(mu: float, u: np.ndarray) -> np.ndarray:
    """
    C_μ(u, T_μ(u)) along the singular support (V=T_μ(U)).
    """
    v = T_mu_of_u(mu, u)
    a, s = a_s_from_v(mu, v)
    return np.minimum(u, a) + np.maximum(0.0, u - s)


def nu_from_mu(mu: float) -> float:
    """Blest's ν for C_μ (piecewise closed form)."""
    if mu <= 1.0:
        return 1.0 - 0.75 * mu**4
    else:
        return -0.75 * mu**4 + 4.0 * mu**3 - 6.0 * mu**2 + 3.0


def D_from_mu(mu: float) -> float:
    """Hoeffding's D for C_μ (piecewise closed form)."""
    if mu <= 1.0:
        return (
            (mu**5) / 240.0
            - (mu**3) / 24.0
            + (mu**2) / 6.0
            - (11.0 / 48.0) * mu
            + 13.0 / 120.0
        )
    else:
        return (
            -(mu**5) / 240.0
            + (5.0 / 48.0) * mu**4
            - 0.5 * mu**3
            + mu**2
            - (43.0 / 48.0) * mu
            + 73.0 / 240.0
        )


# ------------------------- Numerics: B_1 (BKR of cusp) -------------------------
def compute_BKR_for_mu(mu: float, n: int = 600) -> float:
    """
    BKR(C_μ) = ∫_0^1 ∫_0^1 (C_μ(u,v) - u v)^2 du dv.
    Deterministic composite trapezoid on an n×n grid.
    """
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    C = C_mu_on_grid(mu, U, V)
    F = (C - U * V) ** 2
    # integrate over v then u
    I_v = np.trapz(F, v, axis=1)
    I = np.trapz(I_v, u)
    return float(I)


# ----------------------------- Upper boundary -----------------------------
def build_upper_boundary(num_alpha: int = 601, n_grid: int = 600):
    """
    Construct the upper boundary as the union of:
      - Mixture branch α ∈ [0,1] with Π and cusp C_{μ=1}
      - Chevron branch μ ∈ [0,1]
    Return D_upper (increasing) and ν_upper with the vertical reflection.
    """
    # --- Constants and grids
    mu_cusp = 1.0
    D_star = D_from_mu(mu_cusp)  # 1/120
    nu_star = nu_from_mu(mu_cusp)  # 1/4

    # Numerically compute B_1 = ∫(C_cusp - uv)^2 du dv
    B1 = compute_BKR_for_mu(mu_cusp, n=n_grid)

    # (A) Mixture branch with Π: C_α = (1-α)Π + α C_cusp
    #    ν(α) = α * ν_star
    #    D(α) = α^2 * [ (1-α)*B1 + α*D_star ]  (exact)
    alpha = np.linspace(0.0, 1.0, num_alpha)
    D_mix = (alpha**2) * ((1.0 - alpha) * B1 + alpha * D_star)
    nu_mix = alpha * nu_star

    # (B) Chevron branch μ ∈ [0,1]
    mu = np.linspace(0.0, 1.0, 1201)
    D_chev = np.array([D_from_mu(x) for x in mu])
    nu_chev = np.array([nu_from_mu(x) for x in mu])

    # Combine branches into a single increasing D curve (upper envelope)
    D_all = np.concatenate([D_mix, D_chev])
    nu_all = np.concatenate([nu_mix, nu_chev])

    # Sort by D and take the pointwise maximum ν to enforce the envelope
    order = np.argsort(D_all)
    D_sorted = D_all[order]
    nu_sorted = nu_all[order]

    # Collapse to a monotone envelope on a regular D grid
    D_min, D_max = 0.0, D_from_mu(0.0)  # 0 to 13/120
    D_grid = np.linspace(D_min, D_max, 1600)
    nu_upper = np.full_like(D_grid, -np.inf)

    # For speed, use a sliding window max via binning
    bins = np.searchsorted(D_sorted, D_grid)
    left = np.maximum(0, bins - 2)  # small neighborhood
    right = np.minimum(len(D_sorted), bins + 2)
    for i in range(len(D_grid)):
        nu_upper[i] = (
            np.max(nu_sorted[left[i] : right[i]]) if right[i] > left[i] else -np.inf
        )

    # Guarantee boundary includes endpoints explicitly
    nu_upper[0] = 0.0  # Π at (0,0)
    nu_upper[-1] = nu_from_mu(0.0)  # M at (13/120, 1)

    return D_grid, nu_upper


# --------------------------------- Plot ---------------------------------
def main() -> None:
    D_grid, nu_upper = build_upper_boundary()
    nu_lower = -nu_upper  # vertical reflection

    # Key points
    D_M, nu_M = D_from_mu(0.0), nu_from_mu(0.0)  # (13/120, 1)
    D_cusp, nu_cusp = D_from_mu(1.0), nu_from_mu(1.0)  # (1/120, 1/4)
    D_W, nu_W = D_from_mu(2.0), nu_from_mu(2.0)  # (11/240, -1)

    BLUE, FILL = "#00529B", "#D6EAF8"
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        D_grid, nu_upper, color=BLUE, lw=2.5, label=r"Upper boundary $\nu_{\max}(D)$"
    )
    ax.plot(D_grid, nu_lower, color=BLUE, lw=2.0)

    ax.fill_between(
        D_grid,
        nu_lower,
        nu_upper,
        color=FILL,
        alpha=0.9,
        zorder=0,
        label="Attainable region",
    )

    # Mark Π, cusp, M, W
    ax.scatter(
        [0.0, D_cusp, D_M, D_W],
        [0.0, nu_cusp, nu_M, nu_W],
        s=60,
        color="black",
        zorder=5,
    )
    ax.annotate(
        r"$\Pi$",
        (0.0, 0.0),
        xytext=(0, 18),
        textcoords="offset points",
        fontsize=18,
        ha="center",
        va="bottom",
    )
    ax.annotate(
        r"$\mu=1$",
        (D_cusp, nu_cusp),
        xytext=(8, -2),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="center",
    )
    ax.annotate(
        r"$M$",
        (D_M, nu_M),
        xytext=(-10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )
    ax.annotate(
        r"$W$",
        (D_W, nu_W),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="bottom",
    )

    ax.set_xlabel(r"Hoeffding's $D$", fontsize=16)
    ax.set_ylabel(r"Blest's $\nu$", fontsize=16)
    ax.set_xlim(-0.001, max(0.115, float(D_M)))
    ax.set_ylim(-1.05, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    fig.tight_layout()

    pathlib.Path("images").mkdir(exist_ok=True)
    plt.savefig("images/nu-hoeffding-region_corrected.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
