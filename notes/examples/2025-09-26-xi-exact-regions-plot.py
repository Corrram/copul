#!/usr/bin/env python3
"""
Make a single horizontal figure with 3 panels:
1) (xi, rho) attainable region
2) (xi, tau) attainable region
3) (xi, psi) attainable region (uses lower_boundary_final_smooth.csv)

Creates:
- images/xi_tau_rho_psi_panels.png
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

# ----------------------------- Style -------------------------------- #
BLUE = "#00529B"
FILL = "#D6EAF8"

# ========================= Panel 1: xi–rho =========================== #
def b_from_x_regime1(x_val: float) -> float:
    """b(x) for x in (3/10, 1]  with b > 1."""
    if np.isclose(x_val, 1.0):
        return np.inf
    if x_val <= 3 / 10:
        return 1.0 if np.isclose(x_val, 3 / 10) else np.nan
    numer = 5 + np.sqrt(5 * (6 * x_val - 1))
    denom = 10 * (1 - x_val)
    return np.inf if np.isclose(denom, 0) else numer / denom

def b_from_x_regime2(x_val: float) -> float:
    """b(x) for x in (0, 3/10]  with 0 < b ≤ 1."""
    if np.isclose(x_val, 0):
        return 0.0
    if x_val > 3 / 10:
        return 1.0 if np.isclose(x_val, 3 / 10) else np.nan
    theta = (1 / 3) * np.arccos(np.clip(1 - (108 / 25) * x_val, -1.0, 1.0))
    return np.clip((5 / 6) + (5 / 3) * np.cos(theta - 2 * np.pi / 3), 0.0, 1.0)

def M_x_upper_bound_corrected(x_val: float) -> float:
    """Corrected upper bound M_x for rho given xi."""
    if x_val < 0 or x_val > 1:
        return np.nan
    if np.isclose(x_val, 0):
        return 0.0
    if np.isclose(x_val, 1):
        return 1.0
    x_thresh = 3 / 10
    if x_val < x_thresh and not np.isclose(x_val, x_thresh):
        b = b_from_x_regime2(x_val)
        return b - (3 * b**2) / 10
    if x_val > x_thresh and not np.isclose(x_val, x_thresh):
        b = b_from_x_regime1(x_val)
        if np.isinf(b):
            return 1.0
        if np.isnan(b) or b == 0:
            return np.nan
        return 1 - 1 / (2 * b**2) + 1 / (5 * b**3)
    if np.isclose(x_val, x_thresh):  # x = 3/10
        b = 1.0
        return b - (3 * b**2) / 10
    return np.nan

def plot_xi_rho(ax: plt.Axes) -> None:
    eps = 1e-9
    xi_points = np.concatenate([
        np.linspace(0.0, 3/10 - eps, 150),
        np.linspace(3/10 - eps, 3/10 + eps, 50),
        np.linspace(3/10 + eps, 1.0, 150),
    ])
    xi_points = np.unique(np.clip(xi_points, 0.0, 1.0))
    rho_up = np.array([M_x_upper_bound_corrected(x) for x in xi_points])
    valid = ~np.isnan(rho_up)
    xi_v, rho_up_v = xi_points[valid], rho_up[valid]

    # Envelope ±M_x
    ax.plot(xi_v,  rho_up_v, color=BLUE, lw=2.2, label=r"$\pm M_\xi$")
    ax.plot(xi_v, -rho_up_v, color=BLUE, lw=2.2)

    # Fill attainable region
    ax.fill_between(xi_v, -rho_up_v, rho_up_v, color=FILL, alpha=0.7, zorder=0)

    # Axes
    ax.set_title(r"$(\xi,\rho)$", fontsize=13, pad=2)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\rho$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)

# ========================= Panel 2: xi–tau =========================== #
def xi_from_b(b: float) -> float:
    """Chatterjee’s ξ(C_b) – even in b."""
    ab = abs(b)
    return (b*b/10) * (5 - 2*ab) if ab <= 1 else 1 - 1/ab + 3/(10*ab*ab)

def tau_from_b(b: float) -> float:
    """Kendall’s τ(C_b) – odd in b."""
    if b >= 0:
        return (b*(4-b))/6 if b <= 1 else (6*b*b - 4*b + 1)/(6*b*b)
    return -tau_from_b(-b)

xi_vec, tau_vec = map(np.vectorize, (xi_from_b, tau_from_b))

def plot_xi_tau(ax: plt.Axes) -> None:
    b_pos = np.hstack([
        np.linspace(0.0, 1.0, 600, endpoint=False)[1:],  # 0<b≤1
        np.linspace(1.0, 20.0, 1400)                     # long tail
    ])
    xi_pos, tau_pos = xi_vec(b_pos), tau_vec(b_pos)
    idx = np.argsort(xi_pos)
    xi_sorted = xi_pos[idx]
    tau_sorted_pos = tau_pos[idx]

    xi_plot = np.concatenate([xi_sorted, [1.0]])
    tau_plot_pos = np.concatenate([tau_sorted_pos, [1.0]])

    ax.plot(xi_plot,  tau_plot_pos, color=BLUE, lw=2.2, label=r"$\pm\tau(\xi)$")
    ax.plot(xi_plot, -tau_plot_pos, color=BLUE, lw=2.2)

    ax.fill_between(xi_plot, -tau_plot_pos, tau_plot_pos, color=FILL, alpha=0.7, zorder=0)

    ax.set_title(r"$(\xi,\tau)$", fontsize=13, pad=2)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\tau$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)

# ========================= Panel 3: xi–psi =========================== #
def calculate_lower_boundary(mu_values: np.ndarray):
    """Lower boundary param in mu for (xi, psi). Returns (xi_vals, psi_vals)."""
    epsilon = 1e-12
    mu_values = np.maximum(mu_values, 0.5)
    v1 = 2.0 * mu_values / (2.0 * mu_values + 1.0)
    B = 1.0 / (2.0 * mu_values)
    safe_v1 = np.maximum(v1, epsilon)
    psi_vals = -2 * safe_v1**2 + 6 * safe_v1 - 5 + 1.0 / safe_v1
    B_sq = B**2
    poly_part = 4 * safe_v1**3 - 18 * safe_v1**2 + 36 * safe_v1 - 22
    log_part = -12 * np.log(safe_v1)
    b_term_poly = -4 * safe_v1**3 + 6 * safe_v1**2 - 1
    xi_vals = poly_part + log_part + B_sq * b_term_poly
    return xi_vals, psi_vals

def plot_xi_psi(ax: plt.Axes, csv_path: str = "lower_boundary_final_smooth.csv") -> None:
    xi_upper = np.linspace(0, 1, 500)
    psi_upper = np.sqrt(xi_upper)
    mu_vals = np.logspace(4, -4, 2000) + 0.5
    xi_lower, psi_lower = calculate_lower_boundary(mu_vals)
    xi_endpoint = 12 * np.log(2) - 8  # endpoint for the lower curve
    df = pd.read_csv(csv_path)

    fill_lower_x = np.concatenate([df["xi"].values, [1.0]])
    fill_lower_y = np.concatenate([df["psi"].values, [-0.5]])
    fill_poly_x = np.concatenate([xi_upper, fill_lower_x[::-1]])
    fill_poly_y = np.concatenate([psi_upper, fill_lower_y[::-1]])

    ax.plot(xi_upper, psi_upper, color=BLUE, lw=2.2, label="Boundary")
    ax.plot(xi_lower, psi_lower, color=BLUE, lw=2.2)
    ax.plot([xi_endpoint, 1.0], [-0.5, -0.5], color=BLUE, lw=2.2)
    ax.fill(fill_poly_x, fill_poly_y, color=FILL, alpha=0.7, zorder=0)
    ax.plot(df["xi"], df["psi"], ls="--", lw=2.2, zorder=4,
            label=r"$(\xi(C^*_{\mu}), \psi(C^*_{\mu}))$ for $\mu\ge 0$")

    ax.set_title(r"$(\xi,\psi)$", fontsize=13, pad=2)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\psi$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)

# ============================== Main ================================ #
def main():
    fig, axes = plt.subplots(
        1, 3,
        figsize=(10, 5),
        sharex=True,
        sharey=True,
        layout="constrained"
    )
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, wspace=0.01, hspace=0.01)

    plot_xi_rho(axes[0])
    plot_xi_tau(axes[1])
    plot_xi_psi(axes[2])

    # custom x-axis labels: ticks at 0.25 but only label 0,0.5,1
    # custom ticks: show grid every 0.25 but labels only at 0, 0.5, 1
    for ax in axes:
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
        ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["$-1$", "", "$-0.5$", "", "0", "", "0.5", "", "1"])

    # remove y tick labels on middle/right
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    # keep square look
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")

    outdir = pathlib.Path("images")
    outdir.mkdir(parents=False, exist_ok=True)
    outfile = outdir / "xi_tau_rho_psi_panels.png"
    plt.savefig(outfile, dpi=300)
    plt.show()
    print(f"Saved figure to: {outfile}")

if __name__ == "__main__":
    main()
