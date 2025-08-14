import numpy as np
import matplotlib.pyplot as plt


# ------------------------- Helper functions ------------------------- #
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
    # Clip the argument of arccos to the valid range [-1, 1] to avoid numerical errors
    theta = (1 / 3) * np.arccos(np.clip(1 - (108 / 25) * x_val, -1.0, 1.0))
    return np.clip((5 / 6) + (5 / 3) * np.cos(theta - 2 * np.pi / 3), 0.0, 1.0)


def M_x_upper_bound_corrected(x_val: float) -> float:
    """Corrected upper bound M_x."""
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


def main():
    # ----------------------------- Data --------------------------------- #
    eps = 1e-9
    xi_points = np.concatenate(
        [
            np.linspace(0.0, 3 / 10 - eps, 150),
            np.linspace(3 / 10 - eps, 3 / 10 + eps, 50),
            np.linspace(3 / 10 + eps, 1.0, 150),
        ]
    )
    xi_points = np.unique(np.clip(xi_points, 0.0, 1.0))

    rho_up = np.array([M_x_upper_bound_corrected(x) for x in xi_points])
    valid = ~np.isnan(rho_up)
    xi_v, rho_up_v = xi_points[valid], rho_up[valid]

    # ----------------------------- Plotting Colors --------------------------------- #
    BLUE = "#00529B"
    FILL = "#D6EAF8"

    fig, ax = plt.subplots(figsize=(8, 6))

    # main envelope
    ax.plot(rho_up_v, xi_v, color=BLUE, lw=2.5)
    ax.plot(-rho_up_v, xi_v, color=BLUE, lw=2.5)
    ax.plot(rho_up_v, xi_v, color=BLUE, lw=2.5, label=r"$\pm M_\xi$")
    ax.plot(-rho_up_v, xi_v, color=BLUE, lw=2.5)

    # fill central region
    ax.fill_betweenx(
        xi_v,
        -rho_up_v,
        rho_up_v,
        color=FILL,
        alpha=0.7,
        zorder=0,
        label="Attainable region",
    )

    # hatched |ρ| > ξ regions
    mask = rho_up_v > xi_v
    ax.fill_betweenx(
        xi_v,
        xi_v,
        rho_up_v,
        where=mask,
        facecolor="none",
        hatch="..",
        edgecolor=BLUE,
        linewidth=0,
    )
    ax.fill_betweenx(
        xi_v,
        -rho_up_v,
        -xi_v,
        where=mask,
        facecolor="none",
        hatch="..",
        edgecolor=BLUE,
        linewidth=0,
    )

    # key points
    key_rho = [0, 1, -1, 0.7, -0.7]
    key_xi = [0, 1, 1, 0.3, 0.3]
    ax.scatter(key_rho, key_xi, s=60, color="black", zorder=5)

    # nicely positioned labels
    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(0, 20),
        textcoords="offset points",
        fontsize=18,
        ha="center",
        va="top",
    )
    ax.annotate(
        r"$M$",
        (1, 1),
        xytext=(-10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )
    ax.annotate(
        r"$W$",
        (-1, 1),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="top",
    )
    ax.annotate(
        r"$C_1$",
        (0.7, 0.3),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="top",
    )
    ax.annotate(
        r"$C_{-1}$",
        (-0.7, 0.3),
        xytext=(0, 0),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="top",
    )

    # axes, grid, legend
    ax.set_xlabel(r"Spearman's rho", fontsize=16)
    ax.set_ylabel(r"Chatterjee's xi", fontsize=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", lw=0.8)

    ax.legend(loc="center", fontsize=12, frameon=True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
