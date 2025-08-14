import numpy as np
import matplotlib.pyplot as plt


# ------------------------- Helper functions (Corrected) ------------------------- #


def b_from_xi_regime1(xi_val: float) -> float:
    """
    Finds b > 1 from a given xi_val in (3/10, 1].
    This function has been CORRECTED to use the proper formula for xi.
    """
    if not (3 / 10 <= xi_val <= 1.0):
        return np.nan
    if np.isclose(xi_val, 1.0):
        return np.inf

    # The correct formula is: xi = 1 - 1/b + 3/(10b^2)
    # This rearranges to the quadratic equation for b:
    # 10*(1-xi)*b^2 - 10*b + 3 = 0
    a = 10 * (1 - xi_val)
    b_coeff = -10
    c = 3

    discriminant = b_coeff**2 - 4 * a * c
    if discriminant < 0:
        return np.nan

    # We need the larger root for the case b > 1
    sol = (-b_coeff + np.sqrt(discriminant)) / (2 * a)
    return sol


def b_from_xi_regime2(xi_val: float) -> float:
    """Finds 0 < b <= 1 from a given xi_val in (0, 3/10]. (This was correct)"""
    if not (0 <= xi_val <= 3 / 10):
        return np.nan
    if np.isclose(xi_val, 0):
        return 0.0

    # The formula is xi = 1 - b + b^2/5
    # This rearranges to the quadratic equation for b:
    # (1/5)*b^2 - b + (1-xi) = 0
    a = 1 / 5
    b_coeff = -1
    c = 1 - xi_val

    discriminant = b_coeff**2 - 4 * a * c
    if discriminant < 0:
        return np.nan

    # We take the smaller root for b <= 1
    sol = (-b_coeff - np.sqrt(discriminant)) / (2 * a)
    return sol


# ------------------------- Boundary function for Tau (Unchanged) ------------------------- #


def tau_from_xi_upper(xi_val: float) -> float:
    """
    Calculates the upper bound of Kendall's tau for a given Chatterjee's xi.
    """
    if not (0 <= xi_val <= 1):
        return np.nan
    if np.isclose(xi_val, 0):
        return 0.0
    if np.isclose(xi_val, 1):
        return 1.0

    xi_thresh = 3 / 10
    if xi_val <= xi_thresh:
        b = b_from_xi_regime2(xi_val)
        # Formula for tau when 0 < b <= 1
        return b * (4 - b) / 6
    else:  # xi_val > xi_thresh
        b = b_from_xi_regime1(xi_val)
        if np.isinf(b):
            return 1.0
        # Formula for tau when b > 1
        return (6 * b**2 - 4 * b + 1) / (6 * b**2)


def main():
    # ----------------------------- Data Generation --------------------------------- #
    xi_points = np.linspace(0, 1, 400)

    # Calculate the corresponding upper bound for tau
    tau_up = np.array([tau_from_xi_upper(x) for x in xi_points])
    valid = ~np.isnan(tau_up)
    xi_v, tau_up_v = xi_points[valid], tau_up[valid]

    # ----------------------------- Plotting Setup --------------------------------- #
    BLUE = "#00529B"
    FILL = "#D6EAF8"
    RED = "#E41A1C"

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the main envelope boundary
    ax.plot(tau_up_v, xi_v, color=BLUE, lw=2.5, label=r"Boundary $T_\xi$")
    ax.plot(-tau_up_v, xi_v, color=BLUE, lw=2.5)

    # Fill the attainable region
    ax.fill_betweenx(
        xi_v,
        -tau_up_v,
        tau_up_v,
        color=FILL,
        alpha=1.0,
        zorder=0,
        label="Attainable region",
    )

    # Add the line τ = ξ for comparison
    ax.plot(
        [-1, 1], [-1, 1], color="gray", linestyle="--", lw=1.5, label=r"$\tau = \xi$"
    )

    # Hatched region where |τ| > ξ
    mask = tau_up_v > xi_v
    ax.fill_betweenx(
        xi_v,
        xi_v,
        tau_up_v,
        where=mask,
        facecolor="none",
        hatch="..",
        edgecolor=BLUE,
        linewidth=0.0,
    )
    ax.fill_betweenx(
        xi_v,
        -tau_up_v,
        -xi_v,
        where=mask,
        facecolor="none",
        hatch="..",
        edgecolor=BLUE,
        linewidth=0.0,
    )

    # ----------------------------- Key Points and Annotations --------------------------------- #
    b0 = (10 - np.sqrt(10)) / 9
    tau_b0 = b0 * (4 - b0) / 6
    xi_b0 = 1 - b0 + b0**2 / 5

    key_tau = [0, 1, -1, 0.5, -0.5, tau_b0]
    key_xi = [0, 1, 1, 0.3, 0.3, xi_b0]
    ax.scatter(key_tau, key_xi, s=60, color="black", zorder=5)
    ax.scatter([tau_b0], [xi_b0], s=100, color=RED, zorder=6, label=r"Max $\tau-\xi$")

    ax.annotate(
        r"$\Pi$",
        (0, 0),
        xytext=(-5, -5),
        textcoords="offset points",
        fontsize=18,
        ha="right",
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
        (0.5, 0.3),
        xytext=(8, -3),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="center",
    )
    ax.annotate(
        r"$C_{-1}$",
        (-0.5, 0.3),
        xytext=(-8, -3),
        textcoords="offset points",
        fontsize=18,
        ha="right",
        va="center",
    )
    ax.annotate(
        r"$C_{b_0}$",
        (tau_b0, xi_b0),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=18,
        ha="center",
        va="bottom",
        color=RED,
    )

    # ----------------------------- Axes, Grid, and Legend --------------------------------- #
    ax.set_xlabel(r"Kendall's tau ($\tau$)", fontsize=16)
    ax.set_ylabel(r"Chatterjee's xi ($\xi$)", fontsize=16)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        fontsize=12,
        frameon=True,
        ncol=2,
    )
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


if __name__ == "__main__":
    main()
