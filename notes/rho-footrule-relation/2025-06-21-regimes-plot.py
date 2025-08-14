import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  PARAMETER TO TUNE
# =============================================================================
# The global parameter D determines the solution's structure.
# D is related to the limiting Lagrange multiplier (D = 3*zeta_0/2).
# For all three regimes to be visible, D must be in (0, 0.5).
# - If D <= 0 or D >= 0.5, the intermediate regime vanishes.
# =============================================================================


def get_t_in(v, D):
    """
    Calculates the inner switching-time function t_in(v).

    Args:
        v (np.ndarray): Array of v values in [0, 1].
        D (float): The global parameter of the solution.

    Returns:
        np.ndarray: The corresponding t_in values.
    """
    # Create an empty array to store the results
    t_in_values = np.zeros_like(v)

    # Define the conditions for the three regimes
    cond_low_v = (v >= 0) & (v <= D)
    cond_mid_v = (v > D) & (v < 1 - D)
    cond_high_v = (v >= 1 - D) & (v <= 1)

    # Apply the corresponding formula for each regime
    t_in_values[cond_low_v] = 2 * v[cond_low_v] - 2 * D
    t_in_values[cond_mid_v] = v[cond_mid_v] - D
    t_in_values[cond_high_v] = 2 * v[cond_high_v] - 1

    return t_in_values


def get_t_out(v, D):
    """
    Calculates the outer switching-time function t_out(v).

    Args:
        v (np.ndarray): Array of v values in [0, 1].
        D (float): The global parameter of the solution.

    Returns:
        np.ndarray: The corresponding t_out values.
    """
    # Create an empty array to store the results
    t_out_values = np.zeros_like(v)

    # Define the conditions for the three regimes
    cond_low_v = (v >= 0) & (v <= D)
    cond_mid_v = (v > D) & (v < 1 - D)
    cond_high_v = (v >= 1 - D) & (v <= 1)

    # Apply the corresponding formula for each regime
    t_out_values[cond_low_v] = 2 * v[cond_low_v]
    t_out_values[cond_mid_v] = v[cond_mid_v] + D
    t_out_values[cond_high_v] = 2 * v[cond_high_v] - 1 + 2 * D

    return t_out_values


# --- Main script for plotting ---
if __name__ == "__main__":
    D = 0.5
    # 1. Generate data
    v_coords = np.linspace(0, 1, 500)
    t_in_coords = get_t_in(v_coords, D)
    t_out_coords = get_t_out(v_coords, D)

    # 2. Create the plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the main functions
    ax.plot(v_coords, t_in_coords, lw=2.5, label="$t_{in}(v)$ (Inner ramp switch)")
    ax.plot(v_coords, t_out_coords, lw=2.5, label="$t_{out}(v)$ (Outer ramp switch)")

    # Plot reference lines for context
    ax.plot(v_coords, v_coords, "k--", lw=1, alpha=0.7, label="$y=v$")

    # Plot vertical lines for regime boundaries, only if the mid-regime exists
    if 0 < D < 0.5:
        ax.axvline(x=D, color="gray", linestyle="--", lw=1, label=f"$v=D={D}$")
        ax.axvline(
            x=1 - D, color="gray", linestyle="--", lw=1, label=f"$v=1-D={1 - D}$"
        )

    # 3. Customize the plot
    ax.set_title(
        f"Switching-Time Functions for the Limiting Optimizer ($D={D}$)", fontsize=16
    )
    ax.set_xlabel("$v$", fontsize=12)
    ax.set_ylabel("$t$", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(
        "equal", adjustable="box"
    )  # Make the plot square for better intuition

    # 4. Show the plot
    plt.show()
