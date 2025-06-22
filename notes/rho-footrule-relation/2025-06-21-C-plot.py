import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
#   PARAMETER TO TUNE
# =============================================================================
# The global parameter D determines the copula's structure.
# D is the limiting rescaled Lagrange multiplier, D = 3*zeta.
# For all three geometric regimes to be visible, D must be in (0, 1).
# - D -> 0 gives the comonotonicity copula C(u,v) = min(u,v).
# - D -> 1 gives the countermonotonicity copula C(u,v) = max(0, u+v-1).
# A value of D=0.4 shows the three regimes clearly.
D = 0.4
# =============================================================================


def get_t_minus(v, D):
    """
    Calculates the lower breakpoint function t_-(v) from the corrected
    derivation (H_0). This function defines the upper bound of the
    first support interval [0, t_-(v)].
    This function is vectorized to work with numpy arrays.
    """
    # Define the breakpoints between regimes
    v_minus = D / 2.0
    v_plus = 1.0 - D / 2.0

    # Initialize the result array
    t_m_values = np.zeros_like(v)

    # Define conditions for the three regimes
    cond_low_v = v <= v_minus
    cond_mid_v = (v > v_minus) & (v < v_plus)
    cond_high_v = v >= v_plus

    # Apply the piecewise formula
    t_m_values[cond_low_v] = 0.0
    t_m_values[cond_mid_v] = v[cond_mid_v] - D / 2.0
    t_m_values[cond_high_v] = 2.0 * v[cond_high_v] - 1.0
    
    return t_m_values


def get_t_plus(v, D):
    """
    Calculates the upper breakpoint function t_+(v) from the corrected
    derivation (H_0). This function defines the upper bound of the
    second support interval (v, t_+(v)].
    This function is vectorized to work with numpy arrays.
    """
    # Define the breakpoints between regimes
    v_minus = D / 2.0
    v_plus = 1.0 - D / 2.0

    # Initialize the result array
    t_p_values = np.zeros_like(v)

    # Define conditions for the three regimes
    cond_low_v = v <= v_minus
    cond_mid_v = (v > v_minus) & (v < v_plus)
    cond_high_v = v >= v_plus

    # Apply the piecewise formula
    t_p_values[cond_low_v] = 2.0 * v[cond_low_v]
    t_p_values[cond_mid_v] = v[cond_mid_v] + D / 2.0
    t_p_values[cond_high_v] = 1.0
    
    return t_p_values


def get_copula_C(U, V, D):
    """
    Calculates the value of the copula C(u,v) for a given D, based on the
    corrected formula (C_0).

    Args:
        U (np.ndarray): 2D meshgrid of u-coordinates.
        V (np.ndarray): 2D meshgrid of v-coordinates.
        D (float): The global parameter of the solution, must be in [0, 1].

    Returns:
        np.ndarray: The corresponding copula values C(u,v).
    """
    if not (0 <= D <= 1):
        raise ValueError("Parameter D must be between 0 and 1.")
        
    # First, calculate the breakpoint functions for all v-coordinates
    t_minus_vals = get_t_minus(V, D)
    t_plus_vals = get_t_plus(V, D)

    # Initialize the result array
    C = np.zeros_like(U)

    # Define the masks for the four piecewise regions of the copula
    cond1 = U <= t_minus_vals
    cond2 = (U > t_minus_vals) & (U <= V)
    cond3 = (U > V) & (U <= t_plus_vals)
    cond4 = U > t_plus_vals
    
    # Apply the piecewise formula for C(u,v)
    C[cond1] = U[cond1]
    C[cond2] = t_minus_vals[cond2]
    C[cond3] = t_minus_vals[cond3] + U[cond3] - V[cond3]
    C[cond4] = V[cond4] # This is because t_-(v) + t_+(v) - v = v

    return C

# --- Main script for plotting ---
if __name__ == '__main__':
    # 1. Generate data
    resolution = 100
    u_coords = np.linspace(0, 1, resolution)
    v_coords = np.linspace(0, 1, resolution)
    U, V = np.meshgrid(u_coords, v_coords)
    
    # Calculate the copula surface
    C_vals = get_copula_C(U, V, D)

    # 2. Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(U, V, C_vals, cmap='viridis', rstride=1, cstride=1, edgecolor='none', alpha=0.9)

    # 3. Customize the plot
    ax.set_title(f'Corrected Copula Surface $C_0(u,v)$ for $D={D}$', fontsize=16, pad=20)
    ax.set_xlabel('$u$', fontsize=14, labelpad=10)
    ax.set_ylabel('$v$', fontsize=14, labelpad=10)
    ax.set_zlabel('$C_0(u,v)$', fontsize=14, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    
    # Set a good initial viewing angle
    ax.view_init(elev=30, azim=-120)

    # 4. Show the plot
    plt.tight_layout()
    plt.show()