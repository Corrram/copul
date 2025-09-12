import numpy as np
import copul as cp

# =============================================================================
#   PARAMETERS TO TUNE FOR DEMONSTRATION
# =============================================================================
# =============================================================================


def get_t_minus(v, D):
    """
    Calculates the lower breakpoint function t_-(v). [cite]
    This function is vectorized to work with numpy arrays.
    """
    v_minus = D / 2.0
    v_plus = 1.0 - D / 2.0
    t_m_values = np.zeros_like(v)
    cond_low_v = v <= v_minus
    cond_mid_v = (v > v_minus) & (v < v_plus)
    cond_high_v = v >= v_plus
    t_m_values[cond_low_v] = 0.0
    t_m_values[cond_mid_v] = v[cond_mid_v] - D / 2.0
    t_m_values[cond_high_v] = 2.0 * v[cond_high_v] - 1.0
    return t_m_values


def get_t_plus(v, D):
    """
    Calculates the upper breakpoint function t_+(v). [cite]
    This function is vectorized to work with numpy arrays.
    """
    v_minus = D / 2.0
    v_plus = 1.0 - D / 2.0
    t_p_values = np.zeros_like(v)
    cond_low_v = v <= v_minus
    cond_mid_v = (v > v_minus) & (v < v_plus)
    cond_high_v = v >= v_plus
    t_p_values[cond_low_v] = 2.0 * v[cond_low_v]
    t_p_values[cond_mid_v] = v[cond_mid_v] + D / 2.0
    t_p_values[cond_high_v] = 1.0
    return t_p_values


def get_copula_C(U, V, D):
    """
    Calculates the value of the copula C(u,v) for a given D. [cite]
    """
    if not (0 <= D <= 1):
        raise ValueError("Parameter D must be between 0 and 1.")
    t_minus_vals = get_t_minus(V, D)
    t_plus_vals = get_t_plus(V, D)
    # Clamp t_minus to be non-negative for a valid copula calculation
    t_minus_vals = np.maximum(0, t_minus_vals)
    C = np.zeros_like(U)
    cond1 = U <= t_minus_vals
    cond2 = (U > t_minus_vals) & (U <= V)
    cond3 = (U > V) & (U <= t_plus_vals)
    cond4 = U > t_plus_vals
    C[cond1] = U[cond1]
    C[cond2] = t_minus_vals[cond2]
    C[cond3] = t_minus_vals[cond3] + U[cond3] - V[cond3]
    C[cond4] = V[cond4]
    return C


def calculate_mass_matrix(D, n):
    """
    Calculates the probability mass of the copula C_0 on an n x n grid.

    Args:
        D (float): The global parameter of the copula, must be in [0, 1].
        n (int): The number of subdivisions for the grid along each axis.

    Returns:
        np.ndarray: An n x n matrix where each element (i, j) contains the
                    probability mass on the square
                    [j/n, (j+1)/n] x [i/n, (i+1)/n].
    """
    # 1. Create a grid of corner coordinates for the n x n cells.
    # This requires n+1 points along each axis.
    coords = np.linspace(0, 1, n + 1)
    U_grid, V_grid = np.meshgrid(coords, coords)

    # 2. Calculate the copula value C(u,v) at every corner.
    C_at_corners = get_copula_C(U_grid, V_grid, D)

    # 3. Calculate the mass in each cell using the C-volume formula.
    # This can be vectorized by taking differences of the corner values.
    # Mass(i,j) = C(u_j+1, v_i+1) - C(u_j, v_i+1) - C(u_j+1, v_i) + C(u_j, v_i)
    mass_matrix = (
        C_at_corners[1:, 1:]
        - C_at_corners[:-1, 1:]
        - C_at_corners[1:, :-1]
        + C_at_corners[:-1, :-1]
    )

    return mass_matrix


# --- Main script for demonstration ---
if __name__ == "__main__":
    # 1. Calculate the mass matrix
    # The global parameter D for the copula family.
    D_param = 0.4
    # The number of subdivisions for the grid.
    n_grid = 5
    print(f"Calculating {n_grid}x{n_grid} mass matrix for D={D_param}...")
    mass_matrix = calculate_mass_matrix(D_param, n_grid)
    print(f"Calculation complete, min mass: {np.min(mass_matrix):.6f}")
    print(f"{mass_matrix}")
    # 2. Sanity check: the sum of all probabilities should be 1.0
    total_mass = np.sum(mass_matrix)
    ccop = cp.BivCheckPi(mass_matrix)
    rho = ccop.spearmans_rho()
    footrule = ccop.spearmans_footrule()
    print(f"Spearman's rho: {rho:.6f}")
    print(f"Spearman's Footrule: {footrule:.6f}")
    # print(f"Total probability mass in the matrix: {total_mass:.6f}")

    # # 3. Visualize the mass matrix as a heatmap
    # fig, ax = plt.subplots(figsize=(10, 8))
    # im = ax.imshow(mass_matrix, cmap='viridis', origin='lower',
    #                 extent=[0, 1, 0, 1])

    # # 4. Customize the plot
    # ax.set_title(f'Probability Mass of $C_0$ for $D={D_param}$ on a {n_grid}x{n_grid} Grid',
    #              fontsize=16, pad=15)
    # ax.set_xlabel('$u$', fontsize=14)
    # ax.set_ylabel('$v$', fontsize=14)

    # # Add a color bar to show the mapping of colors to probabilities
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label('Probability Mass', fontsize=12)

    # # Show the plot
    # plt.show()
