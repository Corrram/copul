import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import patches
from scipy.integrate import cumulative_trapezoid, trapezoid
import os


def psi(s, alpha, beta):
    """
    Calculates the lower boundary of the diagonal hole.
    This is the Python implementation of the piecewise function ψ(s).
    """
    # Ensure s is a numpy array for vectorized operations
    s = np.asarray(s)
    # Initialize the output array
    ps_val = np.zeros_like(s, dtype=float)

    # Define the masks for the three pieces of the function
    mask_middle = (s > alpha) & (s < 1 - alpha)
    mask_upper = s >= (1 - alpha)

    # Calculate the value for the sloped middle section
    # This check prevents division by zero if alpha is exactly 0.5
    if alpha < 0.5:
        slope = (1 - beta) / (1 - 2 * alpha)
        ps_val[mask_middle] = slope * (s[mask_middle] - alpha)

    # Set the value for the flat upper section
    ps_val[mask_upper] = 1 - beta

    return ps_val


def get_L_t(t_grid, alpha, beta, n_points=1000):
    """
    Numerically calculates L(t), the horizontal width of the hole at height t.
    It integrates an indicator function over a high-resolution grid.
    """
    s_fine = np.linspace(0, 1, n_points)
    psi_vals = psi(s_fine, alpha, beta)  # Shape: (n_points,)

    # Create a 2D boolean matrix where each row corresponds to a t value
    # is_in_hole[i, j] is True if s_fine[j] is in the hole at height t_grid[i]
    is_in_hole = (psi_vals[np.newaxis, :] <= t_grid[:, np.newaxis]) & (
        psi_vals[np.newaxis, :] + beta >= t_grid[:, np.newaxis]
    )

    # Integrate along the s-axis (axis=1) to get the width L(t) for each t
    L_t_values = trapezoid(is_in_hole.astype(float), s_fine, axis=1)
    return L_t_values


def construct_diagonal_copula(u_grid, v_grid, alpha, beta):
    """
    Constructs the valid copula density using the coordinate transformation method.
    This function is analogous to the original 'construct_valid_geometric_copula'.
    """
    if alpha >= 0.5 or beta >= 1:
        raise ValueError("Parameters must satisfy alpha < 0.5 and beta < 1")

    # The transformation is only on the v-axis (t in the LaTeX)
    t_base_pts = v_grid[:, 0]

    # 1. Calculate the geometry and pre-marginal density f_T(t)
    L_t = get_L_t(t_base_pts, alpha, beta)
    hole_area = beta  # Analytically known from the definition

    if hole_area >= 1.0:
        return np.zeros_like(v_grid)

    f_T = (1 - L_t) / (1 - hole_area)

    # 2. Compute the CDF F_T(t) and perform the inverse transform
    F_T = cumulative_trapezoid(f_T, t_base_pts, initial=0)
    # Ensure monotonicity, which can be affected by numerical precision
    F_T = np.maximum.accumulate(F_T)

    # Map the grid from the final 'w' space to the intermediate 't' space
    t_mapped = np.interp(v_grid, F_T, t_base_pts)

    # 3. Define the hole and density in the intermediate (s, t) space
    # The u-axis is not transformed, so s_grid is just u_grid
    s_grid = u_grid

    psi_vals_mapped = psi(s_grid, alpha, beta)
    is_in_hole = (t_mapped >= psi_vals_mapped) & (t_mapped <= psi_vals_mapped + beta)

    # 4. Calculate the final density using the change of variables formula
    f_T_mapped = np.interp(t_mapped, t_base_pts, f_T)
    density = (1 / (1 - hole_area)) * (
        1 / (f_T_mapped + 1e-9)
    )  # Add epsilon for stability
    density[is_in_hole] = 0
    return density


def calculate_and_plot(alpha_val, beta_val, n_points=1000):
    """
    Calculates measures and generates a plot for given alpha and beta.
    """
    print(f"--- Processing for α = {alpha_val:.2f}, β = {beta_val:.2f} ---")
    u = np.linspace(0, 1, n_points)
    v = np.linspace(0, 1, n_points)
    # V corresponds to axis 0 (rows), U to axis 1 (columns)
    V, U = np.meshgrid(v, u, indexing="ij")

    # 1. Calculate the copula density matrix Z = c(u,v)
    Z = construct_diagonal_copula(U, V, alpha_val, beta_val)

    # 2. Calculate Spearman's footrule (psi)
    H = cumulative_trapezoid(Z, v, initial=0, axis=0)
    C_matrix = cumulative_trapezoid(H, u, initial=0, axis=1)
    C_diagonal = np.diag(C_matrix)
    integral_psi = trapezoid(C_diagonal, u)
    spearman_psi = 6 * integral_psi - 2
    print(f"Spearman's footrule (ψ): {spearman_psi:.4f}")

    # 3. Calculate Chatterjee's xi
    xi_integrand = H**2
    integral_xi = trapezoid(trapezoid(xi_integrand, v, axis=0), u)
    chatterjee_xi = 6 * integral_xi - 2
    print(f"Chatterjee's xi (ξ): {chatterjee_xi:.4f}\n")

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 6))
    # Clip max value for better color contrast, ignoring extreme peaks
    vmax = np.percentile(Z[Z > 0], 99.5) if np.any(Z > 0) else 1.0
    pcm = ax.pcolormesh(U, V, Z, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="Density c(u,v)")

    # Rectangles for alpha × beta corners
    zero_density_color = "white"
    rect1 = patches.Rectangle(
        (0, 0),
        alpha_val,
        beta_val,
        linewidth=2,
        linestyle="dashed",
        edgecolor=zero_density_color,
        facecolor="none",
        alpha=1,
    )
    rect2 = patches.Rectangle(
        (1 - alpha_val, 1 - beta_val),
        alpha_val,
        beta_val,
        linewidth=2,
        linestyle="dashed",
        edgecolor=zero_density_color,
        facecolor="none",
        alpha=1,
    )
    ax.add_patch(rect1)
    ax.add_patch(rect2)

    # Labels inside rectangles
    ax.text(
        alpha_val / 2,
        beta_val / 2,
        r"$\alpha\times\beta$",
        color=zero_density_color,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        1 - alpha_val / 2,
        1 - beta_val / 2,
        r"$\alpha\times\beta$",
        color=zero_density_color,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.5,
        r"$H_{\alpha,\beta}$",
        color=zero_density_color,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Overlay the "hole" band outline
    s = np.linspace(0, 1, 400)
    psi_vals = psi(s, alpha_val, beta_val)
    ax.plot(s, psi_vals, color=zero_density_color, lw=1.5, label=r"$\psi(s)$")
    ax.plot(
        s,
        psi_vals + beta_val,
        color=zero_density_color,
        lw=2,
        # linestyle="--",
        label=r"$\psi(s)+\beta$",
    )
    ax.fill_between(s, psi_vals, psi_vals + beta_val, color=zero_density_color, alpha=0)

    title = (
        f"Diagonal Hole Copula (α = {alpha_val:.2f}, β = {beta_val:.2f})\n"
        f"ξ ≈ {chatterjee_xi:.3f} | ψ ≈ {spearman_psi:.3f}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("u", fontsize=12)
    ax.set_ylabel("v", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    ax.set_aspect("equal", "box")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Ensure the 'images' directory exists
    os.makedirs("images", exist_ok=True)
    plt.savefig(
        f"images/two_param_a{alpha_val:.2f}_b{beta_val:.2f}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# --- Main ---
if __name__ == "__main__":
    # List of (alpha, beta) pairs to visualize
    parameter_pairs = [
        # (0.35, 0.8),
        # (0.05, 0.08),  # Small corners, thin band
        # (0.10, 0.16),  # Small corners, very wide band
        # (0.12, 0.2),  # Medium corners, medium band
        # (0.15, 0.25),  # Medium corners, wider band
        (0.20, 0.3),  # Wide corners, thin band
        # (0.22, 0.35),  # Wide corners, medium band
        # (0.25, 0.4),  # Medium corners, wider band
        # (0.28, 0.5),  # Wide corners, very thin band
        (0.30, 0.5),
        # (0.35, 0.5),
        # (0.40, 0.5),  # Wide corners, very wide band
        # (0.45, 0.5),  # Very wide corners, very wide band
        # (0.48, 0.5),  # Almost full width
        # (0.5, 0.5),  # Almost full width
    ]

    for alpha, beta in parameter_pairs:
        calculate_and_plot(alpha_val=alpha, beta_val=beta)
