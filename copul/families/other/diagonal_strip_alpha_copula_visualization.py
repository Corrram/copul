import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Corrected import statement for modern SciPy versions
from scipy.integrate import cumulative_trapezoid, trapezoid


def get_psi(u, r):
    """Calculates the psi function from the paper."""
    return np.minimum(np.maximum(u - r / 2, 0), 1 - r)


def get_fV(v, r):
    """Calculates the density f_V(v) used for the coordinate transform."""
    if r == 0:
        return np.ones_like(v)
    fV = np.zeros_like(v)

    # Region 1: v in [0, r]
    mask1 = (v >= 0) & (v < r)
    fV[mask1] = (1 - v[mask1] - r / 2) / (1 - r)

    # Region 2: v in [r, 1-r]
    mask2 = (v >= r) & (v <= 1 - r)
    fV[mask2] = 1.0

    # Region 3: v in [1-r, 1]
    mask3 = (v > 1 - r) & (v <= 1)
    fV[mask3] = (v[mask3] - r / 2) / (1 - r)

    return fV


def get_v_from_w(w, r):
    """Calculates v = F_V^{-1}(w) by inverting the CDF of f_V."""
    if r == 0:
        return w

    v = np.zeros_like(w)

    # Invert in Region 1: w in [0, r]
    mask1 = (w >= 0) & (w < r)
    if np.any(mask1):
        w_masked = w[mask1]
        discriminant = (2 - r) ** 2 - 8 * w_masked * (1 - r)
        v[mask1] = (2 - r - np.sqrt(discriminant)) / 2

    # Invert in Region 2: w in [r, 1-r]
    mask2 = (w >= r) & (w <= 1 - r)
    v[mask2] = w[mask2]

    # Invert in Region 3: w in [1-r, 1]
    mask3 = (w > 1 - r) & (w <= 1)
    if np.any(mask3):
        w_masked = w[mask3]
        discriminant = r**2 - 4 * (1 - r) * (1 - 2 * w_masked)
        v[mask3] = (r + np.sqrt(discriminant)) / 2

    return v


def diagonal_hole_copula_density(u_grid, w_grid, alpha):
    """Calculates the density c(u,w) of the Diagonal Hole Copula."""
    if alpha == 0:
        return np.ones_like(u_grid)
    if alpha >= 0.5:
        return np.zeros_like(u_grid)

    r = np.sqrt(alpha)
    v_grid = get_v_from_w(w_grid, r)
    psi_u = get_psi(u_grid, r)
    is_in_hole = (v_grid >= psi_u) & (v_grid <= psi_u + r)
    fV_vals = get_fV(v_grid, r)

    # Avoid division by zero, though it shouldn't happen in valid regions
    fV_vals[fV_vals == 0] = 1e-9

    density = 1 / ((1 - r) * fV_vals)
    density[is_in_hole] = 0
    return density


def calculate_and_plot(alpha_val, n_points=200):
    """
    Calculates xi and psi, and generates a plot for a given alpha.
    """
    print(f"--- Processing for α = {alpha_val:.2f} ---")
    u = np.linspace(0, 1, n_points)
    w = np.linspace(0, 1, n_points)
    U, W = np.meshgrid(u, w)

    # 1. Calculate the copula density matrix Z = c(u,w)
    Z = diagonal_hole_copula_density(U, W, alpha_val)

    # 2. Calculate Chatterjee's xi
    # First, get h(u,w) = dC/du = integral_0^w c(u,y) dy
    # This is the cumulative integral down each column of the density matrix.
    H = cumulative_trapezoid(Z, w, initial=0, axis=0)

    # The xi integrand is h(u,w)^2
    xi_integrand = H**2

    # Integrate over the unit square
    integral_xi = trapezoid(trapezoid(xi_integrand, w, axis=0), u, axis=0)
    chatterjee_xi = 6 * integral_xi - 2
    print(f"Chatterjee's xi (ξ): {chatterjee_xi:.4f}")

    # 3. Calculate Spearman's footrule (psi)
    # First, get C(u,w) = integral_0^u h(x,w) dx
    # This is the cumulative integral across each row of the H matrix.
    C_matrix = cumulative_trapezoid(H, u, initial=0, axis=1)

    # We need the diagonal C(u,u)
    C_diagonal = np.diag(C_matrix)

    # Integrate C(u,u) from 0 to 1
    integral_psi = trapezoid(C_diagonal, u)
    spearman_psi = 6 * integral_psi - 2
    print(f"Spearman's footrule (ψ): {spearman_psi:.4f}\n")

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 6))
    if np.all(Z == 0):
        # Handle case where density is all zero
        ax.pcolormesh(U, W, Z, cmap="viridis", vmin=0, vmax=1)
    else:
        # Use logarithmic color scale for better visualization
        pcm = ax.pcolormesh(
            U,
            W,
            Z,
            norm=colors.LogNorm(vmin=Z[Z > 0].min(), vmax=Z.max()),
            cmap="viridis",
        )
        fig.colorbar(pcm, ax=ax, extend="max", label="Density (log scale)")

    title = (
        f"Diagonal Hole Copula Density (α = {alpha_val:.2f})\n"
        f"ξ ≈ {chatterjee_xi:.3f} | ψ ≈ {spearman_psi:.3f}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("u", fontsize=12)
    ax.set_ylabel("w", fontsize=12)
    ax.set_aspect("equal", "box")
    plt.savefig(f"images/diagonal_hole_copula_density_alpha_{alpha_val:.2f}.png")
    plt.show()


# --- Main ---
if __name__ == "__main__":
    alpha_values = [0, 0.01, 0.05, 0.10, 0.15, 0.20, 1 - np.sqrt(0.5), 0.25]
    for alpha in alpha_values:
        calculate_and_plot(alpha_val=alpha)
