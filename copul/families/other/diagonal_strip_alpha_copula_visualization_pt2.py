import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import cumulative_trapezoid, trapezoid


def get_hole_width(v, a):
    """
    Calculates L(v), the width of the white "hole" region, based on the
    geometry. The parameter 'a' is the width of the white rectangles.
    """
    v = np.asarray(v)
    L_v = np.zeros_like(v, dtype=float)
    mask1 = (v >= 0) & (v < 0.5)
    L_v[mask1] = 2 * (1 - 2 * a) * v[mask1] + a
    mask2 = (v >= 0.5) & (v <= 1)
    L_v[mask2] = 1 - (2 * (1 - 2 * a) * (v[mask2] - 0.5) + a)
    return np.clip(L_v, 0, 1)


def construct_valid_geometric_copula(v_grid, u_grid, a):
    """
    Constructs a valid copula density for the specified geometry using
    the coordinate transformation method.
    """
    if a < 0 or a > 0.5:
        raise ValueError("'a' must be between 0 and 0.5")

    v_base_pts = v_grid[:, 0]
    L_v = get_hole_width(v_base_pts, a)
    # FIX 1: Replaced deprecated np.trapz with trapezoid
    hole_area = trapezoid(L_v, v_base_pts)
    if hole_area >= 1.0:
        return np.zeros_like(v_grid)

    f_V = (1 - L_v) / (1 - hole_area)
    F_V = cumulative_trapezoid(f_V, v_base_pts, initial=0)
    F_V = np.maximum.accumulate(F_V)

    v_base_mapped = np.interp(v_grid, F_V, v_base_pts)
    hole_width_mapped = get_hole_width(v_base_mapped, a)

    is_in_hole = np.zeros_like(v_grid, dtype=bool)
    mask_lower = v_base_mapped < 0.5
    is_in_hole[mask_lower] = u_grid[mask_lower] < hole_width_mapped[mask_lower]
    mask_upper = v_base_mapped >= 0.5
    is_in_hole[mask_upper] = u_grid[mask_upper] > (1 - hole_width_mapped[mask_upper])

    f_V_mapped = np.interp(v_base_mapped, v_base_pts, f_V)
    density = (1 / (1 - hole_area)) * (1 / (f_V_mapped + 1e-9))
    density[is_in_hole] = 0
    return density


def calculate_and_plot(a_val, n_points=500):
    """
    Calculates xi and psi, and generates a plot for a given 'a'.
    """
    print(f"--- Processing for a = {a_val:.2f} ---")
    u = np.linspace(0, 1, n_points)
    v = np.linspace(0, 1, n_points)
    # Note: V corresponds to axis 0, U to axis 1
    V, U = np.meshgrid(v, u, indexing="ij")

    # 1. Calculate the copula density matrix Z = c(v,u)
    Z = construct_valid_geometric_copula(V, U, a_val)

    # 2. Calculate Spearman's footrule (psi)
    # h(u,v) = dC/du = integral_0^v c(u,y) dy. Integrate along v-axis (axis 0)
    H = cumulative_trapezoid(Z, v, initial=0, axis=0)
    # C(u,v) = integral_0^u h(x,v) dx. Integrate along u-axis (axis 1)
    C_matrix = cumulative_trapezoid(H, u, initial=0, axis=1)
    # Integrate diagonal C(u,u)
    C_diagonal = np.diag(C_matrix)
    integral_psi = trapezoid(C_diagonal, u)
    spearman_psi = 6 * integral_psi - 2
    print(f"Spearman's footrule (ψ): {spearman_psi:.4f}")

    # 3. Calculate Chatterjee's xi
    # The xi integrand is h(u,v)^2
    xi_integrand = H**2
    # Integrate over the unit square
    # FIX 2: Removed axis=1 from the outer trapezoid call, as its input is 1D
    integral_xi = trapezoid(trapezoid(xi_integrand, v, axis=0), u)
    chatterjee_xi = 6 * integral_xi - 2
    print(f"Chatterjee's xi (ξ): {chatterjee_xi:.4f}\n")

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.percentile(Z[Z > 0], 99.5) if np.any(Z > 0) else 1.0
    pcm = ax.pcolormesh(U, V, Z, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label="Density c(u,v)")

    title = (
        f"Valid Geometric Copula Density (a = {a_val:.2f})\n"
        f"ξ ≈ {chatterjee_xi:.3f} | ψ ≈ {spearman_psi:.3f}"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("u", fontsize=12)
    ax.set_ylabel("v", fontsize=12)
    # Set MAJOR ticks to appear every 0.2 units for the labels
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))

    # Set MINOR ticks to appear every 0.1 units (without labels)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))

    ax.set_aspect("equal", "box")
    plt.grid()
    plt.savefig(
        f"images/copula_a_{a_val:.3f}_measures.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# --- Main ---
if __name__ == "__main__":
    # a=0.5 corresponds to the off-diagonal checkerboard copula
    # a=0 corresponds to a diagonal hole
    a_values = [
        0.5,
        0.45,
        0.4,
        0.35,
        0.3,
        0.29,
        1 - np.sqrt(0.5),
        0.28,
        0.25,
        0.225,
        0.2,
        0.1,
        0.0,
    ]
    for a in a_values:
        calculate_and_plot(a_val=a)
