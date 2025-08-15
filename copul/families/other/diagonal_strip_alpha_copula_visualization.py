import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d


def create_and_plot_corrected_shape(alpha, ax):
    """
    Calculates and plots the corrected non-linear corridor shape for a given alpha.
    """
    if alpha == 0: return
    r = np.sqrt(alpha)

    # 1. Density component c(y) from the original (invalid) model
    # (ensures V-marginal would be uniform if U-marginal were fixed)
    def c_y(y):
        if y < r:
            return 1 / (y + r / 2)
        elif y > 1 - r:
            return 1 / (1 - y + r / 2)
        else:
            return 1 / r

    # 2. Integral of c(y) over a vertical slice of the strip
    def G(z):
        return quad(c_y, z, z + r)[0]

    # 3. Old linear shape psi(u) and resulting non-uniform marginal f(u)
    def psi_old(u):
        return np.minimum(np.maximum(u - r / 2, 0), 1 - r)

    def f_U_old(u):
        # Vectorize the calculation for plotting
        return np.array([G(psi_old(val)) for val in u])

    # 4. Create the inverse CDF F^-1 to use for reparametrization
    u_grid = np.linspace(0, 1, 201)
    f_u_grid = f_U_old(u_grid)

    # Numerically integrate to get the cumulative function F(u)
    F_u_grid = np.cumsum(f_u_grid) * (u_grid[1] - u_grid[0])
    total_area = F_u_grid[-1]
    F_u_normalized_grid = F_u_grid / total_area

    # Create an interpolator to act as the inverse function F^-1(t)
    F_inv = interp1d(F_u_normalized_grid, u_grid, kind='cubic',
                     bounds_error=False, fill_value=(0, 1))

    # 5. Define the new, valid shape function phi(t)
    t_vals = np.linspace(0, 1, 201)
    phi_new = psi_old(F_inv(t_vals))

    # Plotting
    ax.plot(u_grid, psi_old(u_grid), 'r--', label='Old Linear Shape $\\psi(u)$')
    ax.plot(t_vals, phi_new, 'g-', label='New Corrected Shape $\\phi(u)$', linewidth=2)
    ax.set_title(f'Corrected Corridor Shape for $\\alpha = {alpha}$')
    ax.set_xlabel('$u$ (or $t$ for reparametrized axis)')
    ax.set_ylabel('Lower Bound of Strip')
    ax.legend()
    ax.grid(True)


# Create plots for different alpha values
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
create_and_plot_corrected_shape(0.1, axes[0])
create_and_plot_corrected_shape(0.4, axes[1])
fig.tight_layout()
plt.savefig("corrected_corridor_plot.png")