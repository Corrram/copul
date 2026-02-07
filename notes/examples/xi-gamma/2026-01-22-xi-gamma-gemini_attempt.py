import numpy as np
import matplotlib.pyplot as plt


def h_analytical_vectorized(u_grid, v, lam):
    """
    Computes the vector of h_v(u) values for a fixed fiber v and array of u,
    using the explicit analytical solution derived from the KKT conditions.
    """
    h = np.zeros_like(u_grid)
    delta = lam / 2.0  # The flat spot has width lambda, so half-width is lambda/2

    # Logic for fibers below the midline
    if v < 0.5:
        # Structure:
        # 1. Linear ascent (shifted by lam)
        # 2. Upward jump at v (mass on diagonal) -> shift becomes 2*lam
        # 3. Flat spot around 1-v (ironing out the drop)

        flat_start = (1 - v) - delta
        flat_end = (1 - v) + delta

        # Segment 1: u < v
        mask1 = u_grid < v
        h[mask1] = u_grid[mask1] + lam

        # Segment 2: v <= u < flat_start
        mask2 = (u_grid >= v) & (u_grid < flat_start)
        h[mask2] = u_grid[mask2] + 2 * lam

        # Segment 3: Flat spot (Constant value to maintain continuity)
        mask3 = (u_grid >= flat_start) & (u_grid <= flat_end)
        val_flat = flat_start + 2 * lam
        h[mask3] = val_flat

        # Segment 4: u > flat_end
        mask4 = u_grid > flat_end
        h[mask4] = u_grid[mask4] + lam

    # Logic for fibers above the midline (Symmetric)
    else:
        flat_start = (1 - v) - delta
        flat_end = (1 - v) + delta

        # Segment 1: u < flat_start
        mask1 = u_grid < flat_start
        h[mask1] = u_grid[mask1] + lam

        # Segment 2: Flat spot
        mask2 = (u_grid >= flat_start) & (u_grid <= flat_end)
        val_flat = flat_start + lam
        h[mask2] = val_flat

        # Segment 3: flat_end < u < v (Drop region, back to slope 1, no shift)
        mask3 = (u_grid > flat_end) & (u_grid < v)
        h[mask3] = u_grid[mask3]

        # Segment 4: u >= v (Upward jump at diagonal)
        mask4 = u_grid >= v
        h[mask4] = u_grid[mask4] + lam

    # Enforce global bounds for a valid CDF
    return np.clip(h, 0, 1)


# --- Plotting ---
lambdas = [0.05, 0.15, 0.8]
res = 400
u_vals = np.linspace(0, 1, res)
v_vals = np.linspace(0, 1, res)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for k, lam in enumerate(lambdas):
    # Construct the matrix H where H[row, col] corresponds to h_v(u)
    # Row index -> v (y-axis), Col index -> u (x-axis)
    H_matrix = np.zeros((res, res))

    for j, v in enumerate(v_vals):
        H_matrix[j, :] = h_analytical_vectorized(u_vals, v, lam)

    ax = axes[k]
    # Use origin='lower' to place (0,0) at bottom-left
    im = ax.imshow(
        H_matrix, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=0, vmax=1
    )

    ax.set_title(f"$\lambda = {lam}$")
    ax.set_xlabel("u")
    if k == 0:
        ax.set_ylabel("v")

    # Guidelines
    ax.plot([0, 1], [0, 1], "w--", linewidth=0.8, alpha=0.5)
    ax.plot([0, 1], [1, 0], "w:", linewidth=0.8, alpha=0.5)

# Colorbar
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Conditional CDF $h_v(u)$")

plt.show()
