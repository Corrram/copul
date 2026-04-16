"""Lower-boundary density visualizer for the (ξ, ψ) region.

For each μ, solves  minimize  μ·ψ(H) + ξ(H)  (same problem as approx-lower_boundary)
then displays the resulting copula *density* c(v,t) = ∂H/∂v  rather than H itself.
Uses the midpoint-corrected marginal (column sums = 0.5, 1.5, …).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cvxpy as cp
from boundary_solver import solve_h, M_lower, xi_expr, psi_expr


def get_boundary_point(mu, n=32, verbose=False):
    """Return H_opt (or None) for the lower-boundary problem."""
    M = M_lower(n)
    return solve_h(
        lambda H: cp.Minimize(mu * psi_expr(H, n, M) + xi_expr(H, n)),
        n, marginal_offset=0.5, verbose=verbose,
    )


if __name__ == "__main__":
    mu_for_files = [0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0, 5.0, 8.0, 10.0]
    n_vis = 50
    Path("images").mkdir(exist_ok=True)

    for mu_val in mu_for_files:
        H_map = get_boundary_point(mu=mu_val, n=n_vis, verbose=True)
        if H_map is None:
            print(f"Solver failed for μ={mu_val}.")
            continue

        # Copula density: c(v,t) = ∂h/∂v  (finite difference across v-columns)
        h_padded = np.hstack([np.zeros((n_vis, 1)), H_map])
        C_map    = n_vis * np.diff(h_padded, axis=1)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(C_map.T, origin="lower", extent=[0, 1, 0, 1],
                       cmap="inferno", aspect="auto")
        ax.set_title(f"Copula density c(v,t) for $\\mu = {mu_val:.2f}$")
        ax.set_xlabel("t")
        ax.set_ylabel("v")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.02))
        ax.grid()
        plt.colorbar(im, ax=ax, orientation="vertical",
                     fraction=0.046, pad=0.04, label="Density c(v,t)")
        plt.savefig(f"images/xi_footrule_lower_density_mu_{mu_val:.2f}.png")
        plt.close()

    print("All density plots saved.")
