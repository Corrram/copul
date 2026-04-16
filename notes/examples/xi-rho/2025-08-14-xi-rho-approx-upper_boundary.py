"""Upper boundary of the (ξ, ρ) attainable region.

Solves  minimize  μ·ξ(H) − ρ(H)  for a sweep of μ values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from boundary_solver import (
    solve_h, M_lower,
    xi_expr, rho_expr, xi_val, rho_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    M = M_lower(n)
    H_opt = solve_h(lambda H: cp.Minimize(mu * xi_expr(H, n) - rho_expr(H, n, M)),
                    n, verbose=verbose)
    if H_opt is None:
        return None, None, None
    return xi_val(H_opt, n), rho_val(H_opt, n, M), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="upper boundary for (ξ, ρ)")

    x_ref = np.linspace(-1/3, 1, 100)
    plot_boundary(
        pts,
        xlabel=r"Chatterjee's $\xi$", ylabel=r"Spearman's $\rho$",
        title="Attainable Region (Upper Boundary) for (ξ, ρ)",
        xlim=(-0.4, 1.05), ylim=(-1.05, 1.05),
        extra_plots=[
            lambda ax: ax.plot(x_ref, (3*x_ref - 1)/2, "r--",
                               label=r"Lower bound $\rho=(3\xi-1)/2$"),
            lambda ax: ax.axhline(y=1, color="g", ls="--", label=r"Upper bound $\rho=1$"),
        ],
    )
    plot_h_matrices(get_boundary_point, [0.05, 0.5, 1.0, 10.0],
                    title_fmt="h(t,v) for μ={mu:.2f} (ρ upper bound)")
    plt.show()
