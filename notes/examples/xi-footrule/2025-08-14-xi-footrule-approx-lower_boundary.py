"""Lower boundary of the (ξ, ψ) attainable region.

Solves  minimize  ψ(H) + μ·ξ(H)  for a sweep of μ values.
Uses the midpoint-corrected marginal (column sums = 0.5, 1.5, …, n−0.5).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from boundary_solver import (
    solve_h, M_lower,
    xi_expr, psi_expr, xi_val, psi_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    M = M_lower(n)
    H_opt = solve_h(lambda H: cp.Minimize(psi_expr(H, n, M) + mu * xi_expr(H, n)),
                    n, marginal_offset=0.5, verbose=verbose)
    if H_opt is None:
        return None, None, None
    return xi_val(H_opt, n), psi_val(H_opt, n, M), H_opt


if __name__ == "__main__":
    mu_values = np.logspace(-2, 1.5, 30)
    pts = sweep_boundary(get_boundary_point, mu_values, label="lower boundary for (ξ, ψ)")

    x_ref = np.linspace(0.001, 1, 100)
    plot_boundary(
        pts,
        xlabel=r"Chatterjee's $\xi$", ylabel=r"Spearman's footrule $\psi$",
        title="Attainable Region (Lower Boundary) for (ξ, ψ)",
        xlim=(-0.1, 1.05), ylim=(-0.55, 1.05),
        extra_plots=[lambda ax: ax.plot(x_ref, np.sqrt(x_ref), "r--",
                                        label=r"Upper bound $\psi=\sqrt{\xi}$")],
    )
    plot_h_matrices(get_boundary_point,
                    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2, 5, 10.0, 20],
                    title_fmt="h(t,v) for μ={mu:.2f}")
    plt.show()
