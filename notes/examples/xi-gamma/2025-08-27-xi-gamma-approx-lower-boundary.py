"""Lower boundary of the (ξ, γ) attainable region.

Solves  minimize  μ·ξ(H) + γ(H)  for a sweep of μ values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from boundary_solver import (
    solve_h, M_lower, J_anti,
    xi_expr, gamma_expr, xi_val, gamma_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    M, J = M_lower(n), J_anti(n)
    H_opt = solve_h(
        lambda H: cp.Minimize(mu * xi_expr(H, n) + gamma_expr(H, n, M, J)),
        n, verbose=verbose,
    )
    if H_opt is None:
        return None, None, None
    return xi_val(H_opt, n), gamma_val(H_opt, n, M, J), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="lower boundary for (ξ, γ)")
    plot_boundary(
        pts,
        xlabel=r"Chatterjee's $\xi$", ylabel=r"Gini's $\gamma$",
        title="Attainable Region (Lower Boundary) for (ξ, γ)",
        xlim=(-0.05, 1.05), ylim=(-1.05, 1.05),
    )
    plot_h_matrices(
        get_boundary_point,
        [0.05, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2, 2.5, 3, 4, 5, 10.0, 20, 50],
        title_fmt="h(t,v) for μ={mu:.2f} (γ lower bound)",
    )
    plt.show()
