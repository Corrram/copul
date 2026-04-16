"""Lower boundary of the (ν, γ) attainable region.

Solves  minimize  μ·ν(H) + γ(H)  for a sweep of μ values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from boundary_solver import (
    solve_h, M_lower, J_anti,
    nu_expr, gamma_expr, nu_val, gamma_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    M, J = M_lower(n), J_anti(n)
    H_opt = solve_h(lambda H: cp.Minimize(mu * nu_expr(H, n) + gamma_expr(H, n, M, J)),
                    n, verbose=verbose)
    if H_opt is None:
        return None, None, None
    return nu_val(H_opt, n), gamma_val(H_opt, n, M, J), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="lower boundary for (ν, γ)")
    plot_boundary(
        pts,
        xlabel=r"Blest's $\nu$", ylabel=r"Gini's $\gamma$",
        title="Attainable Region (Lower Boundary) for (ν, γ)",
        xlim=(-0.05, 1.05), ylim=(-1.05, 1.05),
    )
    plot_h_matrices(get_boundary_point, [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                    title_fmt="h(t,v) for μ={mu:.2f} (γ lower bound)")
    plt.show()
