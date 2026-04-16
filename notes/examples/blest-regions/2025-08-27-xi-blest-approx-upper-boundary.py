"""Upper boundary of the (ξ, ν) attainable region.

Solves  minimize  μ·ξ(H) − ν(H)  for a sweep of μ values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from boundary_solver import (
    solve_h, xi_expr, nu_expr, xi_val, nu_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    H_opt = solve_h(lambda H: cp.Minimize(mu * xi_expr(H, n) - nu_expr(H, n)),
                    n, verbose=verbose)
    if H_opt is None:
        return None, None, None
    return xi_val(H_opt, n), nu_val(H_opt, n), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="upper boundary for (ξ, ν)")
    plot_boundary(
        pts,
        xlabel=r"Chatterjee's $\xi$", ylabel=r"Blest's $\nu$",
        title="Attainable Region (Upper Boundary) for (ξ, ν)",
        xlim=(-0.05, 1.05), ylim=(-1.05, 1.05),
    )
    plot_h_matrices(get_boundary_point, [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                    title_fmt="h(t,v) for μ={mu:.2f} (ν upper bound)")
    plt.show()
