"""Lower boundary of the (ν, β) attainable region.

Solves  minimize  μ·ν(H) + β(H)  for a sweep of μ values.
β = 4·C(½,½) − 1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from boundary_solver import (
    solve_h, beta_half_params,
    nu_expr, beta_expr, nu_val, beta_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    k_u, col_w = beta_half_params(n)
    H_opt = solve_h(
        lambda H: cp.Minimize(mu * nu_expr(H, n) + beta_expr(H, k_u, col_w, n)),
        n, verbose=verbose,
    )
    if H_opt is None:
        return None, None, None
    return nu_val(H_opt, n), beta_val(H_opt, k_u, col_w, n), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="lower boundary for (ν, β)")
    plot_boundary(
        pts,
        xlabel=r"Blest's $\nu$", ylabel=r"Blomqvist's $\beta$",
        title="Attainable Region (Lower Boundary) for (ν, β)",
        xlim=(-0.05, 1.05), ylim=(-1.05, 1.05),
    )
    plot_h_matrices(get_boundary_point, [0.05, 0.5, 1.0, 10.0],
                    title_fmt="h(t,v) for μ={mu:.2f} (β lower bound)")
    plt.show()
