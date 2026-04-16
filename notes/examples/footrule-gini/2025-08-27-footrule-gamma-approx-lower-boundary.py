"""Lower boundary of the (ψ, γ) attainable region.

Solves  minimize  μ·ψ(H) + γ(H)  for a sweep of μ values.
Both ψ and γ are linear in H, so this is a pure LP.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from boundary_solver import (
    solve_h, M_lower, J_anti,
    psi_expr, gamma_expr, psi_val, gamma_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    M, J = M_lower(n), J_anti(n)
    H_opt = solve_h(lambda H: cp.Minimize(mu * psi_expr(H, n, M) + gamma_expr(H, n, M, J)),
                    n, verbose=verbose)
    if H_opt is None:
        return None, None, None
    return psi_val(H_opt, n, M), gamma_val(H_opt, n, M, J), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="lower boundary for (ψ, γ)")
    plot_boundary(
        pts,
        xlabel=r"Spearman's footrule $\psi$", ylabel=r"Gini's $\gamma$",
        title="Attainable Region (Lower Boundary) for (ψ, γ)",
        xlim=(-0.05, 1.05), ylim=(-1.05, 1.05),
    )
    plot_h_matrices(get_boundary_point, [0.05, 0.5, 1.0, 10.0],
                    title_fmt="h(t,v) for μ={mu:.2f} (ψ–γ lower bound)")
    plt.show()
