"""Upper boundary of the (ξ, ψ) attainable region.

Solves  minimize  μ·ξ(H) − ψ(H)  for a sweep of μ values.
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
    H_opt = solve_h(lambda H: cp.Minimize(mu * xi_expr(H, n) - psi_expr(H, n, M)),
                    n, verbose=verbose)
    if H_opt is None:
        return None, None, None
    return xi_val(H_opt, n), psi_val(H_opt, n, M), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="upper boundary for (ξ, ψ)")

    x_ref = np.linspace(0.001, 1, 100)
    plot_boundary(
        pts,
        xlabel=r"Chatterjee's $\xi$", ylabel=r"Spearman's footrule $\psi$",
        title="Attainable Region (Upper Boundary) for (ξ, ψ)",
        xlim=(-0.1, 1.05), ylim=(-0.55, 1.05),
        extra_plots=[lambda ax: ax.plot(x_ref, np.sqrt(x_ref), "r--",
                                        label=r"Theoretical upper bound $\psi=\sqrt{\xi}$")],
    )
    plot_h_matrices(get_boundary_point, [0.05, 0.5, 1.0, 10.0],
                    title_fmt="h(t,v) for μ={mu:.2f} (ψ upper bound)")
    plt.show()
