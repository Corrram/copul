"""Lower boundary of the (ρ, ν) attainable region.

Solves  minimize  ν(H) − μ·ρ(H)  for a sweep of μ values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cvxpy as cp
from boundary_solver import (
    solve_h, M_lower,
    rho_expr, nu_expr, rho_val, nu_val,
    sweep_boundary, plot_boundary, plot_h_matrices,
)


def get_boundary_point(mu, n=32, verbose=False):
    M = M_lower(n)
    H_opt = solve_h(lambda H: cp.Minimize(nu_expr(H, n) - mu * rho_expr(H, n, M)),
                    n, verbose=verbose)
    if H_opt is None:
        return None, None, None
    return rho_val(H_opt, n, M), nu_val(H_opt, n), H_opt


if __name__ == "__main__":
    pts = sweep_boundary(get_boundary_point, label="lower boundary for (ρ, ν)")
    plot_boundary(
        pts,
        xlabel=r"Spearman's $\rho$", ylabel=r"Blest's $\nu$",
        title="Attainable Region (Lower Boundary) for (ρ, ν)",
        xlim=(-1.05, 1.05), ylim=(-1.05, 1.05),
        extra_plots=[
            lambda ax: ax.plot([0],  [0],  "ks", label="Independence (ρ=0, ν=0)"),
            lambda ax: ax.plot([-1], [-1], "k^", label="Countermonotone (ρ=-1, ν=-1)"),
        ],
    )
    plot_h_matrices(get_boundary_point, [0.05, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0],
                    title_fmt="h(t,v) for μ={mu:.2f} (ρ–ν lower bound)")
    pathlib.Path("images").mkdir(parents=True, exist_ok=True)
    plt.savefig("images/rho_nu_lower_boundary.png", dpi=150)
    plt.show()
