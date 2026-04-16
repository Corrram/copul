"""Lower boundary of the (ρ, γ) attainable region via LP.

For each fixed ρ target, minimizes γ(H) subject to ρ(H) == ρ_target
and the standard copula-derivative constraints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from boundary_solver import M_lower, J_anti, plot_boundary, plot_h_matrix


# ── Solver ────────────────────────────────────────────────────────────────────

def solve_min_gamma_given_rho(rho_target, n=32, verbose=False):
    """LP: minimize γ(H) subject to ρ(H) == rho_target + standard constraints."""
    H = cp.Variable((n, n), name="H")
    M, J = M_lower(n), J_anti(n)

    rho_part   = (12 / n**3) * cp.sum(M @ H)
    gamma_part = (4 / n**2) * (cp.trace(M @ H) + cp.trace(M @ H @ J))

    constraints = [
        H >= 0, H <= 1,
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
        rho_part - 3 == rho_target,
    ]
    problem = cp.Problem(cp.Minimize(gamma_part), constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=verbose, eps_abs=1e-5, eps_rel=1e-5)
    except Exception:
        problem.solve(solver=cp.SCS, verbose=verbose)

    if H.value is None:
        return None, None, None
    H_opt = H.value
    rho_val   = float((12 / n**3) * np.sum(M @ H_opt) - 3)
    gamma_val = float((4 / n**2) * (np.trace(M @ H_opt) + np.trace(M @ H_opt @ J)) - 2)
    return rho_val, gamma_val, H_opt


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rho_targets = np.linspace(-0.99, 0.99, 40)
    pts = []

    print("Tracing min(γ) for given ρ…")
    for r in tqdm(rho_targets):
        rho_res, gamma_res, _ = solve_min_gamma_given_rho(r, n=32)
        if rho_res is not None:
            pts.append((rho_res, gamma_res))
    pts = np.array(pts)

    plot_boundary(
        pts,
        xlabel=r"Spearman's $\rho$", ylabel=r"Gini's $\gamma$",
        title=r"Lower Boundary of Attainable Region $(\rho, \gamma)$",
        xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
        extra_plots=[lambda ax: ax.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="Identity")],
    )

    for r_val in [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]:
        _, gamma_got, H_map = solve_min_gamma_given_rho(r_val, n=64)
        if H_map is not None:
            plot_h_matrix(H_map,
                          f"h(t,v): ρ={r_val:.2f} → γ={gamma_got:.2f} (min γ)")
    plt.show()
