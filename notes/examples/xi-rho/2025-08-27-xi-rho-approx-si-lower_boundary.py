"""Lower boundary of the (ξ, ρ) region under the SI (Stochastically Increasing) constraint.

Uses the Convex-Concave Procedure (CCP) because  ξ(H) ≥ target  is non-convex.
SI adds the constraint H[i, :] ≥ H[i+1, :]  (h non-increasing in t).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from boundary_solver import plot_boundary, plot_h_matrix


# ── CCP solver ────────────────────────────────────────────────────────────────

def solve_lower_bound_ccp(target_xi, n=32, max_iter=10, verbose=False):
    """
    Minimize ρ for a fixed lower bound on ξ, under the SI constraint,
    using the Convex-Concave Procedure (CCP).
    """
    # Warm-start: comonotonic copula, column j filled in first j rows
    H_val = np.zeros((n, n))
    for j in range(n):
        H_val[:j, j] = 1.0

    coeff_xi = 6.0 / n**2
    M        = np.tril(np.ones((n, n)))
    xi_final = rho_final = H_opt_final = None

    for _ in range(max_iter):
        H = cp.Variable((n, n), name="H")
        constraints = [
            H >= 0, H <= 1,
            cp.sum(H, axis=0) == np.arange(n),
            H[:, :-1] <= H[:, 1:],   # non-decreasing in v
            H[:-1, :] >= H[1:, :],   # SI: non-increasing in t
        ]
        # Linearize ξ(H) around current H_val (CCP)
        xi_lin = coeff_xi * np.sum(H_val**2) + cp.sum(
            cp.multiply(2 * coeff_xi * H_val, H - H_val)
        )
        constraints.append(xi_lin >= target_xi)

        rho_term = (12 / n**3) * cp.sum(M @ H)
        problem  = cp.Problem(cp.Minimize(rho_term), constraints)
        try:
            problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
        except cp.SolverError:
            break
        if H.value is None:
            break

        H_val      = H.value
        xi_final   = coeff_xi * np.sum(H_val**2) - 2.0
        rho_final  = float((12 / n**3) * np.sum(M @ H_val) - 3.0)
        H_opt_final = H_val

    return xi_final, rho_final, H_opt_final


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target_xi_values = np.linspace(0, 1.0, 25)
    pts, H_maps = [], []

    print("Tracing the SI lower boundary (min ρ given ξ)…")
    for t_xi in tqdm(target_xi_values):
        xi, rho, H = solve_lower_bound_ccp(t_xi, n=32, max_iter=8)
        if xi is not None:
            pts.append((xi, rho))
            if any(np.isclose(t_xi, x, atol=0.02) for x in [0.0, 0.25, 0.5, 0.75, 1.0]):
                H_maps.append((xi, rho, H))
    pts = np.array(pts)

    x_grid = np.linspace(0, 1, 100)
    plot_boundary(
        pts,
        xlabel=r"Chatterjee's $\xi$", ylabel=r"Spearman's $\rho$",
        title="SI Copulas: min ρ for given ξ",
        xlim=(0, 1.05), ylim=(0, 1.05),
        extra_plots=[
            lambda ax: ax.plot(x_grid, x_grid, "k:", alpha=0.5, label=r"$\rho=\xi$"),
            lambda ax: ax.plot(x_grid, (3*x_grid - 1)/2, "r--", alpha=0.3,
                               label=r"General lower bound $\rho=(3\xi-1)/2$"),
        ],
    )

    if len(H_maps) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (xi_v, rho_v, H_mat) in zip(axes, H_maps[::max(1, len(H_maps)//3)][:3]):
            plot_h_matrix(H_mat, f"ξ≈{xi_v:.2f}, ρ≈{rho_v:.2f}", ax=ax)
        plt.suptitle("h(t,v) structures on the SI lower boundary")
    plt.show()
