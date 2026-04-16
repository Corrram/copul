"""Upper boundary of the (ξ, ρ) region under the LTI (Left-Tail Increasing) constraint.

Uses a μ-sweep with the LTI constraint added per-row.  The objective and ρ functional
use a trapezoid-weighted quadrature (different from the standard rho_expr).
The solver cascade is OSQP → ECOS → SCS.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from boundary_solver import plot_boundary, plot_h_matrix


# ── LTI-specific helpers ──────────────────────────────────────────────────────

def _build_C(H):
    """C ≈ (1/n) · M @ H  (cumulative in u, affine in H)."""
    n = H.shape[0]
    return (1.0 / n) * (np.tril(np.ones((n, n))) @ H)


def _rho_trapezoid(C):
    """ρ via trapezoid weights in both u and v (linear in C)."""
    n = C.shape[0]
    w = np.ones(n); w[0] = w[-1] = 0.5
    W = np.outer(w, w)
    return 12.0 * (1.0 / n**2) * cp.sum(cp.multiply(W, C)) - 3.0


def get_boundary_point(mu, n=32, verbose=False):
    """LTI upper boundary: minimize  μ·ξ(H) − ρ(H)  subject to LTI + standard."""
    H = cp.Variable((n, n), name="H")
    u = (np.arange(n) + 1.0) / n
    C = _build_C(H)

    constraints = [
        H >= 0, H <= 1,
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
    ]
    # LTI: C(u,v)/u non-decreasing ⟺ u_i · C_{i+1,j} ≤ u_{i+1} · C_{i,j}
    for i in range(n - 1):
        constraints.append(cp.multiply(u[i], C[i + 1, :]) <= cp.multiply(u[i + 1], C[i, :]))

    xi_unshifted = (6.0 / n**2) * cp.sum_squares(H)
    mu_eff = max(mu, 1e-6)
    problem = cp.Problem(cp.Minimize(mu_eff * xi_unshifted + _rho_trapezoid(C)), constraints)

    solved = False
    for solver_kwargs in [
        dict(solver=cp.OSQP, max_iter=60000, eps_abs=1e-7, eps_rel=1e-7),
        dict(solver=cp.ECOS, max_iters=12000, abstol=1e-9, reltol=1e-9, feastol=1e-9),
        dict(solver=cp.SCS,  max_iters=100000, eps=2e-6),
    ]:
        try:
            problem.solve(verbose=verbose, **solver_kwargs)
            solved = H.value is not None
        except Exception:
            solved = False
        if solved:
            break
    if not solved:
        return None, None, None

    H_opt = H.value
    xi_val  = float((6.0 / n**2) * np.sum(H_opt**2) - 2.0)
    C_num   = (1.0 / n) * (np.tril(np.ones((n, n))) @ H_opt)
    w       = np.ones(n); w[0] = w[-1] = 0.5
    W       = np.outer(w, w)
    rho_val = float(12.0 * (1.0 / n**2) * np.sum(W * C_num) - 3.0)
    return xi_val, rho_val, H_opt


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mu_values = np.r_[0.0, np.logspace(-3, 2, 36)]
    pts, H_examples = [], {}

    print("Tracing the LTI upper boundary via μ-sweep…")
    for mu in tqdm(mu_values):
        xi, rho, H = get_boundary_point(mu, n=32)
        if xi is not None:
            pts.append((xi, rho))
            if any(np.isclose(mu, x, atol=1e-3) for x in [0.0, 1e-3, 1e-1, 1.0, 10.0]):
                H_examples[float(round(mu, 6))] = H
    pts = np.array(pts)

    plot_boundary(
        pts,
        xlabel=r"Chatterjee's $\xi$", ylabel=r"Spearman's $\rho$",
        title="LTI (ξ, ρ) — upper boundary via μ-sweep",
        xlim=(-0.05, 1.05), ylim=(-1.05, 1.05),
        extra_plots=[lambda ax: ax.scatter([0], [0], marker="x", s=80, label="Independence")],
    )

    for mu_key, H_map in H_examples.items():
        plot_h_matrix(H_map, f"h(t,v) for μ={mu_key} (LTI upper boundary)")
    plt.show()
