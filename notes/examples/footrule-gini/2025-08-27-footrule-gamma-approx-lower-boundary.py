import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_psi_gamma_lower(mu, n=32, verbose=False, solver="OSQP"):
    """
    Solve:
        minimize_H   mu * psi(H) + gamma(H)
    to trace the LOWER boundary of the (psi, gamma) region.

    Midpoint discretization t_i = v_j = (i+0.5)/n with H[i,j] ~ h(t_i, v_j):

      Constraints:
        0 <= H <= 1
        sum_i H[i, j] == j                 (uniform margins)
        H[:, j] <= H[:, j+1]               (nondecreasing in v)

      Build C ≈ M @ H, where M is lower-triangular ones.

      Functionals (constants dropped in optimizer, restored in reporting):
        psi(H)   ≈ (6 / n^2) * tr(M H)                 # Spearman's Footrule
        gamma(H) ≈ (4 / n^2) * [tr(M H) + tr(M H J)]   # Gini's gamma
        (J is the anti-diagonal permutation)

    Lower boundary: minimize mu*psi + gamma.
    """
    H = cp.Variable((n, n), name="H")

    # Feasibility constraints for copula derivative
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
    ]

    # Helpers
    M = np.tril(np.ones((n, n)))  # integrates over u: C ≈ M @ H
    J = np.fliplr(np.eye(n))  # anti-diagonal permutation

    # Objective pieces (without constants)
    psi_term = (6 / n**2) * cp.trace(M @ H)
    gamma_term = (4 / n**2) * (cp.trace(M @ H) + cp.trace(M @ H @ J))

    # LOWER boundary: penalize gamma (plus sign)
    objective = cp.Minimize(mu * psi_term + gamma_term)

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=solver,
            verbose=verbose,
            max_iter=20000,
            eps_abs=1e-5,
            eps_rel=1e-5,
            warm_start=True,
        )
    except Exception:
        problem.solve(solver="SCS", verbose=verbose, max_iters=50000)

    if H.value is None:
        return None, None, None

    H_opt = H.value

    # Report ψ and γ with their conventional constants
    psi_val = (6 / n**2) * np.trace(M @ H_opt) - 2
    gamma_val = (4 / n**2) * (np.trace(M @ H_opt) + np.trace(M @ H_opt @ J)) - 2
    return psi_val, gamma_val, H_opt


# --- Main sweep & plots ---
if __name__ == "__main__":
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the LOWER boundary for (ψ, γ)...")
    for mu in tqdm(mu_values):
        psi, gamma, _ = get_boundary_point_psi_gamma_lower(mu, n=32)
        if psi is not None:
            boundary_points.append((psi, gamma))

    # Plot numerical boundary
    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Lower Boundary (ψ–γ)",
        )

    plt.title("Attainable Region (Lower Boundary) for (ψ, γ)")
    plt.xlabel("Spearman's Footrule ψ")
    plt.ylabel("Gini's γ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # Visualize h(t,v) for a few μ values
    mu_for_files = [0.05, 0.5, 1.0, 10.0]
    n_vis = 64
    for mu_val in mu_for_files:
        _, _, H_map = get_boundary_point_psi_gamma_lower(mu=mu_val, n=n_vis)
        if H_map is not None:
            plt.figure(figsize=(7, 6))
            ax = plt.gca()
            im = ax.imshow(
                H_map,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="viridis",
                vmin=0,
                vmax=1,
                aspect="auto",
            )
            ax.set_title(f"h(t,v) for μ = {mu_val:.2f}  (ψ–γ Lower Boundary)")
            ax.set_xlabel("v")
            ax.set_ylabel("t")
            plt.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.046,
                pad=0.04,
                label="h(t,v)",
            )
        else:
            print(f"Solver failed for μ={mu_val}.")

    plt.show()
