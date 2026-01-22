import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def solve_max_gamma_given_rho(rho_target, n=32, verbose=False):
    """
    Solves the Linear Program:
        Maximize gamma(H)
        Subject to:
            rho(H) == rho_target
            Standard copula derivative constraints on H

    Since both gamma and rho are linear in H, this is a Linear Program (LP).
    """
    H = cp.Variable((n, n), name="H")

    # --- Helper Matrices ---
    # M: Lower triangular ones (integration operator)
    M = np.tril(np.ones((n, n)))
    # J: Anti-diagonal identity (for the secondary diagonal part of gamma)
    J = np.fliplr(np.eye(n))

    # --- Functional Definitions (Discretized) ---

    # Spearman's Rho: proportional to the volume under the copula
    # rho = (12 / n^3) * sum(M @ H) - 3
    rho_linear_part = (12 / n**3) * cp.sum(M @ H)

    # Gini's Gamma: proportional to trace (diagonal) and anti-trace (anti-diagonal) mass
    # gamma = (4 / n^2) * (trace(M @ H) + trace(M @ H @ J)) - 2
    gamma_linear_part = (4 / n**2) * (cp.trace(M @ H) + cp.trace(M @ H @ J))

    # --- Constraints ---
    constraints = [
        H >= 0,
        H <= 1,
        # Uniform marginals (column sums increase linearly)
        cp.sum(H, axis=0) == np.arange(n),
        # Monotonicity in v (columns must be non-decreasing)
        H[:, :-1] <= H[:, 1:],
        # The Target Rho Constraint
        rho_linear_part - 3 == rho_target,
    ]

    # --- Objective ---
    objective = cp.Maximize(gamma_linear_part)

    # --- Solve ---
    problem = cp.Problem(objective, constraints)

    # LPs are generally robust, but we include a fallback just in case
    try:
        problem.solve(solver=cp.OSQP, verbose=verbose, eps_abs=1e-5, eps_rel=1e-5)
    except Exception:
        problem.solve(solver=cp.SCS, verbose=verbose)

    if H.value is None:
        return None, None, None

    # Recover exact numerical values
    H_opt = H.value
    rho_val = (12 / n**3) * np.sum(M @ H_opt) - 3
    gamma_val = (4 / n**2) * (np.trace(M @ H_opt) + np.trace(M @ H_opt @ J)) - 2

    return rho_val, gamma_val, H_opt


if __name__ == "__main__":
    # Sweep rho from -1 to 1 to trace the boundary
    # Note: We avoid exactly -1 and 1 slightly to prevent numerical stiffness
    # if the discretization grid doesn't perfectly support the extreme singular copulas.
    rho_targets = np.linspace(-0.99, 0.99, 40)

    boundary_results = []

    print("Tracing max(γ) for given ρ...")
    for r in tqdm(rho_targets):
        rho_res, gamma_res, _ = solve_max_gamma_given_rho(r, n=32)
        if rho_res is not None:
            boundary_results.append((rho_res, gamma_res))

    # --- Plotting the Boundary ---
    boundary_results = np.array(boundary_results)

    plt.figure(figsize=(8, 8))
    plt.plot(
        boundary_results[:, 0],
        boundary_results[:, 1],
        "o-",
        label=r"Max $\gamma$ given $\rho$",
    )

    # Reference lines
    plt.plot([-1, 1], [-1, 1], "k--", alpha=0.3, label="Identity line")
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)

    plt.title(r"Boundary of Attainable Region $(\rho, \gamma)$")
    plt.xlabel(r"Spearman's $\rho$")
    plt.ylabel(r"Gini's $\gamma$")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect("equal")

    # --- Visualization of H for specific Rho values ---
    # We pick interesting points: negative correlation, zero, positive, high positive
    target_rhos_vis = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]
    n_vis = 64

    for r_val in target_rhos_vis:
        _, gamma_got, H_map = solve_max_gamma_given_rho(r_val, n=n_vis)

        if H_map is not None:
            plt.figure(figsize=(6, 5))
            plt.imshow(
                H_map,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            plt.colorbar(label="h(t,v)")
            plt.title(
                f"Structure of h(t,v)\nTarget $\\rho={r_val:.2f}$ (Result $\\gamma={gamma_got:.2f}$)"
            )
            plt.xlabel("v")
            plt.ylabel("t")
        else:
            print(f"Solver failed for rho={r_val}")

    plt.show()
