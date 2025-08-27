import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def _half_indices_and_weights(n, u_star=0.5, v_star=0.5):
    """
    Utilities for discretizing C(u*, v*) ≈ \int_0^{u*} h(t, v*) dt
    over the midpoint grid t_i = (i+0.5)/n, v_j = (j+0.5)/n.

    Returns:
      k_u : number of rows i with t_i <= u*
      col_weights : length-n array w over columns such that
                    h(., v*) ≈ sum_j w[j] * h(., v_j)  (linear interp in v)
    """
    t_mid = (np.arange(n) + 0.5) / n
    v_mid = (np.arange(n) + 0.5) / n

    # rows with t_i <= u*
    k_u = int(np.sum(t_mid <= u_star))  # equals ceil(n/2) for u*=0.5

    # columns around v*
    j_right = int(np.searchsorted(v_mid, v_star, side="right"))
    j_left = j_right - 1
    if j_left < 0:
        j_left, j_right = 0, 0
        lam = 0.0
    elif j_right >= n:
        j_left, j_right = n - 1, n - 1
        lam = 0.0
    else:
        # linear interpolation weight toward the right column
        lam = (v_star - v_mid[j_left]) / (v_mid[j_right] - v_mid[j_left])

    col_weights = np.zeros(n)
    col_weights[j_left] += 1.0 - lam
    col_weights[j_right] += lam
    return k_u, col_weights


def get_boundary_point_beta_upper(mu, n=32, verbose=False, solver="OSQP"):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative h that minimizes  mu * xi - beta.
    This traces the upper boundary of the (xi, beta) region.

    Discretization (midpoint grid):
      - t_i = (i+0.5)/n,  v_j = (j+0.5)/n  for i,j = 0,...,n-1
      - H[i,j] ≈ h(t_i, v_j)

    Constraints:
      - 0 <= H <= 1
      - sum_i H[i, j] == j                 (enforces ∫ h(·, v_j) dt = v_j up to scaling)
      - H[:, j] <= H[:, j+1]               (monotone in v)

    Objectives (constant offsets dropped):
      - xi  ≈ (6 / n^2) * sum(H^2)
      - beta = 4 * C(1/2, 1/2) - 1, with
               C(1/2,1/2) ≈ (1/n) * sum_{i: t_i <= 1/2} h(t_i, v=1/2)
               and v=1/2 obtained by linear interpolation between adjacent v_j.
    """
    # CVXPY variable
    H = cp.Variable((n, n), name="H")

    # --- Constraints ---
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),  # column sums enforce marginals
        H[:, :-1] <= H[:, 1:],  # nondecreasing in v
    ]

    # --- Objective terms ---
    # xi term: quadratic in H
    xi_term = (6 / n**2) * cp.sum_squares(H)

    # beta term: from C(1/2,1/2)
    k_u, col_weights = _half_indices_and_weights(n, u_star=0.5, v_star=0.5)
    # sum over first k_u rows, with column mixing for v*=1/2
    # C_half = (1/n) * sum_{i < k_u} sum_j col_weights[j] * H[i,j]
    C_half = (1 / n) * cp.sum(cp.multiply(H[:k_u, :], col_weights[None, :]))
    beta_term = 4 * C_half - 1  # linear in H

    # We minimize mu * xi - beta  (maximize beta at fixed xi)
    objective = cp.Minimize(mu * xi_term - beta_term)

    # --- Solve ---
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
        # Fallback to SCS if chosen solver fails
        problem.solve(solver="SCS", verbose=verbose, max_iters=50000)

    # --- Return ---
    if H.value is None:
        return None, None, None

    H_opt = H.value
    xi_val = (6 / n**2) * np.sum(H_opt**2) - 2
    # recompute beta using the same discretization
    k_u_val, col_w_val = _half_indices_and_weights(n, 0.5, 0.5)
    C_half_val = (1 / n) * np.sum(H_opt[:k_u_val, :] * col_w_val[None, :])
    beta_val = 4 * C_half_val - 1
    return xi_val, beta_val, H_opt


# --- Main simulation loop ---
if __name__ == "__main__":
    # Range of mu to trace the upper boundary; adjust density as desired
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the upper boundary for (xi, beta)...")
    for mu in tqdm(mu_values):
        xi, beta, _ = get_boundary_point_beta_upper(mu, n=32)
        if xi is not None:
            boundary_points.append((xi, beta))

    # --- Plot: Attainable upper boundary (numerical) ---
    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Upper Boundary (ξ–β)",
        )

    # (Optional) If you have a closed-form upper boundary for (ξ, β),
    # you can overlay it here. Left commented to avoid asserting a formula.
    # x_theory = np.linspace(0.0, 1.0, 200)
    # beta_theory = ...  # supply known expression if available
    # plt.plot(x_theory, beta_theory, "k--", label="Theoretical upper boundary")

    plt.title("Attainable Region (Upper Boundary) for (ξ, β)")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Blomqvist's β")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Visualize H(t,v) for a few μ values ---
    mu_for_files = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_vis = 64

    for mu_val in mu_for_files:
        _, _, H_map = get_boundary_point_beta_upper(mu=mu_val, n=n_vis)
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (β Upper Bound)")
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
