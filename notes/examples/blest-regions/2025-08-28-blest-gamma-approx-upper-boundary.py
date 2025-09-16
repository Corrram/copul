import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_nu_gamma_upper(mu, n=32, verbose=False, solver="OSQP"):
    """
    Solve the discretized problem for a given mu:
        minimize_H   mu * nu(H) - gamma(H)
    to trace the upper boundary of the (nu, gamma) region.

    Discretization & constraints (midpoint grid):
      - H[i,j] ~ h(t_i, v_j),  t_i=v_j=(i+0.5)/n
      - 0 <= H <= 1
      - column sums enforce marginals: sum_i H[i,j] == j
      - monotone in v: H[:, j] <= H[:, j+1]

    Objective pieces:
      - nu ≈ (12 / n^2) * sum((1 - t_i)^2 * H[i,j]) - 2
      - gamma = (4 / n^2) * (trace(MH) + trace(MHJ)) - 2,
        with M lower-triangular ones, J the anti-diagonal flip.
      - Drop constants (-2) inside optimizer; add back on evaluation.
    """
    H = cp.Variable((n, n), name="H")

    # Constraints
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),  # marginals (scaled)
        H[:, :-1] <= H[:, 1:],  # monotone in v
    ]

    # Precompute grid weights
    t_mid = (np.arange(n) + 0.5) / n
    w = (1.0 - t_mid) ** 2

    # nu term (without -2)
    nu_term = (12 / n**2) * cp.sum(cp.multiply(w[:, None], H))

    # gamma term (without -2)
    M = np.tril(np.ones((n, n)))
    J = np.fliplr(np.eye(n))
    gamma_term = (4 / n**2) * (cp.trace(M @ H) + cp.trace(M @ H @ J))

    # Objective: minimize mu * nu - gamma
    objective = cp.Minimize(mu * nu_term - gamma_term)

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

    # Evaluate functionals
    H_opt = H.value
    nu_val = (12 / n**2) * np.sum(w[:, None] * H_opt) - 2
    gamma_val = (4 / n**2) * (np.trace(M @ H_opt) + np.trace(M @ H_opt @ J)) - 2

    return nu_val, gamma_val, H_opt


# --- Main simulation loop ---
if __name__ == "__main__":
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the upper boundary for (nu, gamma)...")
    for mu in tqdm(mu_values):
        nu, gamma, _ = get_boundary_point_nu_gamma_upper(mu, n=32)
        if nu is not None:
            boundary_points.append((nu, gamma))

    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        boundary_points = boundary_points[np.argsort(boundary_points[:, 0])]
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Upper Boundary (ν–γ)",
        )

    plt.title("Attainable Region (Upper Boundary) for (ν, γ)")
    plt.xlabel("Blest's ν")
    plt.ylabel("Gini's γ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # Visualize H(t,v) for selected μ
    mu_for_files = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    n_vis = 64
    for mu_val in mu_for_files:
        _, _, H_map = get_boundary_point_nu_gamma_upper(mu=mu_val, n=n_vis)
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (γ Upper Bound)")
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
