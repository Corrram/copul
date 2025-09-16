import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_nu_upper(mu, n=32, verbose=False, solver="OSQP"):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative h that minimizes  mu * xi - nu.
    This traces the upper boundary of the (xi, nu) region.

    Discretization:
      - t, v in { (i+0.5)/n : i=0,...,n-1 }  (midpoint grid)
      - H[i,j] ~ h(t_i, v_j)
      - Constraints:
          0 <= H <= 1
          sum_i H[i, j] == j               (enforces ∫ h(·, v_j) dt = v_j up to scaling)
          H[:, j] <= H[:, j+1]             (monotone in v)
      - xi ≈ (6 / n^2) * sum(H^2) - 2
      - nu ≈ (12 / n^2) * sum( ((1 - t_i)^2) * H[i,j] ) - 2
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

    # nu term: linear in H with weight w_i = (1 - t_i)^2
    t_mid = (np.arange(n) + 0.5) / n
    w = (1.0 - t_mid) ** 2  # shape (n,)
    # Multiply each row i by w[i]
    nu_term = (12 / n**2) * cp.sum(cp.multiply(w[:, None], H))

    # We minimize mu * xi - nu  (constant offsets in xi/nu don't affect the argmin)
    objective = cp.Minimize(mu * xi_term - nu_term)

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
    nu_val = (12 / n**2) * np.sum(w[:, None] * H_opt) - 2
    return xi_val, nu_val, H_opt


# --- Main simulation loop ---
if __name__ == "__main__":
    # Range of mu to trace the upper boundary; adjust density as desired
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the upper boundary for (xi, nu)...")
    for mu in tqdm(mu_values):
        xi, nu, _ = get_boundary_point_nu_upper(mu, n=32)
        if xi is not None:
            boundary_points.append((xi, nu))

    # --- Plot: Attainable upper boundary (numerical) + theoretical curve nu = sqrt(xi) ---
    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Upper Boundary (ξ–ν)",
        )

    # Theoretical upper boundary (Fréchet family): nu = sqrt(xi), xi in [0,1]
    x_theory = np.linspace(0.0, 1.0, 200)
    plt.plot(x_theory, np.sqrt(x_theory), "k--", label=r"Theoretical: $\nu=\sqrt{\xi}$")

    plt.title("Attainable Region (Upper Boundary) for (ξ, ν)")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Blest's ν")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Visualize H(t,v) for a few μ values ---
    mu_for_files = [0.05, 0.5, 1.0, 10.0]
    n_vis = 64

    for mu_val in mu_for_files:
        _, _, H_map = get_boundary_point_nu_upper(mu=mu_val, n=n_vis)
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (ν Upper Bound)")
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
