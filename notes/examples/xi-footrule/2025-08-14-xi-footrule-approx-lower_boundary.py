import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# The get_boundary_point function remains unchanged.
def get_boundary_point(mu, n=32, verbose=False):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative H that minimizes psi + mu * xi.
    """
    H = cp.Variable((n, n), name="H")
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n) + 0.5,
        H[:, :-1] <= H[:, 1:],
    ]
    xi_term = (6 / n**2) * cp.sum_squares(H)
    M = np.tril(np.ones((n, n)))
    psi_term = (6 / n**2) * cp.trace(M @ H)
    objective = cp.Minimize(psi_term + mu * xi_term)
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.OSQP, verbose=verbose, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5
    )

    if H.value is not None:
        H_opt = H.value
        xi_val = (6 / n**2) * np.sum(H_opt**2) - 2
        psi_val = (6 / n**2) * np.trace(M @ H_opt) - 2
        return xi_val, psi_val, H_opt
    else:
        return None, None, None


# --- Main simulation loop ---
if __name__ == "__main__":
    # A range of mu values to trace the boundary
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the lower boundary for (xi, psi)...")
    for mu in tqdm(mu_values):
        xi, psi, _ = get_boundary_point(mu, n=32)
        if xi is not None:
            boundary_points.append((xi, psi))

    # --- Plot 1: The attainable region boundary ---
    plt.figure(figsize=(8, 8))
    boundary_points = np.array(boundary_points)
    plt.plot(
        boundary_points[:, 0],
        boundary_points[:, 1],
        "o-",
        label="Numerical Lower Bound",
    )
    x_theory = np.linspace(0.001, 1, 100)
    plt.plot(x_theory, np.sqrt(x_theory), "r--", label="Upper Bound (y = sqrt(x))")
    plt.title("Attainable Region for (ξ, ψ)")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Spearman's Footrule ψ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.1, 1.05)
    plt.ylim(-0.55, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Compute H matrices and save them to CSV files ---
    mu_for_files = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2, 5, 10.0, 20]
    n_vis = 64  # Use a higher resolution for visualization

    for mu_val in mu_for_files:
        # Compute the optimal H matrix
        _, _, H_map = get_boundary_point(mu=mu_val, n=n_vis)

        if H_map is not None:
            # --- VISUALIZE (optional but helpful) ---
            plt.figure(figsize=(7, 6))
            ax = plt.gca()
            im = ax.imshow(
                H_map,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f}")
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
            print(f"Solver failed for μ={mu_val}, could not save file.")

    # Display all created figures
    plt.show()
