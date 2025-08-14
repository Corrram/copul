import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_psi_rho(mu, n=32, verbose=False):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative h that minimizes mu * psi - rho.
    This traces the upper boundary of the (psi, rho) region.
    """
    # Define the CVXPY variable for the discretized copula derivative h(t,v).
    H = cp.Variable((n, n), name="H")

    # --- Define the constraints for H ---
    # These constraints ensure that H corresponds to a valid copula derivative.
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
    ]

    # --- Define the objective function terms ---
    M = np.tril(np.ones((n, n)))

    # Spearman's Footrule (psi) term
    psi_term = (6 / n**2) * cp.trace(M @ H)

    # Spearman's Rho (rho) term
    rho_term = (12 / n**3) * cp.sum(M @ H)

    # --- Define the objective function to be minimized ---
    # We are minimizing mu * psi - rho.
    objective = cp.Minimize(mu * psi_term - rho_term)

    # --- Solve the optimization problem ---
    problem = cp.Problem(objective, constraints)
    problem.solve(
        solver=cp.OSQP, verbose=verbose, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5
    )

    # --- Return the results ---
    if H.value is not None:
        H_opt = H.value
        # The constants (-2 and -3) are added to normalize the values.
        psi_val = (6 / n**2) * np.trace(M @ H_opt) - 2
        rho_val = (12 / n**3) * np.sum(M @ H_opt) - 3
        return psi_val, rho_val, H_opt
    else:
        # Return None if the solver fails.
        return None, None, None


# --- Main simulation loop ---
if __name__ == "__main__":
    # A range of mu values to trace the boundary.
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the upper boundary for (psi, rho)...")
    for mu in tqdm(mu_values):
        psi, rho, _ = get_boundary_point_psi_rho(mu, n=32)
        if psi is not None and rho is not None:
            boundary_points.append((psi, rho))

    # --- Plot 1: The attainable region boundary ---
    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Upper Bound",
        )

    # For comparison, plot the known theoretical bounds
    psi_theory = np.linspace(-0.5, 1, 100)
    rho_lower_bound = (3 * psi_theory - 1) / 2
    rho_upper_bound = np.minimum(2 * psi_theory + 1, 1)  # A tighter upper bound

    plt.plot(psi_theory, rho_lower_bound, "r--", label="Theoretical Lower Bound")
    plt.plot(psi_theory, rho_upper_bound, "g--", label="Theoretical Upper Bound")

    plt.title("Attainable Region for (ψ, ρ)")
    plt.xlabel("Spearman's Footrule ψ")
    plt.ylabel("Spearman's ρ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.55, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Compute and visualize H matrices for specific mu values ---
    print("\nComputing and saving H matrices to CSV files...")
    mu_for_files = [0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_vis = 64  # Use a higher resolution for visualization

    for mu_val in mu_for_files:
        # Compute the optimal H matrix
        _, _, H_map = get_boundary_point_psi_rho(mu=mu_val, n=n_vis)

        if H_map is not None:
            # --- Save to CSV ---
            filename = f"h_matrix_psi_rho_upper_mu_{mu_val:.2f}.csv"
            np.savetxt(filename, H_map, delimiter=",")
            print(f"Successfully saved data to {filename}")

            # --- Visualize the H matrix ---
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (ψ-ρ Upper Bound)")
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
