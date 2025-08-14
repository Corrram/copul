import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_tau_upper(mu, n=32, verbose=False):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative h that minimizes mu * xi - tau.
    This traces the upper boundary of the (xi, tau) region.
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
    # xi is a convex quadratic term (L2 norm of h).
    xi_term = (6 / n**2) * cp.sum_squares(H)

    # Kendall's Tau (tau) is related to 4 * E[C(U,V)].
    # This term is non-convex.
    M = np.tril(np.ones((n, n)))
    C_matrix = (1 / n) * (M @ H)
    dH_dv_matrix = n * cp.diff(H, axis=1)
    tau_term = 4 * cp.sum(cp.multiply(C_matrix[:, :-1], dH_dv_matrix)) / n**2

    # --- Define the objective function to be minimized ---
    # The objective is mu*xi - tau. This is a non-convex problem.
    objective = cp.Minimize(mu * xi_term - tau_term)

    # --- Solve the optimization problem ---
    problem = cp.Problem(objective, constraints)
    # Use a try-except block as the solver may fail due to non-convexity.
    try:
        problem.solve(
            solver=cp.OSQP, verbose=verbose, max_iter=25000, eps_abs=1e-6, eps_rel=1e-6
        )
    except cp.SolverError:
        print(f"Solver failed for mu={mu}, likely due to non-convexity.")
        return None, None, None

    # --- Return the results ---
    if H.value is not None:
        H_opt = H.value
        # The constants (-2 and -1) are added to normalize the values.
        xi_val = (6 / n**2) * np.sum(H_opt**2) - 2

        C_opt = (1 / n) * (M @ H_opt)
        dH_dv_opt = n * np.diff(H_opt, axis=1)
        tau_val = 4 * np.sum(np.multiply(C_opt[:, :-1], dH_dv_opt)) / n**2 - 1

        return xi_val, tau_val, H_opt
    else:
        # Return None if the solver fails.
        return None, None, None


# --- Main simulation loop ---
if __name__ == "__main__":
    # A range of mu values to trace the boundary.
    mu_values = np.logspace(-1, 2, 30)
    boundary_points = []

    print("Tracing the upper boundary for (xi, tau)...")
    for mu in tqdm(mu_values):
        xi, tau, _ = get_boundary_point_tau_upper(mu, n=32)
        if xi is not None and tau is not None:
            boundary_points.append((xi, tau))

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

    # For comparison, plot the known theoretical bounds |tau| <= xi
    x_theory = np.linspace(0, 1, 100)
    plt.plot(x_theory, -x_theory, "r--", label="Theoretical Lower Bound (τ = -ξ)")
    plt.plot(x_theory, x_theory, "g--", label="Theoretical Upper Bound (τ = ξ)")

    plt.title("Attainable Region for (ξ, τ)")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Kendall's τ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Compute and visualize H matrices for specific mu values ---
    print("\nComputing and saving H matrices to CSV files...")
    mu_for_files = [0.5, 1.0, 5.0, 10.0]
    n_vis = 64  # Use a higher resolution for visualization

    for mu_val in mu_for_files:
        # Compute the optimal H matrix
        _, _, H_map = get_boundary_point_tau_upper(mu=mu_val, n=n_vis)

        if H_map is not None:
            # --- Save to CSV ---
            filename = f"h_matrix_tau_upper_mu_{mu_val:.2f}.csv"
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (τ Upper Bound)")
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
