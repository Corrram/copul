import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point(mu, n=32, verbose=False):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative h that minimizes mu * xi - psi.
    This traces the upper boundary of the (xi, psi) region.
    """
    # Define the CVXPY variable for the discretized copula derivative h(t,v).
    # In the code, this is represented by the matrix H.
    H = cp.Variable((n, n), name="H")

    # --- Define the constraints for H ---
    # These constraints ensure that H corresponds to a valid copula.
    # 1. 0 <= h(t,v) <= 1
    # 2. Integral of h(t,v) dv from 0 to 1 is t. Discretized: sum over columns is proportional to column index.
    # 3. H(t,v) must be non-decreasing in v for fixed t.
    constraints = [
        H >= 0,
        H <= 1,
        # cp.sum(H, axis=0) represents the integral over t for each v.
        # This constraint seems to be for the partial derivative w.r.t t, dC/dt = H.
        # The integral of dC/dt dv should be C(t,1) = t.
        # The provided constraint seems to be sum(H, axis=0) == np.arange(n),
        # which might be a slight variation or a specific discretization choice.
        # Let's stick to the user's original formulation.
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
    ]

    # --- Define the objective function terms ---
    # xi is related to the L2 norm of h.
    xi_term = (6 / n**2) * cp.sum_squares(H)

    # psi is related to the integral of C(t,t).
    # M is a lower-triangular matrix of ones used to compute the cumulative sum,
    # which corresponds to the integral to get C from H.
    M = np.tril(np.ones((n, n)))
    psi_term = (6 / n**2) * cp.trace(M @ H)

    # --- Define the objective function to be minimized ---
    # This is the key change: we now minimize (mu * xi - psi).
    objective = cp.Minimize(mu * xi_term - psi_term)

    # --- Solve the optimization problem ---
    problem = cp.Problem(objective, constraints)
    # Using OSQP solver, which is good for quadratic programs.
    problem.solve(
        solver=cp.OSQP, verbose=verbose, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5
    )

    # --- Return the results ---
    if H.value is not None:
        H_opt = H.value
        # The constants (-2) are added to normalize the values.
        xi_val = (6 / n**2) * np.sum(H_opt**2) - 2
        psi_val = (6 / n**2) * np.trace(M @ H_opt) - 2
        return xi_val, psi_val, H_opt
    else:
        # Return None if the solver fails to find a solution.
        return None, None, None


# --- Main simulation loop ---
if __name__ == "__main__":
    # A range of mu values to trace the boundary.
    # A logarithmic scale is used to sample points more densely for small mu.
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the upper boundary for (xi, psi)...")
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
        label="Numerical Upper Bound",
    )

    # For comparison, plot the known theoretical upper bound psi = sqrt(xi)
    x_theory = np.linspace(0.001, 1, 100)
    plt.plot(
        x_theory,
        np.sqrt(x_theory),
        "r--",
        label="Theoretical Upper Bound (ψ = sqrt(ξ))",
    )

    plt.title("Attainable Region for (ξ, ψ)")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Spearman's Footrule ψ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.1, 1.05)
    plt.ylim(-0.55, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Compute and visualize H matrices for specific mu values ---
    print("\nComputing and saving H matrices to CSV files...")
    mu_for_files = [0.05, 0.5, 1.0, 10.0]
    n_vis = 64  # Use a higher resolution for visualization

    for mu_val in mu_for_files:
        # Compute the optimal H matrix
        _, _, H_map = get_boundary_point(mu=mu_val, n=n_vis)

        if H_map is not None:
            # --- Save to CSV ---
            filename = f"h_matrix_upper_mu_{mu_val:.2f}.csv"
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (Upper Bound)")
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
