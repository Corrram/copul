import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point(mu, n=32, verbose=False):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative H that minimizes psi + mu * xi.
    """
    H = cp.Variable((n, n), name="H")

    # Explicitly define constants with float64 dtype
    marginal_rhs = np.arange(n, dtype=np.float64) + 0.5
    M = np.tril(np.ones((n, n), dtype=np.float64))

    constraints = [
        # --- FIX: Use floating point constants for inequalities ---
        # This ensures all constants passed to the solver are floats.
        H >= 0.0,
        H <= 1.0,
        # This constraint correctly enforces the marginal property
        cp.sum(H, axis=0) == marginal_rhs,
        # This constraint enforces monotonicity in v for each t
        H[:, :-1] <= H[:, 1:],
    ]

    xi_term = (6 / n**2) * cp.sum_squares(H)
    psi_term = (6 / n**2) * cp.trace(M @ H)
    objective = cp.Minimize(psi_term + mu * xi_term)

    # We will try the SCS solver as it can be more robust for this problem type
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)

    if H.value is not None and problem.status == "optimal":
        return H.value
    else:
        print(
            f"Solver failed or did not find an optimal solution for μ={mu:.2f}. Status: {problem.status}"
        )
        return None


# --- Main simulation and plotting loop ---
if __name__ == "__main__":
    # A selection of mu values to visualize
    mu_for_files = [0.05, 0.2, 0.5, 1.0, 5.0, 20.0]
    n_vis = 64  # Use a higher resolution for visualization

    print("\nComputing H matrices and plotting corresponding densities c(t,v)...")

    for mu_val in mu_for_files:
        # 1. Compute the optimal H matrix (conditional distribution)
        H_map = get_boundary_point(mu=mu_val, n=n_vis)

        if H_map is not None:
            # 2. Calculate the copula density c(t,v) from H
            # We approximate the partial derivative c = ∂h/∂v using finite differences.
            # The step size dv is 1/n_vis.
            # c(i,j) ≈ (H(i,j) - H(i, j-1)) / (1/n_vis) = n_vis * (H(i,j) - H(i, j-1))

            # Prepend a column of zeros to H to correctly calculate the derivative at the first column
            h_padded = np.hstack([np.zeros((n_vis, 1)), H_map])

            # Calculate the difference between adjacent columns
            C_map = n_vis * np.diff(h_padded, axis=1)

            # 3. Visualize the resulting density
            plt.figure(figsize=(7, 6))
            ax = plt.gca()

            # Use a different colormap that's good for densities (like 'inferno' or 'magma')
            # We don't set vmin/vmax, allowing matplotlib to auto-scale the color
            # to the range of density values, which can be > 1.
            im = ax.imshow(
                C_map,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="inferno",
                aspect="auto",
            )
            ax.set_title(f"Copula Density c(t,v) for μ = {mu_val:.2f}")
            ax.set_xlabel("v")
            ax.set_ylabel("t")
            plt.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.046,
                pad=0.04,
                label="Density c(t,v)",
            )
        else:
            # This handles the case where the solver might fail for a given mu
            pass

    # Display all created figures at the end
    plt.show()
