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
        H >= 0.0,
        H <= 1.0,
        cp.sum(H, axis=0) == marginal_rhs,
        H[:, :-1] <= H[:, 1:],
    ]

    xi_term = (6 / n**2) * cp.sum_squares(H)
    psi_term = (6 / n**2) * cp.trace(M @ H)
    objective = cp.Minimize(psi_term + mu * xi_term)

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
    mu_for_files = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 2, 5, 10.0, 20]
    n_vis = 64

    for mu_val in mu_for_files:
        # 1. Compute the optimal H matrix (H(t,v) where t=rows, v=columns)
        H_map = get_boundary_point(mu=mu_val, n=n_vis)

        if H_map is not None:
            # 2. Calculate the copula density c(t,v) from H
            h_padded = np.hstack([np.zeros((n_vis, 1)), H_map])
            C_map = n_vis * np.diff(h_padded, axis=1)

            # 3. Visualize the resulting density with axes swapped
            plt.figure(figsize=(7, 6))
            ax = plt.gca()

            # KEY CHANGE: Plot the transpose of C_map to interchange the axes.
            # This puts the original 'v' axis on the y-axis and 't' on the x-axis.
            im = ax.imshow(
                C_map.T,
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="inferno",
                aspect="auto",
            )

            # KEY CHANGE: Update title and swap labels to match the new orientation.
            ax.set_title(f"Copula Density c(v,t) for μ = {mu_val:.2f}")
            ax.set_xlabel("t")
            ax.set_ylabel("v")
            plt.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.046,
                pad=0.04,
                label="Density c(v,t)",
            )
        else:
            pass
        plt.savefig(
            f"images/xi_footrule_approx_lower_boundary_density_mu_{mu_val:.2f}.png"
        )
    plt.show()
