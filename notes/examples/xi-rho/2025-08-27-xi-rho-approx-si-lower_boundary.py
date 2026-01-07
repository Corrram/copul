import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def solve_lower_bound_ccp(target_xi, n=32, max_iter=10, verbose=False):
    """
    Minimizes Spearman's rho for a FIXED lower bound on Chatterjee's xi
    subject to Stochastically Increasing (SI) constraints.

    Uses the Convex-Concave Procedure (CCP) because the constraint
    xi(H) >= target_xi is non-convex.
    """
    # --- 1. Initialization ---
    # We need a feasible starting point H0.
    # The comonotonic copula (Upper Fréchet) has xi=1, rho=1.
    # It satisfies all constraints and xi >= target for any target <= 1.

    # Construct H_comonotonic: H[t, v] = 1 if v >= t else 0
    # Note: Based on the previous script's indexing sum(H,0) = 0..n-1
    # We construct an Upper Triangular matrix.
    H_val = np.triu(np.ones((n, n)))
    # Adjust to match the specific column sum constraint sum(col_j) = j
    # The standard Upper Tri has sums 1, 2, ..., n.
    # The previous script used sum = 0, 1, ..., n-1.
    # We shift the logic slightly: The first column should be all zeros.
    H_val = np.zeros((n, n))
    for j in range(n):
        # We need sum equal to j. Fill first j rows with 1.
        H_val[:j, j] = 1.0

    # --- 2. CCP Loop ---
    rho_final = None
    xi_final = None
    H_opt_final = None

    for k in range(max_iter):
        H = cp.Variable((n, n), name="H")

        # --- Constraints ---
        constraints = [
            H >= 0,
            H <= 1,
            cp.sum(H, axis=0) == np.arange(n), # Marginal constraint
            H[:, :-1] <= H[:, 1:],             # Increasing in v (std copula)
            H[:-1, :] >= H[1:, :]              # Decreasing in t (SI condition)
        ]

        # --- Linearized Xi Constraint ---
        # We want: xi(H) >= target_xi
        # xi is convex (sum of squares). We linearize it around the previous point H_val.
        # Convex inequality: f(x) >= f(x0) + g^T(x-x0) >= target

        # Calculate xi and gradient for current H_val
        coeff_xi = (6 / n**2)
        xi_current = coeff_xi * np.sum(H_val**2) - 2
        grad_xi = 2 * coeff_xi * H_val

        # Linearized constraint
        linearized_xi = xi_current + cp.sum(cp.multiply(grad_xi, H - H_val))
        constraints.append(linearized_xi >= target_xi)

        # --- Objective: Minimize Rho ---
        M = np.tril(np.ones((n, n)))
        rho_term = (12 / n**3) * cp.sum(M @ H)
        # Note: Previous script formula for rho was (term) - 3.
        # We minimize the term directly.
        objective = cp.Minimize(rho_term)

        # --- Solve ---
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
        except cp.SolverError:
            break

        if H.value is None:
            break

        # Update H_val for next iteration
        H_val = H.value

        # Calculate actual metrics
        xi_final = coeff_xi * np.sum(H_val**2) - 2
        rho_final = (12 / n**3) * np.sum(M @ H_val) - 3
        H_opt_final = H_val

    return xi_final, rho_final, H_opt_final

# --- Main simulation loop ---
if __name__ == "__main__":
    # We sweep target xi from 0 to 1
    target_xi_values = np.linspace(0, 1.0, 25)
    lower_bound_points = []

    print("Tracing the lower bound (Min Rho given Xi) for SI copulas...")

    H_maps = []

    for t_xi in tqdm(target_xi_values):
        xi, rho, H_out = solve_lower_bound_ccp(t_xi, n=32, max_iter=8)
        if xi is not None:
            lower_bound_points.append((xi, rho))
            # Save a few specific examples for visualization
            if t_xi in [0.0, 0.25, 0.5, 0.75, 1.0]:
                # Find closest actual xi
                H_maps.append((xi, rho, H_out))

    # --- Visualization ---
    plt.figure(figsize=(10, 8))

    # 1. Plot the Numerical Lower Bound
    if lower_bound_points:
        lb_arr = np.array(lower_bound_points)
        plt.plot(lb_arr[:, 0], lb_arr[:, 1], "o-", color="blue", label="Numerical SI Lower Bound (Min ρ)")

    # 2. Theoretical Reference Lines
    x_grid = np.linspace(0, 1, 100)
    plt.plot(x_grid, x_grid, "k:", alpha=0.5, label="Diagonal (ρ = ξ)")

    # Theoretical lower bound for GENERAL copulas (approximate) is much lower.
    # For SI copulas, it is known that rho >= (3*xi - 1)/2 is a loose bound,
    # but the tightness is often debated.
    plt.plot(x_grid, (3*x_grid - 1)/2, "r--", alpha=0.3, label="General Lower Bound (ρ = (3ξ-1)/2)")

    plt.title("Attainable Region for SI Copulas: Min Spearman's ρ for given Chatterjee's ξ")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Spearman's ρ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    # --- Plot H matrices for the lower boundary ---
    # We pick 3 representative points from our run
    if len(H_maps) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        indices_to_plot = [0, len(H_maps)//2, -1]

        for ax, idx in zip(axes, indices_to_plot):
            xi_val, rho_val, H_mat = H_maps[idx]
            im = ax.imshow(H_mat, origin="lower", extent=[0,1,0,1], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"ξ ≈ {xi_val:.2f}, ρ ≈ {rho_val:.2f}")
            ax.set_xlabel("v")
            ax.set_ylabel("t")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle("Copula Derivative Structures h(t,v) on the Lower Boundary")

    plt.show()