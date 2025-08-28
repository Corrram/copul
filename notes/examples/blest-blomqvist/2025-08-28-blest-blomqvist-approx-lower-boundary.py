import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_beta_lower(mu, n=32, verbose=False, solver="OSQP"):
    """
    Discretized optimization to trace the LOWER boundary of the (nu, beta) region.

    Grid/variable:
      - Midpoints: t_i = (i+0.5)/n, v_j = (j+0.5)/n, i,j=0,...,n-1
      - H[i,j] ~ h(t_i, v_j) = ∂C(t_i, v)/∂v at v=v_j

    Constraints:
      - 0 <= H <= 1
      - sum_i H[i, j] == j            (marginals)
      - H[:, j] <= H[:, j+1]          (nondecreasing in v)

    Measures:
      - Blest's ν:
            nu(H) ≈ (12 / n^2) * Σ_{i,j} ((1 - t_i)^2 * H[i,j]) - 2
      - Blomqvist's β:
            β = 4*C(1/2,1/2) - 1,
        with C(1/2,1/2) ≈ (1/n) * Σ_{j : v_j <= 1/2} H[i0, j],
        where i0 is the row with t_i closest to 1/2.

    Objective for LOWER boundary:
      - minimize   mu * nu(H) + beta(H)
        (constants in ν and β don't affect argmin)
    """
    H = cp.Variable((n, n), name="H")

    # --- Constraints ---
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
    ]

    # --- Geometry / weights ---
    t_mid = (np.arange(n) + 0.5) / n
    w = (1.0 - t_mid) ** 2  # weights for ν
    i0 = int(np.argmin(np.abs(t_mid - 0.5)))  # row nearest t=1/2
    j_half = np.where(t_mid <= 0.5)[0]  # columns with v<=1/2 (reuse t_mid array)

    # --- Measures (objective pieces) ---
    nu_term = (12 / n**2) * cp.sum(cp.multiply(w[:, None], H))
    C_half_half = (1 / n) * cp.sum(H[i0, j_half])
    beta_term = 4 * C_half_half - 1

    # LOWER envelope: minimize mu * ν + β
    objective = cp.Minimize(mu * nu_term + beta_term)

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
        problem.solve(solver="SCS", verbose=verbose, max_iters=50000)

    if H.value is None:
        return None, None, None

    # --- Evaluate ν and β ---
    H_opt = H.value
    nu_val = (12 / n**2) * np.sum(w[:, None] * H_opt) - 2
    C_val = (1 / n) * np.sum(H_opt[i0, j_half])
    beta_val = 4 * C_val - 1

    return nu_val, beta_val, H_opt


# --- Main sweep to trace the LOWER boundary ---
if __name__ == "__main__":
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the lower boundary for (nu, beta)...")
    for mu in tqdm(mu_values):
        nu, beta, _ = get_boundary_point_beta_lower(mu, n=32)
        if nu is not None:
            boundary_points.append((nu, beta))

    # --- Plot: Numerical lower boundary ---
    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        boundary_points = boundary_points[np.argsort(boundary_points[:, 0])]
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Lower Boundary (ν–β)",
        )

    plt.title("Attainable Region (Lower Boundary) for (ν, β)")
    plt.xlabel("Blest's ν")
    plt.ylabel("Blomqvist's β")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)  # ν scale as per definition used here
    plt.ylim(-1.05, 1.05)  # β ∈ [-1,1]
    plt.gca().set_aspect("equal", "box")

    # --- Visualize H(t,v) for representative μ values ---
    mu_for_files = [0.0001, 0.001, 0.01, 0.05, 0.5, 1.0, 10.0]
    n_vis = 64

    for mu_val in mu_for_files:
        _, _, H_map = get_boundary_point_beta_lower(mu=mu_val, n=n_vis)
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (β Lower Bound)")
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
