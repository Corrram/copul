import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_nu_vs_rho_upper(mu, n=32, verbose=False, solver="OSQP"):
    """
    Solves the discretized optimization problem for a given mu.
    Finds the copula derivative H ~ h(t_i, v_j) that minimizes  mu * rho - nu.
    This traces the UPPER boundary of the (rho, nu) region.

    Grid / constraints (midpoint discretization):
      - t_i, v_j in { (k+0.5)/n : k=0,...,n-1 }
      - H[i,j] ~ h(t_i, v_j)
      - 0 <= H <= 1
      - sum_i H[i, j] == j                 (∫ h(·, v_j) dt = v_j, up to scaling)
      - H[:, j] <= H[:, j+1]               (nondecreasing in v)

    Functionals (same normalizations you used elsewhere):
      - rho ≈ (12 / n^3) * sum( (M @ H) ) - 3,
        where M is lower-triangular of ones (discrete integral in t)
      - nu  ≈ (12 / n^2) * sum( ((1 - t_i)^2) * H[i,j] ) - 2

    Objective:
      - minimize  mu * rho_term - nu_term
        (constant offsets -3 and -2 do not affect the argmin)
    """
    # Variable
    H = cp.Variable((n, n), name="H")

    # Constraints: box, marginal, SI in v
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
    ]

    # Terms
    # rho: discrete integral of C over the unit square; C(t_i, v_j) ~ (M @ H)[i, j]
    M = np.tril(np.ones((n, n)))
    rho_term = (12 / n**3) * cp.sum(M @ H)

    # nu: linear in H with weights w_i = (1 - t_i)^2 on rows
    t_mid = (np.arange(n) + 0.5) / n
    w = (1.0 - t_mid) ** 2  # shape (n,)
    nu_term = (12 / n**2) * cp.sum(cp.multiply(w[:, None], H))

    # Objective: minimize mu*rho - nu
    objective = cp.Minimize(mu * rho_term - nu_term)

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        if solver == "OSQP":
            problem.solve(
                solver=cp.OSQP,
                verbose=verbose,
                max_iter=20000,
                eps_abs=1e-5,
                eps_rel=1e-5,
                warm_start=True,
            )
        else:
            problem.solve(solver=solver, verbose=verbose)
    except Exception:
        # Fallback to SCS if the chosen solver fails
        problem.solve(solver="SCS", verbose=verbose, max_iters=50000)

    if H.value is None:
        return None, None, None

    H_opt = H.value
    rho_val = (12 / n**3) * np.sum(M @ H_opt) - 3
    nu_val = (12 / n**2) * np.sum(w[:, None] * H_opt) - 2
    return rho_val, nu_val, H_opt


# --- Main: sweep mu and plot boundary ---
if __name__ == "__main__":
    # Sweep a range of slopes for supporting lines; feel free to densify
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the upper boundary for (rho, nu)...")
    for mu in tqdm(mu_values):
        rho, nu, _ = get_boundary_point_nu_vs_rho_upper(mu, n=32)
        if rho is not None:
            boundary_points.append((rho, nu))

    # --- Plot attainable upper boundary (numerical) ---
    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        # sort by rho for a nicer curve
        boundary_points = boundary_points[np.argsort(boundary_points[:, 0])]
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Upper Boundary (ρ–ν)",
        )

    # Reference markers
    plt.plot([0], [0], "ks", label="Independence (ρ=0, ν=0)")
    plt.plot([1], [1], "k^", label="Comonotone (ρ=1, ν=1)")

    plt.title("Attainable Region (Upper Boundary) for (ρ, ν)")
    plt.xlabel("Spearman's ρ")
    plt.ylabel("Blest's ν")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Visualize H(t,v) for a few μ values ---
    mu_for_files = [0.05, 0.1, 0.2, 0.5, 1.0, 1.2, 1.5, 2.0]
    n_vis = 64

    for mu_val in mu_for_files:
        _, _, H_map = get_boundary_point_nu_vs_rho_upper(mu=mu_val, n=n_vis)
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
            ax.set_title(
                f"Structure of h(t,v) for μ = {mu_val:.2f} (ρ–ν Upper Boundary)"
            )
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
