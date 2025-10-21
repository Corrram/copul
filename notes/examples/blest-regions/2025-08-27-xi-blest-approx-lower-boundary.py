import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt


def get_boundary_point_nu_lower(mu, n=32, verbose=False, solver="OSQP"):
    """
    Discrete convex program for the LOWER boundary of (xi, nu):
        minimize   xi(H) + mu * nu(H)
        subject to 0 <= H <= 1,
                   column sums enforce marginals,
                   H monotone nondecreasing in v.
    Returns (xi, nu, H) for the optimal H at this mu.
    """
    # H[i,j] ~ h(t_i, v_j) on an n x n midpoint grid
    H = cp.Variable((n, n), name="H")

    # --- Constraints ---
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),  # ∑_i H[i,j] == j
        H[:, :-1] <= H[:, 1:],  # monotone in v
    ]

    # --- Quadratic xi term ---
    xi_term = (6 / n**2) * cp.sum_squares(H)  # (constant -2 omitted in objective)

    # --- Linear nu term: weight w_i = (1 - t_i)^2 applied row-wise ---
    t_mid = (np.arange(n) + 0.5) / n
    w = (1.0 - t_mid) ** 2  # shape (n,)
    nu_term = (12 / n**2) * cp.sum(cp.multiply(w[:, None], H))

    # --- Scalarized lower-bound objective: minimize xi + mu * nu ---
    objective = cp.Minimize(xi_term + mu * nu_term)

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

    H_opt = H.value
    # Report the properly normalized statistics (with constants)
    xi_val = (6 / n**2) * np.sum(H_opt**2) - 2
    nu_val = (12 / n**2) * np.sum(w[:, None] * H_opt) - 2
    return xi_val, nu_val, H_opt


# --- Main sweep over mu to trace the lower boundary ---
if __name__ == "__main__":
    # You can push the range higher if you want more negative ν;
    # very large μ emphasizes minimizing ν over ξ and can produce sharp bands.
    mu_values = np.logspace(-2, 2.0, 36)
    boundary_points = []

    print("Tracing the lower boundary for (xi, nu)...")
    for mu in mu_values:
        xi, nu, _ = get_boundary_point_nu_lower(mu, n=32)
        if xi is not None:
            boundary_points.append((xi, nu))

    # --- Plot: numerical lower boundary ---
    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary = np.array(boundary_points)
        # Keep only the lower envelope in case of minor ordering wiggles
        # (optional: sort by xi)
        idx = np.argsort(boundary[:, 0])
        boundary = boundary[idx]
        plt.plot(
            boundary[:, 0],
            boundary[:, 1],
            "o-",
            label="Numerical Lower Boundary (ξ–ν)",
        )

    # For reference, also plot the theoretical upper boundary ν = sqrt(ξ)
    x_ref = np.linspace(0.0, 1.0, 200)
    plt.plot(x_ref, np.sqrt(x_ref), "k--", label=r"Upper: $\nu=\sqrt{\xi}$")

    plt.title("Attainable Region (Lower Boundary) for (ξ, ν)")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Blest's ν")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Visualize h(t,v) for a few μ values to see the structure ---
    mu_for_maps = [0.05, 0.5, 2.0, 10.0, 50.0]
    n_vis = 64

    for mu_val in mu_for_maps:
        _, _, H_map = get_boundary_point_nu_lower(mu=mu_val, n=n_vis)
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
            ax.set_title(f"Structure of h(t,v) for μ = {mu_val:.2f} (ν Lower Boundary)")
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
