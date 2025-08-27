import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_boundary_point_psi_nu_lower(mu, n=32, verbose=False, solver="OSQP"):
    """
    Lower boundary of (psi, nu): minimize  mu * psi(H) + nu(H)  over feasible H.

    Grid/constraints:
      - midpoint grid t_i, v_j = (k+0.5)/n
      - H[i,j] ~ h(t_i, v_j)
      - 0 <= H <= 1
      - sum_i H[i, j] == j                 (marginal in v)
      - H[:, j] <= H[:, j+1]               (nondecreasing in v)

    Discrete functionals (constants dropped in the objective):
      - psi ≈ (6 / n^2) * tr(M @ H) - 2,        M = tril(1)
      - nu  ≈ (12 / n^2) * sum(((1 - t_i)^2) * H[i,j]) - 2
    """
    H = cp.Variable((n, n), name="H")

    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),
        H[:, :-1] <= H[:, 1:],
    ]

    # Footrule ψ term (no constant in the objective)
    M = np.tril(np.ones((n, n)))
    psi_term = (6 / n**2) * cp.trace(M @ H)

    # Blest ν term (no constant in the objective)
    t_mid = (np.arange(n) + 0.5) / n
    w = (1.0 - t_mid) ** 2
    nu_term = (12 / n**2) * cp.sum(cp.multiply(w[:, None], H))

    # Lower envelope: minimize  mu * psi  +  nu
    objective = cp.Minimize(mu * psi_term + nu_term)

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
        problem.solve(solver="SCS", verbose=verbose, max_iters=50000)

    if H.value is None:
        return None, None, None

    H_opt = H.value
    psi_val = (6 / n**2) * np.trace(M @ H_opt) - 2
    nu_val = (12 / n**2) * np.sum(w[:, None] * H_opt) - 2
    return psi_val, nu_val, H_opt


# --- Main sweep & plotting ---
if __name__ == "__main__":
    mu_values = np.logspace(-2, 1.5, 30)
    boundary_points = []

    print("Tracing the lower boundary for (ψ, ν)...")
    for mu in tqdm(mu_values):
        psi, nu, _ = get_boundary_point_psi_nu_lower(mu, n=32)
        if psi is not None:
            boundary_points.append((psi, nu))

    plt.figure(figsize=(8, 8))
    if boundary_points:
        boundary_points = np.array(boundary_points)
        # sort by ψ for a clean curve
        boundary_points = boundary_points[np.argsort(boundary_points[:, 0])]
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "o-",
            label="Numerical Lower Boundary (ψ–ν)",
        )

    # reference anchor
    plt.plot([0], [0], "ks", label="Independence (ψ=0, ν=0)")

    plt.title("Attainable Region (Lower Boundary) for (ψ, ν)")
    plt.xlabel("Spearman's Footrule ψ")
    plt.ylabel("Blest's ν")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.55, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # --- Visualize H(t,v) for a few μ values ---
    mu_for_files = [0.05, 0.2, 0.5, 1.0, 1.5, 2, 10.0]
    n_vis = 64

    for mu_val in mu_for_files:
        _, _, H_map = get_boundary_point_psi_nu_lower(mu=mu_val, n=n_vis)
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
            ax.set_title(f"h(t,v) for μ = {mu_val:.2f} (ψ–ν Lower Boundary)")
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
