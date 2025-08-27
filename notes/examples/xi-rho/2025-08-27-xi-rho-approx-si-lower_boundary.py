import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def rho_expr_from_identity(H):
    """
    Discrete Spearman's rho via ∫∫ C = ∫∫ (1-t) h_v(t) dt dv.
    Use midpoint rule in t (u-index) and rectangle in v.
    """
    n = H.shape[0]
    # midpoint nodes t_u ≈ (u+0.5)/n, weights (1 - t_u)
    t_mid = (np.arange(n) + 0.5) / n
    w_u = 1.0 - t_mid  # shape (n,)
    # integral ≈ (1/n^2) * sum_{u,v} w_u * H[u,v]
    integral_C = (1.0 / n**2) * cp.sum(cp.multiply(w_u.reshape(-1, 1), H))
    # Spearman rho = 12 * integral - 3
    return 12.0 * integral_C - 3.0


def solve_SD_fixed_rho(rho_target, n=32, tol=5e-3, verbose=False):
    """
    Lower boundary at fixed rho: minimize ξ subject to SD constraints and ρ ∈ [ρ_tgt±tol].
    """
    H = cp.Variable((n, n), name="H")

    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),  # same discretization as your original
        H[:, :-1] <= H[:, 1:],  # monotone in v (cdf in v)
        H[:-1, :] <= H[1:, :],  # SD: non-decreasing in u
    ]

    # rho via (1 - t) identity (linear in H)
    rho = rho_expr_from_identity(H)
    constraints += [rho >= rho_target - tol, rho <= rho_target + tol]

    # minimize ξ (constant shift -2 irrelevant for optimizer)
    xi = (6.0 / n**2) * cp.sum_squares(H)

    prob = cp.Problem(cp.Minimize(xi), constraints)

    # robust solver chain
    solvers = [
        dict(
            solver=cp.OSQP, max_iter=40000, eps_abs=1e-7, eps_rel=1e-7, verbose=verbose
        ),
        dict(
            solver=cp.ECOS,
            max_iters=2000,
            abstol=1e-8,
            reltol=1e-8,
            feastol=1e-8,
            verbose=verbose,
        ),
        dict(solver=cp.SCS, max_iters=50000, eps=5e-6, verbose=verbose),
    ]
    last_err = None
    for opts in solvers:
        try:
            prob.solve(**opts)
            if H.value is not None:
                break
        except Exception as e:
            last_err = e

    if H.value is None:
        if last_err is not None and verbose:
            print("Last solver error:", repr(last_err))
        return None, None, None

    H_opt = H.value
    xi_val = (6.0 / n**2) * np.sum(H_opt**2) - 2.0
    rho_val = (12.0 / n**2) * np.sum(
        (1.0 - (np.arange(n) + 0.5) / n)[:, None] * H_opt
    ) - 3.0
    return xi_val, rho_val, H_opt


if __name__ == "__main__":
    n = 32
    # SD typically supports negative dependence too, but the feasible ρ-range is grid- and constraint-dependent.
    # Start modestly; widen if needed.
    rho_targets = np.linspace(-0.98, 0.95, 40)
    tol = 1e-2

    boundary = []
    H_examples = {}

    print("Tracing the LOWER boundary of the SD region by minimizing ξ at fixed ρ...")
    for rho_tgt in tqdm(rho_targets):
        xi, rho, H = solve_SD_fixed_rho(rho_tgt, n=n, tol=tol, verbose=False)
        if xi is not None:
            boundary.append((xi, rho))
            # stash a few illustrative heatmaps
            if any(
                np.isclose(rho_tgt, x, atol=0.02) for x in [-0.9, -0.5, 0.0, 0.5, 0.9]
            ):
                H_examples[float(np.round(rho, 3))] = H

    # Plot ξ vs ρ frontier
    boundary = np.array(boundary)
    plt.figure(figsize=(8, 8))
    if boundary.size:
        plt.plot(
            boundary[:, 0],
            boundary[:, 1],
            "o-",
            label="Lower boundary (SD): min ξ at fixed ρ",
        )
    # optional: independence line reference at ρ=0, ξ=0
    plt.scatter([0.0], [0.0], marker="x", s=80, label="Independence")
    plt.title("Attainable Region (ξ, ρ) under Stochastic Decrease (SD)")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Spearman's ρ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # Visualize selected H maps
    for rho_val, H_map in H_examples.items():
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
        ax.set_title(f"h(t,v) at ρ ≈ {rho_val:.2f} (min ξ under SD)")
        ax.set_xlabel("v")
        ax.set_ylabel("t")
        plt.colorbar(
            im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, label="h(t,v)"
        )

    plt.show()
