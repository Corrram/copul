import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def rho_expr_from_identity(H):
    """
    Spearman's rho via  ∬C = ∬ (1-t) h_v(t) dt dv,
    using midpoint in t and trapezoid in v to remove the independence bias.
    """
    n = H.shape[0]
    t_mid = (np.arange(n) + 0.5) / n  # midpoints in t
    w_t = 1.0 - t_mid  # row weights (1 - t)

    w_v = np.ones(n)  # trapezoid weights in v
    w_v[0] = 0.5
    w_v[-1] = 0.5

    W = np.outer(w_t, w_v)  # (n,n) weight matrix
    integral_C = (1.0 / n**2) * cp.sum(cp.multiply(W, H))
    return 12.0 * integral_C - 3.0


def solve_SD_rho_max_at_xi(xi_target, n=32, tol=5e-3, verbose=False):
    """
    For a given target ξ (normalized so that independence is ξ=0),
    find the SD-feasible H that MAXIMIZES ρ subject to ξ <= ξ_target + tol.
    Returns (xi_val, rho_val, H_opt).
    """
    H = cp.Variable((n, n), name="H")

    # SD & CDF-in-v constraints
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),  # your discretization
        H[:, :-1] <= H[:, 1:],  # non-decreasing in v
        H[:-1, :] <= H[1:, :],  # SD: non-decreasing in u
    ]

    # ξ definition (unshifted); recall reported ξ = ξ_unshifted - 2
    xi_unshifted = (6.0 / n**2) * cp.sum_squares(H)
    # Map user's xi_target (in [-? , 1]) to unshifted target
    xi_unshifted_cap = xi_target + 2.0 + tol
    constraints += [xi_unshifted <= xi_unshifted_cap]

    # ρ (linear) — maximize ρ  <=>  minimize (-ρ)
    rho = rho_expr_from_identity(H)
    objective = cp.Minimize(-rho)

    prob = cp.Problem(objective, constraints)

    # Use a conic solver (quadratic constraint present)
    solved = False
    try:
        prob.solve(
            solver=cp.ECOS,
            max_iters=4000,
            abstol=1e-8,
            reltol=1e-8,
            feastol=1e-8,
            verbose=verbose,
        )
        solved = H.value is not None
    except Exception:
        solved = False
    if not solved:
        try:
            prob.solve(solver=cp.SCS, max_iters=60000, eps=5e-6, verbose=verbose)
            solved = H.value is not None
        except Exception:
            solved = False
    if not solved:
        return None, None, None

    H_opt = H.value
    xi_val = (6.0 / n**2) * np.sum(H_opt**2) - 2.0
    rho_val = (12.0 / n**2) * np.sum(
        (1.0 - (np.arange(n) + 0.5) / n)[:, None] * H_opt
    ) - 3.0
    return xi_val, rho_val, H_opt


if __name__ == "__main__":
    n = 32
    # sweep ξ from 0 (independence) up to ≈ 1 (strong functional dependence)
    xi_targets = np.linspace(0.0, 0.99, 5)
    tol = 5e-3

    frontier = []
    H_examples = {}

    print("Tracing the UPPER boundary of the SD region by maximizing ρ at fixed ξ...")
    for xi_tgt in tqdm(xi_targets):
        xi, rho, H = solve_SD_rho_max_at_xi(xi_tgt, n=n, tol=tol, verbose=False)
        if xi is not None:
            frontier.append((xi, rho))
            # cache a few examples to visualize shapes
            if any(np.isclose(xi_tgt, x, atol=0.02) for x in [0.0, 0.3, 0.6, 0.9]):
                H_examples[float(np.round(xi, 3))] = H
    print(frontier)
    frontier = np.array(frontier)

    # plot upper boundary ρ_max(ξ)
    plt.figure(figsize=(8, 8))
    if frontier.size:
        # sort by xi for a clean curve
        order = np.argsort(frontier[:, 0])
        plt.plot(
            frontier[order, 0],
            frontier[order, 1],
            "o-",
            label="Upper boundary (SD): max ρ at fixed ξ",
        )
    plt.scatter([0.0], [0.0], marker="x", s=80, label="Independence")
    plt.title("Upper Boundary of the (ξ, ρ) Region under SD")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Spearman's ρ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # visualize selected H maps
    for xi_val, H_map in H_examples.items():
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
        ax.set_title(f"h(t,v) at ξ ≈ {xi_val:.2f} (max ρ under SD)")
        ax.set_xlabel("v")
        ax.set_ylabel("t")
        plt.colorbar(
            im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, label="h(t,v)"
        )

    plt.show()
