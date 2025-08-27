# lti_xi_rho_upper_boundary_support.py
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ----- helpers -----
def build_C_from_H(H):
    """C = (1/n) * M @ H, cumulative in u (rows)."""
    n = H.shape[0]
    M = np.tril(np.ones((n, n)))
    return (1.0 / n) * (M @ H)  # affine in H


def rho_trapezoid_from_C(C):
    """
    Spearman's rho via ∬C with trapezoid weights in both u and v (linear in C).
    ρ = 12 * ∬C - 3,   ∬C ≈ (1/n^2) * Σ_ij W_ij C_ij
    """
    n = C.shape[0]
    w_u = np.ones(n)
    w_u[0] = 0.5
    w_u[-1] = 0.5
    w_v = np.ones(n)
    w_v[0] = 0.5
    w_v[-1] = 0.5
    W = np.outer(w_u, w_v)
    return 12.0 * (1.0 / n**2) * cp.sum(cp.multiply(W, C)) - 3.0


# ----- one support point (upper boundary) -----
def get_boundary_point_LTI_upper(mu, n=32, verbose=False):
    """
    Upper boundary support point under LTI:
        minimize  mu * xi(H) - rho(H)
    subject to LTI + copula discretization constraints.

    Returns (xi_val, rho_val, H_opt) with xi,rho using the usual normalization.
    """
    H = cp.Variable((n, n), name="H")

    # Base copula-derivative constraints
    constraints = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n),  # column sums so independence is feasible
        H[:, :-1] <= H[:, 1:],  # for each u, h(u,·) non-decreasing in v (cdf in v)
    ]

    # LTI: u ↦ C(u,v)/u is non-decreasing  ⇒  C_{i+1,j}/u_{i+1} >= C_{i,j}/u_i
    C = build_C_from_H(H)
    u = (np.arange(n) + 1.0) / n
    for i in range(n - 1):
        constraints.append(
            cp.multiply(u[i], C[i + 1, :]) <= cp.multiply(u[i + 1], C[i, :])
        )

    # Objective: minimize mu * xi - rho  (convex)
    xi_unshifted = (6.0 / n**2) * cp.sum_squares(H)  # reported xi = xi_unshifted - 2
    rho_expr = rho_trapezoid_from_C(C)

    mu_eff = max(mu, 1e-6)  # tiny regularization so mu=0 doesn’t become a pure LP
    objective = cp.Minimize(mu_eff * xi_unshifted + rho_expr)

    prob = cp.Problem(objective, constraints)

    # QP → OSQP first; ECOS/SCS fallback
    solved = False
    try:
        prob.solve(
            solver=cp.OSQP, max_iter=60000, eps_abs=1e-7, eps_rel=1e-7, verbose=verbose
        )
        solved = H.value is not None
    except Exception:
        solved = False
    if not solved:
        try:
            prob.solve(
                solver=cp.ECOS,
                max_iters=12000,
                abstol=1e-9,
                reltol=1e-9,
                feastol=1e-9,
                verbose=verbose,
            )
            solved = H.value is not None
        except Exception:
            solved = False
    if not solved:
        try:
            prob.solve(solver=cp.SCS, max_iters=100000, eps=2e-6, verbose=verbose)
            solved = H.value is not None
        except Exception:
            solved = False

    if not solved:
        return None, None, None

    H_opt = H.value
    # report xi, rho with the same quadrature used above
    xi_val = (6.0 / n**2) * np.sum(H_opt**2) - 2.0
    C_num = (1.0 / n) * (np.tril(np.ones((n, n))) @ H_opt)
    w_u = np.ones(n)
    w_u[0] = 0.5
    w_u[-1] = 0.5
    w_v = np.ones(n)
    w_v[0] = 0.5
    w_v[-1] = 0.5
    W = np.outer(w_u, w_v)
    rho_val = 12.0 * (1.0 / n**2) * np.sum(W * C_num) - 3.0

    return xi_val, rho_val, H_opt


# ----- sweep & plot -----
if __name__ == "__main__":
    n = 32
    # Sweep μ over a few decades to traverse the support curve.
    mu_values = np.r_[0.0, np.logspace(-3, 2, 36)]
    boundary = []
    H_examples = {}

    print("Tracing the UPPER LTI boundary via support sweep: min μ·ξ − ρ ...")
    for mu in tqdm(mu_values):
        xi, rho, H = get_boundary_point_LTI_upper(mu, n=n, verbose=False)
        if xi is not None:
            boundary.append((xi, rho))
            # keep a few shapes to inspect
            if any(np.isclose(mu, x, rtol=0, atol=1e-12) for x in [0.0]) or any(
                np.isclose(mu, x, rtol=0, atol=1e-3) for x in [1e-3, 1e-1, 1.0, 10.0]
            ):
                H_examples[float(np.round(mu, 6))] = H

    boundary = np.array(boundary)

    # Plot the support-traced upper boundary
    plt.figure(figsize=(8, 8))
    if boundary.size:
        order = np.argsort(boundary[:, 0])
        plt.plot(
            boundary[order, 0],
            boundary[order, 1],
            "o-",
            label="LTI upper boundary (support sweep: min μξ − ρ)",
        )
    plt.scatter([0.0], [0.0], marker="x", s=80, label="Independence")
    plt.title("LTI (ξ, ρ) — upper boundary via μ-sweep")
    plt.xlabel("Chatterjee's ξ")
    plt.ylabel("Spearman's ρ")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.gca().set_aspect("equal", "box")

    # Visualize a few h(t,v) maps
    for mu_key, H_map in H_examples.items():
        plt.figure(figsize=(7, 6))
        im = plt.imshow(
            H_map, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", vmin=0, vmax=1
        )
        plt.title(f"h(t,v) for μ = {mu_key} (LTI, upper support)")
        plt.xlabel("v")
        plt.ylabel("t")
        plt.colorbar(
            im, orientation="vertical", fraction=0.046, pad=0.04, label="h(t,v)"
        )

    plt.show()
