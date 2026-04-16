"""
boundary_solver.py – shared CVXPY infrastructure for tracing rank-correlation
attainable-region boundaries via scalarization (μ-sweep).

Import from any script in this tree with:

    # scripts in a sub-folder (blest-regions/, xi-rho/, xi-footrule/, …):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from boundary_solver import solve_h, xi_expr, rho_expr, xi_val, rho_val, ...

    # scripts directly in notes/examples/:
    from boundary_solver import ...
"""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Discretization helpers ────────────────────────────────────────────────────


def t_midpoints(n: int) -> np.ndarray:
    """Midpoint grid: t_i = (i + 0.5) / n  for i = 0, …, n−1."""
    return (np.arange(n) + 0.5) / n


def M_lower(n: int) -> np.ndarray:
    """Lower-triangular ones matrix (discrete integration operator)."""
    return np.tril(np.ones((n, n)))


def J_anti(n: int) -> np.ndarray:
    """Anti-diagonal permutation matrix (reversal in v)."""
    return np.fliplr(np.eye(n))


def nu_weights(n: int) -> np.ndarray:
    """Row weights w_i = (1 − t_i)² for Blest's ν."""
    return (1.0 - t_midpoints(n)) ** 2


def beta_half_params(n: int, u_star: float = 0.5, v_star: float = 0.5):
    """
    Compute k_u and col_w for discretizing C(u*, v*) ≈ Blomqvist's β.

    Returns
    -------
    k_u   : number of rows t_i ≤ u*
    col_w : length-n weight vector (linear interpolation at v = v*)
    """
    t = t_midpoints(n)
    k_u = int(np.sum(t <= u_star))
    j_r = int(np.searchsorted(t, v_star, side="right"))
    j_l = j_r - 1
    if j_l < 0:
        j_l = j_r = 0
        lam = 0.0
    elif j_r >= n:
        j_l = j_r = n - 1
        lam = 0.0
    else:
        lam = (v_star - t[j_l]) / (t[j_r] - t[j_l])
    col_w = np.zeros(n)
    col_w[j_l] += 1.0 - lam
    col_w[j_r] += lam
    return k_u, col_w


# ── CVXPY expression builders (constants omitted; add them back in *_val) ─────


def xi_expr(H, n: int):
    """ξ term (without −2): (6/n²)·‖H‖²."""
    return (6 / n**2) * cp.sum_squares(H)


def rho_expr(H, n: int, M: np.ndarray):
    """ρ term (without −3): (12/n³)·∑(M @ H)."""
    return (12 / n**3) * cp.sum(M @ H)


def psi_expr(H, n: int, M: np.ndarray):
    """Footrule ψ term (without −2): (6/n²)·tr(M @ H)."""
    return (6 / n**2) * cp.trace(M @ H)


def nu_expr(H, n: int):
    """Blest ν term (without −2): (12/n²)·∑((1−t_i)²·H)."""
    w = nu_weights(n)
    return (12 / n**2) * cp.sum(cp.multiply(w[:, None], H))


def gamma_expr(H, n: int, M: np.ndarray, J: np.ndarray):
    """Gini γ term (without −2): (4/n²)·[tr(MH) + tr(MHJ)]."""
    return (4 / n**2) * (cp.trace(M @ H) + cp.trace(M @ H @ J))


def beta_expr(H, k_u: int, col_w: np.ndarray, n: int):
    """Blomqvist β = 4·C(½,½) − 1 (linear in H via interpolated column weights)."""
    C_half = (1 / n) * cp.sum(cp.multiply(H[:k_u, :], col_w[None, :]))
    return 4 * C_half - 1


# ── NumPy evaluations (include the conventional constant offsets) ─────────────


def xi_val(H: np.ndarray, n: int) -> float:
    return float((6 / n**2) * np.sum(H**2) - 2)


def rho_val(H: np.ndarray, n: int, M: np.ndarray) -> float:
    return float((12 / n**3) * np.sum(M @ H) - 3)


def psi_val(H: np.ndarray, n: int, M: np.ndarray) -> float:
    return float((6 / n**2) * np.trace(M @ H) - 2)


def nu_val(H: np.ndarray, n: int) -> float:
    w = nu_weights(n)
    return float((12 / n**2) * np.sum(w[:, None] * H) - 2)


def gamma_val(H: np.ndarray, n: int, M: np.ndarray, J: np.ndarray) -> float:
    return float((4 / n**2) * (np.trace(M @ H) + np.trace(M @ H @ J)) - 2)


def beta_val(H: np.ndarray, k_u: int, col_w: np.ndarray, n: int) -> float:
    C_half = (1 / n) * np.sum(H[:k_u, :] * col_w[None, :])
    return float(4 * C_half - 1)


# ── Core solver ───────────────────────────────────────────────────────────────


def solve_h(
    objective_fn,
    n: int = 32,
    extra_constraints=(),
    marginal_offset: float = 0.0,
    verbose: bool = False,
) -> np.ndarray | None:
    """
    Build H (n×n copula-derivative variable) with the four standard constraints,
    apply *objective_fn(H)* → ``cp.Minimize(…)``, solve, and return ``H.value``
    or ``None`` on failure.

    Parameters
    ----------
    objective_fn      : callable  H → cp.Minimize(expression)
    extra_constraints : iterable of additional CVXPY Constraint objects, OR
                        a callable  H → list[Constraint]  (evaluated lazily).
    marginal_offset   : added to ``np.arange(n)`` in the column-sum constraint.
                        Use 0.5 for the midpoint-corrected marginal variant.
    verbose           : passed to the solver.
    """
    H = cp.Variable((n, n), name="H")
    base = [
        H >= 0,
        H <= 1,
        cp.sum(H, axis=0) == np.arange(n) + marginal_offset,
        H[:, :-1] <= H[:, 1:],
    ]
    extra = extra_constraints(H) if callable(extra_constraints) else list(extra_constraints)
    problem = cp.Problem(objective_fn(H), base + extra)
    try:
        problem.solve(solver=cp.OSQP, verbose=verbose,
                      max_iter=20000, eps_abs=1e-5, eps_rel=1e-5, warm_start=True)
    except Exception:
        problem.solve(solver="SCS", verbose=verbose, max_iters=50000)
    return H.value


# ── μ-sweep helper ────────────────────────────────────────────────────────────


def sweep_boundary(
    get_point_fn,
    mu_values=None,
    n: int = 32,
    label: str = "boundary",
) -> np.ndarray:
    """
    Call ``get_point_fn(mu, n=n)`` → ``(x, y, H)`` for each μ, collecting
    valid ``(x, y)`` pairs.  Returns an (m, 2) array (empty if nothing succeeded).
    """
    if mu_values is None:
        mu_values = np.logspace(-2, 1.5, 30)
    pts = []
    print(f"Tracing {label}…")
    for mu in tqdm(mu_values):
        x, y, _ = get_point_fn(mu, n=n)
        if x is not None:
            pts.append((x, y))
    return np.array(pts) if pts else np.empty((0, 2))


# ── Plotting helpers ──────────────────────────────────────────────────────────


def plot_boundary(
    pts: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    xlim: tuple,
    ylim: tuple,
    extra_plots=(),
    sort_col: int = 0,
    figsize: tuple = (8, 8),
) -> None:
    """
    Plot a numerical boundary curve from a (m, 2) point array.

    Parameters
    ----------
    extra_plots : iterable of callables ``f(ax)`` that add reference lines /
                  annotations to the axes.
    sort_col    : column index (0 or 1) used to sort pts before plotting.
    """
    _, ax = plt.subplots(figsize=figsize)
    if pts.size:
        p = pts[np.argsort(pts[:, sort_col])]
        ax.plot(p[:, 0], p[:, 1], "o-", label="Numerical boundary")
    for fn in extra_plots:
        fn(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":")
    ax.legend()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", "box")


def plot_h_matrix(H_map: np.ndarray, title: str, ax=None) -> None:
    """Visualise a single n×n copula-derivative matrix h(t, v) via imshow."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(H_map, origin="lower", extent=[0, 1, 0, 1],
                   cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("v")
    ax.set_ylabel("t")
    plt.colorbar(im, ax=ax, orientation="vertical",
                 fraction=0.046, pad=0.04, label="h(t,v)")


def plot_h_matrices(
    get_point_fn,
    mu_vals,
    n: int = 64,
    title_fmt: str = "h(t,v) for μ = {mu:.2f}",
) -> None:
    """
    Run ``get_point_fn(mu=mu, n=n)`` → ``(x, y, H)`` for each μ in *mu_vals*
    and call :func:`plot_h_matrix` on the result.
    """
    for mu in mu_vals:
        _, _, H_map = get_point_fn(mu=mu, n=n)
        if H_map is not None:
            plot_h_matrix(H_map, title_fmt.format(mu=mu))
        else:
            print(f"Solver failed for μ={mu}.")
