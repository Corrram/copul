import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt


# ============================================================
# Numerical integration / discretization settings
# ============================================================

N_T = 801  # grid size in t
N_V = 301  # grid size in v
N_B = 160  # number of b-values on the parameter grid
B_MIN = 1e-4
B_MAX = 1e3

t_grid = np.linspace(0.0, 1.0, N_T)
v_grid = np.linspace(0.0, 1.0, N_V)


# ============================================================
# Utilities
# ============================================================


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def trapz2(values, x_grid, y_grid):
    inner = np.trapezoid(values, x_grid, axis=1)
    return np.trapezoid(inner, y_grid)


def solve_section_offset(v, b, weight):
    """
    Solve for a in
        ∫_0^1 clip(a + b * weight(t), 0, 1) dt = v.
    """

    def f(a):
        h = clip01(a + b * weight)
        return np.trapezoid(h, t_grid) - v

    return brentq(f, -2.0 - b, 2.0)


def build_H_matrix(kind, b):
    """
    kind = "rho" : clamped linear family
    kind = "nu"  : clamped quadratic family
    """
    if kind == "rho":
        weight = 1.0 - t_grid
    elif kind == "nu":
        weight = (1.0 - t_grid) ** 2
    else:
        raise ValueError("kind must be 'rho' or 'nu'")

    H = np.empty((len(v_grid), len(t_grid)))

    for i, v in enumerate(v_grid):
        a = solve_section_offset(v, b, weight)
        H[i, :] = clip01(a + b * weight)

    return H


# ============================================================
# Functionals from H = ∂_1 C
# ============================================================


def xi_from_H(H):
    return 6.0 * trapz2(H**2, t_grid, v_grid) - 2.0


def rho_from_H(H):
    integrand = (1.0 - t_grid)[None, :] * H
    return 12.0 * trapz2(integrand, t_grid, v_grid) - 3.0


def nu_from_H(H):
    integrand = ((1.0 - t_grid) ** 2)[None, :] * H
    return 12.0 * trapz2(integrand, t_grid, v_grid) - 2.0


# ============================================================
# Boundary construction
# ============================================================


def boundary_curve(kind, b_min=B_MIN, b_max=B_MAX, n_b=N_B):
    """
    Returns a numerical parametrization of the upper boundary:
      kind = "rho" -> (xi(b), U_rho(xi(b)))
      kind = "nu"  -> (xi(b), U_nu(xi(b)))
    """
    b_values = np.geomspace(b_min, b_max, n_b)
    xi_values = []
    upper_values = []

    for b in b_values:
        H = build_H_matrix(kind, b)
        xi_val = xi_from_H(H)

        if kind == "rho":
            upper_val = rho_from_H(H)
        elif kind == "nu":
            upper_val = nu_from_H(H)
        else:
            raise ValueError("kind must be 'rho' or 'nu'")

        xi_values.append(xi_val)
        upper_values.append(upper_val)

    xi_values = np.asarray(xi_values)
    upper_values = np.asarray(upper_values)
    b_values = np.asarray(b_values)

    order = np.argsort(xi_values)
    xi_values = xi_values[order]
    upper_values = upper_values[order]
    b_values = b_values[order]

    keep = np.concatenate([[True], np.diff(xi_values) > 1e-10])
    xi_values = xi_values[keep]
    upper_values = upper_values[keep]
    b_values = b_values[keep]

    return xi_values, upper_values, b_values


# ============================================================
# Explicit upper boundary for (xi, psi)
# ============================================================


def upper_psi(x):
    """
    For Fréchet mixtures:
        xi = alpha^2, psi = alpha
    hence U_psi(x) = sqrt(x).
    """
    x = np.asarray(x)
    return np.sqrt(np.clip(x, 0.0, 1.0))


# ============================================================
# Formatting helpers
# ============================================================


def fmt(x, digits=6):
    return f"{x:.{digits}f}"


def latex_table(results, digits=6):
    """
    Returns a ready-to-paste LaTeX tabular.
    """
    r1 = results["nu_minus_rho"]
    r2 = results["rho_minus_psi"]

    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \begin{tabular}{lccc}",
        r"        \hline",
        r"        Comparison & maximizing $\xi$ & maximal gap & boundary values at maximizer \\",
        r"        \hline",
        (
            r"        $U_\nu(\xi)-U_\rho(\xi)$"
            + f" & {fmt(r1['xi_star'], digits)}"
            + f" & {fmt(r1['gap_star'], digits)}"
            + f" & $U_\\nu={fmt(r1['U_nu_at_star'], digits)},\\ U_\\rho={fmt(r1['U_rho_at_star'], digits)}$ \\\\"
        ),
        (
            r"        $U_\rho(\xi)-U_\psi(\xi)$"
            + f" & {fmt(r2['xi_star'], digits)}"
            + f" & {fmt(r2['gap_star'], digits)}"
            + f" & $U_\\rho={fmt(r2['U_rho_at_star'], digits)},\\ U_\\psi={fmt(r2['U_psi_at_star'], digits)}$ \\\\"
        ),
        r"        \hline",
        r"    \end{tabular}",
        r"    \caption{Numerically computed maximal vertical gaps between the upper boundary functions.}",
        r"    \label{tab:maximal_gaps}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_summary_sentence(results, digits=4):
    r1 = results["nu_minus_rho"]
    r2 = results["rho_minus_psi"]

    return (
        r"Numerical computations indicate that "
        + r"\(\max_{\xi\in[0,1]}(U_\nu(\xi)-U_\rho(\xi))\approx "
        + fmt(r1["gap_star"], digits)
        + r"\), attained at \(\xi\approx "
        + fmt(r1["xi_star"], digits)
        + r"\), while "
        + r"\(\max_{\xi\in[0,1]}(U_\rho(\xi)-U_\psi(\xi))\approx "
        + fmt(r2["gap_star"], digits)
        + r"\), attained at \(\xi\approx "
        + fmt(r2["xi_star"], digits)
        + r"\)."
    )


# ============================================================
# Main comparison routine
# ============================================================


def compute_max_gaps(make_plots=True):
    xi_rho, U_rho_vals, _ = boundary_curve("rho")
    xi_nu, U_nu_vals, _ = boundary_curve("nu")

    U_rho = PchipInterpolator(xi_rho, U_rho_vals)
    U_nu = PchipInterpolator(xi_nu, U_nu_vals)

    # max_x [U_nu(x) - U_rho(x)]
    x_left = max(xi_rho.min(), xi_nu.min())
    x_right = min(xi_rho.max(), xi_nu.max())

    def gap_nu_rho(x):
        return float(U_nu(x) - U_rho(x))

    opt1 = minimize_scalar(
        lambda x: -gap_nu_rho(x),
        bounds=(x_left, x_right),
        method="bounded",
        options={"xatol": 1e-8},
    )
    x_star_1 = opt1.x
    gap_star_1 = gap_nu_rho(x_star_1)

    # max_x [U_rho(x) - U_psi(x)]
    x_left_2 = xi_rho.min()
    x_right_2 = xi_rho.max()

    def gap_rho_psi(x):
        return float(U_rho(x) - upper_psi(x))

    opt2 = minimize_scalar(
        lambda x: -gap_rho_psi(x),
        bounds=(x_left_2, x_right_2),
        method="bounded",
        options={"xatol": 1e-8},
    )
    x_star_2 = opt2.x
    gap_star_2 = gap_rho_psi(x_star_2)

    results = {
        "nu_minus_rho": {
            "xi_star": x_star_1,
            "gap_star": gap_star_1,
            "U_rho_at_star": float(U_rho(x_star_1)),
            "U_nu_at_star": float(U_nu(x_star_1)),
        },
        "rho_minus_psi": {
            "xi_star": x_star_2,
            "gap_star": gap_star_2,
            "U_rho_at_star": float(U_rho(x_star_2)),
            "U_psi_at_star": float(upper_psi(x_star_2)),
        },
    }

    print("=" * 72)
    print("NUMERICAL MAXIMAL GAP BETWEEN U_nu AND U_rho")
    print("=" * 72)
    print(f"x*        = {results['nu_minus_rho']['xi_star']:.12f}")
    print(f"U_rho(x*) = {results['nu_minus_rho']['U_rho_at_star']:.12f}")
    print(f"U_nu(x*)  = {results['nu_minus_rho']['U_nu_at_star']:.12f}")
    print(f"max gap   = {results['nu_minus_rho']['gap_star']:.12f}")
    print()

    print("=" * 72)
    print("NUMERICAL MAXIMAL GAP BETWEEN U_rho AND U_psi")
    print("=" * 72)
    print(f"x*        = {results['rho_minus_psi']['xi_star']:.12f}")
    print(f"U_rho(x*) = {results['rho_minus_psi']['U_rho_at_star']:.12f}")
    print(f"U_psi(x*) = {results['rho_minus_psi']['U_psi_at_star']:.12f}")
    print(f"max gap   = {results['rho_minus_psi']['gap_star']:.12f}")
    print()

    print("=" * 72)
    print("READY-TO-PASTE LATEX SENTENCE")
    print("=" * 72)
    print(latex_summary_sentence(results, digits=4))
    print()

    print("=" * 72)
    print("READY-TO-PASTE LATEX TABLE")
    print("=" * 72)
    print(latex_table(results, digits=6))
    print()

    if make_plots:
        xs_common = np.linspace(x_left, x_right, 1200)
        xs_rho = np.linspace(x_left_2, x_right_2, 1200)

        plt.figure(figsize=(8, 5))

        plt.plot(
            xs_common,
            [gap_nu_rho(x) for x in xs_common],
            label=r"$U_\nu(\xi)-U_\rho(\xi)$",
            lw=2,
        )
        plt.plot(
            xs_rho,
            [gap_rho_psi(x) for x in xs_rho],
            label=r"$U_\rho(\xi)-U_\psi(\xi)$",
            lw=2,
        )

        plt.axvline(x_star_1, linestyle="--", lw=2)
        plt.axvline(x_star_2, linestyle="--", lw=2, color="orange")

        plt.xlabel(r"$\xi$")
        # plt.ylabel("gap")
        # plt.title("
        plt.grid(linestyle="--")
        plt.ylim(0, 0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    results = compute_max_gaps(make_plots=True)
