"""
Find, for several copula families, the parameter value that maximises
the gap  τ − ξ, and typeset the results in a LaTeX table.

The *.pkl files are assumed to have the column layout
    0 = parameter value, 1 = xi, 3 = rho, 4 = tau
and live in  copul/docs/rank_correlation_estimates/.
"""

import pickle
from pathlib import Path
import importlib.resources as pkg_resources

import numpy as np
import scipy.interpolate as si


# ------------------------------------------------------------------ helpers
def fetch_measure(arr: np.ndarray, which: str = "xi") -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (x, y) columns for *which* ∈ {"xi","rho","tau"} from the raw array.
    """
    col_map = {
        "xi": 1,
        "rho": 3,
        "tau": 4,
        "footrule": 5,
        "ginis_gamma": 6,
        "blomqvists_beta": 7,
    }
    try:
        col = col_map[which]
    except KeyError:
        raise ValueError(f"unknown measure {which!r}")
    return arr[:, 0], arr[:, col]


def maximize_tau_minus_xi(  # <-- renamed
    x: np.ndarray, tau: np.ndarray, xi: np.ndarray
) -> tuple[float, float, float]:
    """
    Use cubic splines, then a dense grid search, to find the parameter that
    maximises τ − ξ.  Returns (best_param, tau(best), xi(best)).
    """
    # spline interpolants (no extrapolation)
    s_tau = si.CubicSpline(x, tau, extrapolate=False)
    s_xi = si.CubicSpline(x, xi, extrapolate=False)

    # dense grid inside [x.min, x.max]
    x_dense = np.linspace(x.min(), x.max(), 20_001)
    diff = s_tau(x_dense) - s_xi(x_dense)

    idx_max = np.nanargmax(diff)  # ignore possible NaNs (edges)
    best_x = x_dense[idx_max]
    return best_x, s_tau(best_x), s_xi(best_x)


# ---------------------------------------------------------------- optimizer
def optimize_for(family: str, data_dir: Path) -> tuple[float, float, float]:
    """
    Load data for *family* and return (parameter, tau, xi) at the maximum gap.
    """
    file = data_dir / f"{family}Data.pkl"
    arr = pickle.loads(file.read_bytes())

    x, xi = fetch_measure(arr, "xi")
    _, tau = fetch_measure(arr, "tau")

    return maximize_tau_minus_xi(x, tau, xi)  # (param, tau, xi)


# ---------------------------------------------------------------- main block
def main() -> None:
    families = [
        "Clayton",
        "Frank",
        "GumbelHougaard",
        "Joe",
    ]  # Gaussian handled separately
    # directory with the *.pkl files
    with pkg_resources.path("copul", "docs") as docs_path:
        data_dir = docs_path / "rank_correlation_estimates"

    rows = []
    # --- numeric rows ----------------------------------------------------
    for fam in families:
        p, t, x = optimize_for(fam, data_dir)
        rows.append((fam, p, t, x, t - x))

    # --- analytic Gaussian row ------------------------------------------
    p_gauss = 1 / np.sqrt(2)  # optimum ≈ ρ = 1/√2
    tau_gauss = 2 / np.pi * np.arcsin(p_gauss)  # τ(ρ) = 2/π·arcsin(ρ)
    xi_gauss = 3 / np.pi * np.arcsin(3 / 4) - 0.5  # Chatterjee's ξ for Gaussian
    rows.append(("Gaussian", p_gauss, tau_gauss, xi_gauss, tau_gauss - xi_gauss))

    # --- manual C_b row --------------------------------------------------
    rows.append(("\\(C_b\\)", 1.0, 0.5, 0.3, 0.2))  # τ = 0.5, ξ = 0.3 at b = 1

    # --- sort alphabetically by family name -----------------------------
    rows_sorted = sorted(rows, key=lambda r: r[0].lower())

    # -------------------------------------------------------------------- table
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"    \toprule")
    print(r"    Family & Parameter & $\tau$ & $\xi$ & $\tau-\xi$ \\")
    print(r"    \midrule")
    for fam, p, t_, x_, diff in rows_sorted:
        print(
            f"    {fam:14s} & {p:10.3f} & {t_:10.3f} & {x_:10.3f} & {diff:10.3f} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")

    # --------------- caption --------------------------------------------
    print(r"  \caption{%")
    print(
        r"  Parameter values that approximately maximise the gap "
        r"$\tau-\xi$ for the listed copula families, together with the "
    )
    print(
        r"  corresponding Kendall's~$\tau$, Chatterjee's~$\xi$, and their "
        r"  difference.  Except for the Gaussian family and the band copula "
        r"  $C_b$, where closed-form formulae are available, the entries are "
        r"  obtained by a dense grid search for the parameter and numerical "
        r"  approximations of $\tau$ and $\xi$.}"
    )
    print(r"  \label{tab:tau_minus_xi_max}")
    print(r"\end{table}")
    print("Done!")


if __name__ == "__main__":
    main()
