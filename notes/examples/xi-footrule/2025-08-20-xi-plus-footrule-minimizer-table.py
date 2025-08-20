"""
Find, for several copula families, the parameter value that minimises
the sum  ψ + ξ (Spearman's Footrule plus Chatterjee's xi), and typeset
the results in a LaTeX table.

The *.pkl files are assumed to have the column layout
    0 = parameter value, 1 = xi, 4 = footrule
and live in  copul/docs/rank_correlation_estimates/.
"""

import pickle
from pathlib import Path
import importlib.resources as pkg_resources
from typing import Optional

import numpy as np
import scipy.interpolate as si
from dataclasses import dataclass


# This class definition is assumed to exist
@dataclass
class CorrelationData:
    """Class to store correlation data for various metrics."""

    params: np.ndarray
    xi: np.ndarray
    rho: Optional[np.ndarray] = None
    tau: Optional[np.ndarray] = None
    footrule: Optional[np.ndarray] = None
    ginis_gamma: Optional[np.ndarray] = None
    blomqvists_beta: Optional[np.ndarray] = None


# ------------------------------------------------------------------ helpers
def fetch_measure(arr: CorrelationData, which: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (parameter, measure) columns from the CorrelationData object.
    """
    col_map = {"xi": "xi", "footrule": "footrule"}
    try:
        measure_attr = col_map[which]
    except KeyError:
        raise ValueError(f"unknown measure {which!r}")

    if not hasattr(arr, measure_attr) or getattr(arr, measure_attr) is None:
        raise AttributeError(
            f"Data object for {arr} does not contain '{measure_attr}' data."
        )

    return arr.params, getattr(arr, measure_attr)


def minimize_footrule_plus_xi(
    x: np.ndarray, footrule: np.ndarray, xi: np.ndarray
) -> tuple[float, float, float]:
    """
    Use cubic splines and a dense grid search to find the parameter that
    minimises ψ + ξ. Returns (best_param, footrule(best), xi(best)).
    """
    # Create spline interpolants (without extrapolation)
    s_footrule = si.CubicSpline(x, footrule, extrapolate=False)
    s_xi = si.CubicSpline(x, xi, extrapolate=False)

    # Search on a dense grid within the parameter's observed range
    x_dense = np.linspace(x.min(), x.max(), 20_001)
    the_sum = s_footrule(x_dense) + s_xi(x_dense)

    idx_min = np.nanargmin(the_sum)  # Ignore potential NaNs at the edges
    best_x = x_dense[idx_min]
    return best_x, s_footrule(best_x), s_xi(best_x)


# ---------------------------------------------------------------- optimizer
def optimize_for(family: str, data_dir: Path) -> tuple[float, float, float]:
    """
    Load data for a family and return (parameter, footrule, xi) at the min sum.
    """
    file = data_dir / f"{family} Copula_data.pkl"
    with open(file, "rb") as f:
        data_obj = pickle.load(f)

    params, xi = fetch_measure(data_obj, "xi")
    _, footrule = fetch_measure(data_obj, "footrule")

    return minimize_footrule_plus_xi(params, footrule, xi)


# ---------------------------------------------------------------- main block
def main() -> None:
    families = [
        "BivClayton",
        "Frank",
        "Gaussian",
        "GumbelHougaard",
        "Joe",
    ]

    try:
        # Use importlib.resources for robust path handling
        with pkg_resources.path("copul", "docs/rank_correlation_estimates") as data_dir:
            rows = []
            # --- numeric rows from saved data --------------------------------
            for fam in families:
                p, fr, x = optimize_for(fam, data_dir)
                rows.append((fam.replace("Biv", ""), p, fr, x, fr + x))

    except ModuleNotFoundError:
        print("Could not find 'copul' package. Please ensure it is installed.")
        return

    # --- manual independence copula row ----------------------------------
    # For the independence copula Pi, ψ=0 and ξ=0, so ψ+ξ=0.
    rows.append(("\\Pi", np.nan, 0.0, 0.0, 0.0))

    # --- sort alphabetically by family name -----------------------------
    rows_sorted = sorted(rows, key=lambda r: r[0].lower())

    # ------------------------------------------------------------ LaTeX table
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"    \toprule")
    print(r"    Family & Parameter & $\psi$ & $\xi$ & $\psi+\xi$ \\")
    print(r"    \midrule")
    for fam, p, fr_, x_, sum_val in rows_sorted:
        # Handle nan for the parameter of Pi
        param_str = f"{p:10.3f}" if not np.isnan(p) else f"{'N/A':>10}"
        print(
            f"    {fam:14s} & {param_str} & {fr_:10.3f} & {x_:10.3f} & {sum_val:10.3f} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")

    # --------------- caption --------------------------------------------
    print(r"  \caption{%")
    print(
        r"  Parameter values that approximately minimise the sum "
        r"$\psi+\xi$ for the listed copula families, together with the "
    )
    print(
        r"  corresponding Spearman's~footrule~$\psi$, Chatterjee's~$\xi$, and their "
        r"  sum. The entries are obtained by a dense grid search for the "
        r"  parameter and numerical approximations of $\psi$ and $\xi$.}"
    )
    print(r"  \label{tab:footrule_plus_xi_min}")
    print(r"\end{table}")
    print("\nDone!")


if __name__ == "__main__":
    main()
