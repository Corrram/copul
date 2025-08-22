"""
Find, for several copula families, the parameter value that maximises
the gap  ψ − ξ (Spearman's Footrule minus Chatterjee's xi), and typeset
the results in a LaTeX table.

The *.pkl files are assumed to have the column layout
    0 = parameter value, 1 = xi, 5 = footrule
and live in  copul/docs/rank_correlation_estimates/.
"""

import pickle
from pathlib import Path
import importlib.resources as pkg_resources
from typing import Optional

import numpy as np
import scipy.interpolate as si

from dataclasses import dataclass


# Add this class definition to your script
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
def fetch_measure(arr: np.ndarray, which: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (parameter, measure) columns from the raw data array.
    """
    # As per the CorrelationData object structure
    col_map = {"xi": 1, "rho": 2, "tau": 3, "footrule": 4}
    try:
        col_map[which]
    except KeyError:
        raise ValueError(f"unknown measure {which!r}")
    return arr.params, getattr(arr, which)


def maximize_footrule_minus_xi(
    x: np.ndarray, footrule: np.ndarray, xi: np.ndarray
) -> tuple[float, float, float]:
    """
    Use cubic splines and a dense grid search to find the parameter that
    maximises ψ − ξ. Returns (best_param, footrule(best), xi(best)).
    """
    # Create spline interpolants (without extrapolation)
    s_footrule = si.CubicSpline(x, footrule, extrapolate=False)
    s_xi = si.CubicSpline(x, xi, extrapolate=False)

    # Search on a dense grid within the parameter's observed range
    x_dense = np.linspace(x.min(), x.max(), 20_001)
    diff = s_footrule(x_dense) - s_xi(x_dense)

    idx_max = np.nanargmax(diff)  # Ignore potential NaNs at the edges
    best_x = x_dense[idx_max]
    return best_x, s_footrule(best_x), s_xi(best_x)


# ---------------------------------------------------------------- optimizer
def optimize_for(family: str, data_dir: Path) -> tuple[float, float, float]:
    """
    Load data for a family and return (parameter, footrule, xi) at the max gap.
    """
    file = data_dir / f"{family} Copula_data.pkl"
    with open(file, "rb") as f:
        data_obj = pickle.load(f)

    params, xi = fetch_measure(data_obj, "xi")
    _, footrule = fetch_measure(data_obj, "footrule")

    return maximize_footrule_minus_xi(params, footrule, xi)


# ---------------------------------------------------------------- main block
def main() -> None:
    families = [
        "BivClayton",
        "Frank",
        "Gaussian",
        "GumbelHougaard",
        "Joe",
    ]  # Gaussian and C_b are handled separately

    with pkg_resources.path("copul", "docs/rank_correlation_estimates") as data_dir:
        rows = []
        # --- numeric rows from saved data --------------------------------
        for fam in families:
            p, fr, x = optimize_for(fam, data_dir)
            rows.append((fam, p, fr, x, fr - x))

    # --- manual C_b row --------------------------------------------------
    # For the band copula C_b at b=1, ψ=2/3 and ξ=0.3
    footrule_cb = 2 / 3
    xi_cb = 0.3
    rows.append(("\\(C_b\\)", 1.0, footrule_cb, xi_cb, footrule_cb - xi_cb))

    # --- sort alphabetically by family name -----------------------------
    rows_sorted = sorted(rows, key=lambda r: r[0].lower())

    # ------------------------------------------------------------ LaTeX table
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"    \toprule")
    print(r"    Family & Parameter & $\psi$ & $\xi$ & $\psi-\xi$ \\")
    print(r"    \midrule")
    for fam, p, fr_, x_, diff in rows_sorted:
        print(
            f"    {fam:14s} & {p:10.3f} & {fr_:10.3f} & {x_:10.3f} & {diff:10.3f} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")

    # --------------- caption --------------------------------------------
    print(r"  \caption{%")
    print(
        r"  Parameter values that approximately maximise the gap "
        r"$\psi-\xi$ for the listed copula families, together with the "
    )
    print(
        r"  corresponding Spearman's~footrule~$\psi$, Chatterjee's~$\xi$, and their "
        r"  difference. Except for the Gaussian family and the band copula "
        r"  $C_b$, where closed-form formulae are available, the entries are "
        r"  obtained by a dense grid search for the parameter and numerical "
        r"  approximations of $\psi$ and $\xi$.}"
    )
    print(r"  \label{tab:footrule_minus_xi_max}")
    print(r"\end{table}")
    print("\nDone!")


if __name__ == "__main__":
    main()
