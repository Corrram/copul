#!/usr/bin/env python3
"""
Generate a LaTeX table for (beta, xi) at the parameter value that maximises
the gap beta - xi for several copula families.

The script expects each *.pkl to contain either:
  (A) an object with attributes: params, xi, (blomqvists_beta, rho, tau, footrule optional), or
  (B) a numpy array with columns laid out as below.

Adjust COLUMN_MAP_NDARRAY if your on-disk arrays use different indices.

Data files live in: copul/docs/rank_correlation_estimates/
and are named:      <Family>_data.pkl
"""

import pickle
from pathlib import Path
import importlib.resources as pkg_resources
from typing import Optional, Tuple

import numpy as np
import scipy.interpolate as si
from dataclasses import dataclass


# ------------------- ndarray fallback column map (edit if needed) -------------------
# If .pkl contains a raw numpy array, we’ll use these indices.
# (Adjust the index for 'blomqvists_beta' to match your array structure if needed).
COLUMN_MAP_NDARRAY = {
    "param": 0,
    "chatterjees_xi": 1,
    "rho": 2,
    "tau": 3,
    "footrule": 4,
    "blomqvist_beta": 5
}


# ------------------- typed container (used if unpickled object has attributes) ------
@dataclass
class CorrelationData:
    params: np.ndarray
    xi: np.ndarray
    rho: Optional[np.ndarray] = None
    tau: Optional[np.ndarray] = None
    footrule: Optional[np.ndarray] = None
    ginis_gamma: Optional[np.ndarray] = None
    blomqvists_beta: Optional[np.ndarray] = None
    nu: Optional[np.ndarray] = None


# ------------------------------------------------------------------ helpers
def get_params_and_measure(data_obj, which: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robustly extract (params, measure) from either an attribute-based object
    or a raw ndarray according to COLUMN_MAP_NDARRAY.
    """
    # Attribute-style (preferred)
    if (
        hasattr(data_obj, "params")
        and hasattr(data_obj, "values")
        and which in data_obj.values
    ):
        params = np.asarray(data_obj.params)
        meas = np.asarray(data_obj.values[which])
        return params, meas

    # Raw ndarray fallback
    arr = np.asarray(data_obj)
    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported data format for {which}: expected 2D array for ndarray fallback."
        )

    try:
        pcol = COLUMN_MAP_NDARRAY["param"]
        mcol = COLUMN_MAP_NDARRAY[which]
    except KeyError:
        raise ValueError(f"Unknown measure {which!r} in COLUMN_MAP_NDARRAY.")

    params = arr[:, pcol]
    meas = arr[:, mcol]
    return params, meas


def maximize_beta_minus_xi(
    x: np.ndarray, beta: np.ndarray, xi: np.ndarray
) -> Tuple[float, float, float]:
    """
    Use cubic splines and a dense grid to find the parameter that
    maximises beta - xi. Returns (best_param, beta(best), xi(best)).
    """
    # Monotone-ish smoothing; no extrapolation (edges can be NaN)
    s_beta = si.CubicSpline(x, beta, extrapolate=False)
    s_xi = si.CubicSpline(x, xi, extrapolate=False)

    # Dense search in observed parameter range
    x_dense = np.linspace(np.nanmin(x), np.nanmax(x), 20_001)
    diff = s_beta(x_dense) - s_xi(x_dense)

    # Ignore NaNs near edges
    idx = np.nanargmax(diff)
    best_x = float(x_dense[idx])
    return best_x, float(s_beta(best_x)), float(s_xi(best_x))


# ---------------------------------------------------------------- optimizer
def optimize_for(family: str, data_dir: Path) -> Tuple[float, float, float]:
    """
    Load data for a family and return (parameter, beta, xi) at the max gap.
    """
    file = data_dir / f"{family}_data.pkl"
    with open(file, "rb") as f:
        data_obj = pickle.load(f)

    params, xi = get_params_and_measure(data_obj, "chatterjees_xi")
    _, beta = get_params_and_measure(data_obj, "blomqvist_beta")

    return maximize_beta_minus_xi(params, beta, xi)


# ---------------------------------------------------------------- main block
def main() -> None:
    families = [
        "Frank",
        "BivClayton",
        "Gaussian",
        "GumbelHougaard",
        # "Joe",
    ]

    with pkg_resources.path("copul", "docs/rank_correlation_estimates") as data_dir:
        rows = []

        # --- numeric rows from saved data --------------------------------
        for fam in families:
            p, beta_val, xi_val = optimize_for(fam, data_dir)
            rows.append((fam, p, beta_val, xi_val, beta_val - xi_val))

    # --- manual C_a^\# row --------------------------------------------------
    # For the symmetric 2x2 checkerboard boundary copula C_a^\# at a=1/2:
    # beta(C_{0.5}^\#) = 1.0, xi(C_{0.5}^\#) = 0.5
    # This theoretically maximizes the gap beta - xi for the entire exact region.
    beta_ca = 1.0
    xi_ca = 0.5
    rows.append((r"\(C_a^\#\)", 0.5, beta_ca, xi_ca, beta_ca - xi_ca))

    # --- analytic Marshall-Olkin (alpha_1=1, alpha_2 free) row --------------
    # beta = 2^a2 - 1,  xi = 2*a2 / (3 - a2),  a2 in [0, 1]
    a2_grid = np.linspace(0, 1, 20_001)
    mo_beta = 2.0 ** a2_grid - 1.0
    mo_xi = 2.0 * a2_grid / (3.0 - a2_grid)
    mo_best_a2, mo_beta_val, mo_xi_val = maximize_beta_minus_xi(a2_grid, mo_beta, mo_xi)
    rows.append((r"Marshall-Olkin ($\alpha_1=1$)", mo_best_a2, mo_beta_val, mo_xi_val,
                 mo_beta_val - mo_xi_val))

    # --- sort alphabetically by family name -----------------------------
    rows_sorted = sorted(rows, key=lambda r: r[0].lower())

    # ------------------------------------------------------------ LaTeX table
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"    \toprule")
    print(r"    Family & Parameter & $\beta$ & $\xi$ & $\beta-\xi$ \\")
    print(r"    \midrule")
    for fam, p, beta_, x_, diff in rows_sorted:
        print(
            f"    {fam:30s} & {p:10.3f} & {beta_:10.3f} & {x_:10.3f} & {diff:10.3f} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")

    # --------------- caption --------------------------------------------
    print(r"  \caption{%")
    print(
        r"  Parameter values that approximately maximise the gap "
        r"$\beta-\xi$ for the listed copula families, together with the "
    )
    print(
        r"  corresponding Blomqvist's beta~$\beta$, Chatterjee's rank correlation~$\xi$, and their "
        r"  difference. Except for special cases with closed-form evaluations, "
        r"  the entries are obtained by a dense grid search in the parameter "
        r"  and cubic-spline interpolation of $\beta$ and $\xi$.}"
    )
    print(r"  \label{tab:beta_minus_xi_max}")
    print(r"\end{table}")
    print("\nDone!")


if __name__ == "__main__":
    main()