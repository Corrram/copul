#!/usr/bin/env python3
"""
Generate a LaTeX table for (nu, xi) at the parameter value that maximises
the gap  nu − xi for several copula families.

The script expects each *.pkl to contain either:
  (A) an object with attributes: params, xi, (nu, rho, tau, footrule optional), or
  (B) a numpy array with columns laid out as below.

Adjust COLUMN_MAP_NDARRAY if your on-disk arrays use different indices.

Data files live in: copul/docs/rank_correlation_estimates/
and are named:      <Family> Copula_data.pkl
"""

import pickle
from pathlib import Path
import importlib.resources as pkg_resources
from typing import Optional, Tuple

import numpy as np
import scipy.interpolate as si
from dataclasses import dataclass


# ------------------- ndarray fallback column map (edit if needed) -------------------
# If .pkl contains a raw numpy array, we’ll use these indices:
#   0 = parameter value, 1 = xi, 2 = rho, 3 = tau, 4 = footrule, 5 = nu
COLUMN_MAP_NDARRAY = {"param": 0, "xi": 1, "rho": 2, "tau": 3, "footrule": 4, "nu": 5}


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


def maximize_nu_minus_xi(
    x: np.ndarray, nu: np.ndarray, xi: np.ndarray
) -> Tuple[float, float, float]:
    """
    Use cubic splines and a dense grid to find the parameter that
    maximises ν − ξ. Returns (best_param, nu(best), xi(best)).
    """
    # Monotone-ish smoothing; no extrapolation (edges can be NaN)
    s_nu = si.CubicSpline(x, nu, extrapolate=False)
    s_xi = si.CubicSpline(x, xi, extrapolate=False)

    # Dense search in observed parameter range
    x_dense = np.linspace(np.nanmin(x), np.nanmax(x), 20_001)
    diff = s_nu(x_dense) - s_xi(x_dense)

    # Ignore NaNs near edges
    idx = np.nanargmax(diff)
    best_x = float(x_dense[idx])
    return best_x, float(s_nu(best_x)), float(s_xi(best_x))


# ---------------------------------------------------------------- optimizer
def optimize_for(family: str, data_dir: Path) -> Tuple[float, float, float]:
    """
    Load data for a family and return (parameter, nu, xi) at the max gap.
    """
    file = data_dir / f"{family} Copula_data.pkl"
    with open(file, "rb") as f:
        data_obj = pickle.load(f)

    params, xi = get_params_and_measure(data_obj, "chatterjees_xi")
    _, nu = get_params_and_measure(data_obj, "blests_nu")

    return maximize_nu_minus_xi(params, nu, xi)


# ---------------------------------------------------------------- main block
def main() -> None:
    families = [
        "Frank",
        "BivClayton",
        "Gaussian",
        "GumbelHougaard",
        "Joe",
    ]

    with pkg_resources.path("copul", "docs/rank_correlation_estimates") as data_dir:
        rows = []

        # --- numeric rows from saved data --------------------------------
        for fam in families:
            p, nu_val, xi_val = optimize_for(fam, data_dir)
            rows.append((fam, p, nu_val, xi_val, nu_val - xi_val))

    # --- manual C_b row --------------------------------------------------
    # For the band copula C_b at b=1:
    #   xi(C_1) = 32/105,  nu(C_1) = 76/105
    nu_cb = 76.0 / 105.0
    xi_cb = 32.0 / 105.0
    rows.append(("\\(C_b\\)", 1.0, nu_cb, xi_cb, nu_cb - xi_cb))

    # --- sort alphabetically by family name -----------------------------
    rows_sorted = sorted(rows, key=lambda r: r[0].lower())

    # ------------------------------------------------------------ LaTeX table
    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"    \toprule")
    print(r"    Family & Parameter & $\nu$ & $\xi$ & $\nu-\xi$ \\")
    print(r"    \midrule")
    for fam, p, nu_, x_, diff in rows_sorted:
        print(
            f"    {fam:14s} & {p:10.3f} & {nu_:10.3f} & {x_:10.3f} & {diff:10.3f} \\\\"
        )
    print(r"    \bottomrule")
    print(r"  \end{tabular}")

    # --------------- caption --------------------------------------------
    print(r"  \caption{%")
    print(
        r"  Parameter values that approximately maximise the gap "
        r"$\nu-\xi$ for the listed copula families, together with the "
    )
    print(
        r"  corresponding Blest's rank correlation~$\nu$, Chatterjee's~$\xi$, and their "
        r"  difference. Except for special cases with closed-form evaluations, "
        r"  the entries are obtained by a dense grid search in the parameter "
        r"  and cubic-spline interpolation of $\nu$ and $\xi$.}"
    )
    print(r"  \label{tab:nu_minus_xi_max}")
    print(r"\end{table}")
    print("\nDone!")


if __name__ == "__main__":
    main()
