#!/usr/bin/env python3
"""
Comprehensive simulation study for the checkerboard estimator xi_n^kappa.

Addresses reviewer comments (p.41, Anmerkungen_PhD_Rockel.pdf):
 - Justification for the arithmetic-mean estimator
 - Whether xi_n^kappa in [0,1] (possible negative values)
 - Finite-sample performance across copula families and dependence levels
 - Comparison with the classical nearest-neighbour estimator xi_n
"""

import os
import warnings
import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import copul
from copul.chatterjee import xi_ncalculate

warnings.filterwarnings("ignore")
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# Font sizes are set for a target print width of ~7.5 in, scaled to ~5.9 in (A4 \linewidth
# with 30 mm margins).  Scale factor ≈ 0.79, so a 14 pt font appears as ~11 pt in print.
_SUPTITLE  = 18
_AX_TITLE  = 14
_AX_LABEL  = 14
_TICK      = 13
_LEGEND    = 13

mpl.rcParams.update({
    "font.size":        _AX_LABEL,
    "axes.titlesize":   _AX_TITLE,
    "axes.labelsize":   _AX_LABEL,
    "xtick.labelsize":  _TICK,
    "ytick.labelsize":  _TICK,
    "legend.fontsize":  _LEGEND,
    "figure.titlesize": _SUPTITLE,
})

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

KAPPA = 1 / 3
N_REPS = 300
SEED = 42
N_TRUE = 300_000       # large-sample size for reference-xi estimation
SAMPLE_SIZES = [100, 200, 500, 1000, 2000, 5000]

# Target dependence levels for the calibrated simulation design.
# Parameters are chosen family-by-family so that the corresponding
# reference Chatterjee xi is close to these targets.
TARGET_LEVELS = [
    ("low", 0.05),
    ("moderate", 0.30),
    ("strong", 0.65),
]

FAMILIES = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]

# Non-Gaussian calibration is Monte Carlo based. The final reference xi values
# are still recomputed with N_TRUE in run_simulation().
N_CALIB = 80_000
CALIBRATION_ITERS = 14

PARAM_BOUNDS = {
    "Clayton": (0.05, 50.0),
    "GumbelHougaard": (1.001, 50.0),
    "Frank": (0.05, 80.0),
}

# Filled in at runtime by calibrate_design().
SETTINGS: list[tuple[str, dict[str, float]]] = []
DEP_LEVEL: dict[tuple[str, float], str] = {}
ESTIMATOR_KEYS = ["check_avg", "nn", "check_pi", "check_min"]

ESTIMATOR_LABELS = {
    "check_avg": r"CheckAvg ($\xi_n^\kappa$)",
    "nn":        r"Baseline ($\xi_n$)",
    "check_pi":  r"CheckPi ($\underline{\xi}_n^\kappa$)",
    "check_min": r"CheckMin ($\overline{\xi}_n^\kappa$)",
}

ESTIMATOR_COLORS = {
    "check_pi":  "#1f77b4",
    "check_min": "#ff7f0e",
    "check_avg": "#2ca02c",
    "nn":        "#d62728",
}

ESTIMATOR_STYLES = {
    "check_pi":  "--",
    "check_min": "-.",
    "check_avg": "-",
    "nn":        (0, (3, 1, 1, 1)),   # dash-dot-dot
}

FAMILY_DISPLAY = {
    "Gaussian":       "Gaussian",
    "Clayton":        "Clayton",
    "GumbelHougaard": "Gumbel--Hougaard",
    "Frank":          "Frank",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────


def gaussian_xi_exact(rho: float) -> float:
    """Exact Chatterjee xi for the bivariate Gaussian copula with correlation rho.

    Formula from Ansari & Fuchs (2022), Proposition 2.7:
        xi = 3/pi * arcsin((1 + rho^2) / 2) - 1/2.
    """
    return 3 / np.pi * np.arcsin((1 + rho ** 2) / 2) - 0.5


def make_copula(family: str, params: dict):
    if family == "Gaussian":
        return copul.Gaussian(params["rho"])
    if family == "Clayton":
        return copul.Clayton(params["theta"])
    if family == "GumbelHougaard":
        return copul.GumbelHougaard(params["theta"])
    if family == "Frank":
        return copul.Frank(params["theta"])
    raise ValueError(f"Unknown family: {family}")

def _param_name(family: str) -> str:
    if family == "Gaussian":
        return "rho"
    return "theta"


def _params_for(family: str, param: float) -> dict[str, float]:
    return {_param_name(family): float(param)}


def gaussian_rho_from_xi(target_xi: float) -> float:
    """Invert xi(C_rho) = 3/pi arcsin((1+rho^2)/2) - 1/2."""
    s = np.sin(np.pi * (target_xi + 0.5) / 3.0)
    rho_sq = 2.0 * s - 1.0
    return float(np.sqrt(max(rho_sq, 0.0)))


def _xi_for_calibration(
    family: str,
    param: float,
    seed: int,
    n_calib: int = N_CALIB,
) -> float:
    """Reference xi used only for parameter calibration."""
    if family == "Gaussian":
        return gaussian_xi_exact(param)

    cop = make_copula(family, _params_for(family, param))
    data = cop.rvs(n_calib, random_state=seed)
    return float(xi_ncalculate(data[:, 0], data[:, 1]))


def _calibrate_one_parameter(
    family: str,
    target_xi: float,
    level: str,
    seed: int = SEED,
) -> tuple[float, float]:
    """Choose a copula parameter whose xi is close to target_xi."""
    if family == "Gaussian":
        rho = gaussian_rho_from_xi(target_xi)
        return rho, gaussian_xi_exact(rho)

    lo, hi = PARAM_BOUNDS[family]

    xi_lo = _xi_for_calibration(family, lo, seed=seed + 11)
    xi_hi = _xi_for_calibration(family, hi, seed=seed + 13)

    if not (xi_lo <= target_xi <= xi_hi):
        raise RuntimeError(
            f"Calibration target xi={target_xi:.3f} for {family}/{level} "
            f"is outside bracket [{xi_lo:.3f}, {xi_hi:.3f}] "
            f"from parameters [{lo}, {hi}]."
        )

    for it in range(CALIBRATION_ITERS):
        mid = 0.5 * (lo + hi)
        xi_mid = _xi_for_calibration(
            family,
            mid,
            seed=seed + 1000 + 37 * it,
        )

        if xi_mid < target_xi:
            lo = mid
        else:
            hi = mid

    param = 0.5 * (lo + hi)
    xi_calib = _xi_for_calibration(
        family,
        param,
        seed=seed + 9999,
        n_calib=N_CALIB,
    )
    return float(param), float(xi_calib)


def _settings_from_design(
    design: pd.DataFrame,
) -> tuple[list[tuple[str, dict[str, float]]], dict[tuple[str, float], str]]:
    settings: list[tuple[str, dict[str, float]]] = []
    dep_level: dict[tuple[str, float], str] = {}

    for family in FAMILIES:
        for level, _target in TARGET_LEVELS:
            row = design[
                (design["family"] == family)
                & (design["dep_level"] == level)
            ].iloc[0]

            param = float(row["param"])
            settings.append((family, _params_for(family, param)))
            dep_level[(family, param)] = level

    return settings, dep_level


def calibrate_design(
    cache_path: str | None = None,
    force: bool = False,
) -> tuple[list[tuple[str, dict[str, float]]], dict[tuple[str, float], str]]:
    """Calibrate copula parameters to the target xi levels.

    The result is cached because non-Gaussian calibration is Monte Carlo based.
    """
    if cache_path is not None and os.path.exists(cache_path) and not force:
        design = pd.read_csv(cache_path)
        print(f"Loaded calibrated design from {cache_path}")
        return _settings_from_design(design)

    print("Calibrating copula parameters to target xi values ...")
    rows = []

    for family in FAMILIES:
        for level, target_xi in TARGET_LEVELS:
            param, xi_calib = _calibrate_one_parameter(
                family=family,
                target_xi=target_xi,
                level=level,
            )

            rows.append(
                {
                    "family": family,
                    "dep_level": level,
                    "target_xi": target_xi,
                    "param_name": _param_name(family),
                    "param": param,
                    "xi_calib": xi_calib,
                }
            )

            print(
                f"  {family:15s} {level:9s} "
                f"{_param_name(family)}={param:8.4f} "
                f"target={target_xi:.3f} xi_calib={xi_calib:.4f}"
            )

    design = pd.DataFrame(rows)

    if cache_path is not None:
        design.to_csv(cache_path, index=False)
        print(f"Saved calibrated design to {cache_path}")

    return _settings_from_design(design)

def reference_xi(family: str, params: dict, seed: int = 0) -> float:
    """Compute reference xi: exact for Gaussian, large-sample NN otherwise."""
    if family == "Gaussian":
        return gaussian_xi_exact(params["rho"])
    cop = make_copula(family, params)
    data = cop.rvs(N_TRUE, random_state=seed)
    return float(xi_ncalculate(data[:, 0], data[:, 1]))


def single_estimate(data: np.ndarray) -> dict[str, float]:
    """All four estimators from an (n x 2) data array."""
    df = pd.DataFrame(data, columns=["X", "Y"])
    cp_cop = copul.BivCheckPi.from_data(df, kappa=KAPPA)
    cm_cop = copul.BivCheckMin.from_data(df, kappa=KAPPA)
    xi_pi  = float(cp_cop.chatterjees_xi())
    xi_min = float(cm_cop.chatterjees_xi())
    xi_avg = (xi_pi + xi_min) / 2.0
    xi_nn  = float(xi_ncalculate(data[:, 0], data[:, 1]))
    return {"check_pi": xi_pi, "check_min": xi_min, "check_avg": xi_avg, "nn": xi_nn}


def rep_seed(base: int, fi: int, pi: int, ni: int, rep: int) -> int:
    """Deterministic seed for a Monte Carlo repetition."""
    return (base + fi * 1_000_000 + pi * 10_000 + ni * 100 + rep) % (2 ** 31)


# ─── SIMULATION ───────────────────────────────────────────────────────────────


def run_simulation() -> pd.DataFrame:
    print("Pre-computing reference xi values ...")
    true_xis: dict[tuple, float] = {}
    for family, params in SETTINGS:
        key = (family, list(params.values())[0])
        xi_ref = reference_xi(family, params, seed=SEED)
        true_xis[key] = xi_ref
        print(f"  {family:15s}  param={list(params.values())[0]:5.2f}  xi = {xi_ref:.4f}")

    print("\nRunning Monte Carlo experiments ...")
    records = []

    for fi, (family, params) in enumerate(tqdm(SETTINGS, desc="Settings")):
        param_val = list(params.values())[0]
        true_xi = true_xis[(family, param_val)]
        cop = make_copula(family, params)

        for ni, n in enumerate(SAMPLE_SIZES):
            raw: dict[str, list[float]] = {k: [] for k in ESTIMATOR_KEYS}

            for rep in range(N_REPS):
                seed = rep_seed(SEED, fi, 0, ni, rep)
                data = cop.rvs(n, random_state=seed)
                est = single_estimate(data)
                for k in ESTIMATOR_KEYS:
                    raw[k].append(est[k])

            for est_name, vals in raw.items():
                arr = np.array(vals)
                bias      = float(np.mean(arr) - true_xi)
                rmse      = float(np.sqrt(np.mean((arr - true_xi) ** 2)))
                frac_neg  = float(np.mean(arr < 0))
                frac_gt1  = float(np.mean(arr > 1))
                records.append(
                    {
                        "family":    family,
                        "param":     param_val,
                        "dep_level": DEP_LEVEL[(family, param_val)],
                        "true_xi":   true_xi,
                        "n":         n,
                        "estimator": est_name,
                        "bias":      bias,
                        "rmse":      rmse,
                        "mean_est":  float(np.mean(arr)),
                        "std_est":   float(np.std(arr)),
                        "frac_neg":  frac_neg,
                        "frac_gt1":  frac_gt1,
                        "raw_vals":  vals,   # kept for box-plots
                    }
                )

    return pd.DataFrame(records)


# ─── PLOTTING ─────────────────────────────────────────────────────────────────


def _add_legend(fig, estimator_keys=None):
    if estimator_keys is None:
        estimator_keys = ESTIMATOR_KEYS
    handles = [
        plt.Line2D(
            [0], [0],
            color=ESTIMATOR_COLORS[e],
            linestyle=ESTIMATOR_STYLES[e],
            linewidth=2,
            marker="o", markersize=5,
            label=ESTIMATOR_LABELS[e],
        )
        for e in estimator_keys
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(estimator_keys),
               bbox_to_anchor=(0.5, 0.0))


def plot_rmse(results: pd.DataFrame, save_path: str | None = None):
    families = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]
    dep_levels = ["low", "moderate", "strong"]

    fig, axes = plt.subplots(
        3, 4,
        figsize=(7.5, 7.3),
        sharex=True,
        constrained_layout=False,
    )

    fig.suptitle(
        r"RMSE of $\xi$ estimators by copula family and dependence level",
        y=0.985,
    )

    for col, family in enumerate(families):
        for row, dep in enumerate(dep_levels):
            ax = axes[row, col]
            param = next(
                p for (f, p), d in DEP_LEVEL.items() if f == family and d == dep
            )
            sub = results[(results["family"] == family) & (results["param"] == param)]
            true_xi = sub["true_xi"].iloc[0]

            for est in ESTIMATOR_KEYS:
                esub = sub[sub["estimator"] == est].sort_values("n")
                ax.plot(
                    esub["n"], esub["rmse"],
                    color=ESTIMATOR_COLORS[est],
                    linestyle=ESTIMATOR_STYLES[est],
                    linewidth=1.6,
                    marker="o",
                    markersize=4.2,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

            # Short two-line title prevents horizontal collisions.
            ax.set_title(
                f"{FAMILY_DISPLAY[family]}\n{dep}, $\\xi\\approx {true_xi:.2f}$",
                fontsize=10,
                pad=4,
            )

            if col == 0:
                ax.set_ylabel("RMSE")
            else:
                ax.set_ylabel("")

            if row == 2:
                ax.set_xlabel("$n$")
            else:
                ax.set_xlabel("")

            ax.tick_params(axis="both", which="major", labelsize=9)

    _add_legend(fig)

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.13,
        top=0.90,
        wspace=0.42,
        hspace=0.42,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

def plot_bias(results: pd.DataFrame, save_path: str | None = None):
    families = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]
    dep_levels = ["low", "moderate", "strong"]
    fig, axes = plt.subplots(3, 4, figsize=(7.5, 7.0), sharex=True)
    fig.suptitle(r"Bias of $\xi$ estimators by copula family and dependence level")

    for col, family in enumerate(families):
        for row, dep in enumerate(dep_levels):
            ax = axes[row, col]
            param = next(
                p for (f, p), d in DEP_LEVEL.items() if f == family and d == dep
            )
            sub = results[(results["family"] == family) & (results["param"] == param)]
            true_xi = sub["true_xi"].iloc[0]
            for est in ESTIMATOR_KEYS:
                esub = sub[sub["estimator"] == est].sort_values("n")
                ax.plot(
                    esub["n"], esub["bias"],
                    color=ESTIMATOR_COLORS[est],
                    linestyle=ESTIMATOR_STYLES[est],
                    linewidth=1.8, marker="o", markersize=5,
                )
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            title = f"{FAMILY_DISPLAY[family]}, {dep} ($\\xi \\approx {true_xi:.2f}$)"
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel("Bias")
            if row == 2:
                ax.set_xlabel("$n$")

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_negative_fraction(results: pd.DataFrame, save_path: str | None = None):
    """Fraction of estimates < 0 (and > 1) for low-dependence settings."""
    low = results[results["dep_level"] == "low"]
    families = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]
    fam_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharey=True)
    for ax, est, title in zip(
        axes,
        ["check_pi", "check_avg"],
        [r"CheckPi ($\underline{\xi}_n^\kappa$)", r"CheckAvg ($\xi_n^\kappa$)"],
    ):
        for fi, (family, col) in enumerate(zip(families, fam_colors)):
            sub = low[(low["family"] == family) & (low["estimator"] == est)].sort_values("n")
            ax.plot(
                sub["n"], sub["frac_neg"] * 100,
                color=col, linewidth=1.8, marker="o", markersize=6,
                label=FAMILY_DISPLAY[family],
            )
        ax.set_xscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel(r"Fraction of $\hat\xi < 0$ (\%)")
    fig.suptitle(r"Proportion of negative estimates at low-dependence settings")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_boxplots(results: pd.DataFrame, save_path: str | None = None):
    """Box plots comparing CheckAvg and NN for selected families and sample sizes."""
    sel_families = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]
    sel_ns = [100, 500, 2000]
    sel_dep = "moderate"
    sel_ests = ["check_avg", "nn"]

    fig, axes = plt.subplots(
        len(sel_families), len(sel_ns),
        figsize=(7.5, 8.5),
        sharey="row",
    )
    fig.suptitle(
        r"Distribution of $\xi$ estimates: $\xi_n^\kappa$ vs. $\xi_n$ (moderate dependence)"
    )

    for row, family in enumerate(sel_families):
        param = next(
            p for (f, p), d in DEP_LEVEL.items() if f == family and d == sel_dep
        )
        sub = results[(results["family"] == family) & (results["param"] == param)]
        true_xi = sub["true_xi"].iloc[0]

        for col, n in enumerate(sel_ns):
            ax = axes[row, col]
            nsub = sub[sub["n"] == n]
            bp_data = [nsub[nsub["estimator"] == e]["raw_vals"].iloc[0] for e in sel_ests]
            bp = ax.boxplot(
                bp_data,
                patch_artist=True,
                medianprops={"linewidth": 2},
                widths=0.5,
            )
            for patch, est in zip(bp["boxes"], sel_ests):
                patch.set_facecolor(ESTIMATOR_COLORS[est])
                patch.set_alpha(0.6)
            ax.axhline(true_xi, color="red", linewidth=1.5, linestyle="--", label=r"True $\xi$")
            ax.set_xticks([1, 2])
            ax.set_xticklabels([r"$\xi_n^\kappa$", r"$\xi_n$"])
            ax.grid(True, axis="y", alpha=0.3)
            if col == 0:
                ax.set_ylabel(FAMILY_DISPLAY[family])
            if row == 0:
                ax.set_title(f"$n = {n}$")
            if row == 0 and col == len(sel_ns) - 1:
                ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


# ─── LATEX TABLE ──────────────────────────────────────────────────────────────


def _holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm--Bonferroni step-down adjustment of a dict of raw p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    prev = 0.0
    for rank, (key, p) in enumerate(items):
        p_adj = min(1.0, p * (m - rank))
        p_adj = max(p_adj, prev)   # enforce monotonicity
        adj[key] = p_adj
        prev = p_adj
    return adj


def _summary_significance(
    n_sub: pd.DataFrame,
    families: list[str],
    est_order: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], str, dict[str, float]]:
    """Per (dep, n) row: mean Bias / mean RMSE / MC SE of RMSE across families,
    plus the identifier of the best estimator and Holm-adjusted one-sided
    p-values from the paired t-test of (squared error vs best, pooled across
    families, paired by (family, rep))."""
    mean_bias = {e: 0.0 for e in est_order}
    mean_rmse = {e: 0.0 for e in est_order}
    se_rmse_sq = {e: 0.0 for e in est_order}        # sum of per-family SE^2
    sq_err_pool: dict[str, list[float]] = {e: [] for e in est_order}
    n_fam = 0

    for fam in families:
        fam_n = n_sub[n_sub["family"] == fam]
        if len(fam_n) == 0:
            continue
        n_fam += 1
        true_xi = fam_n["true_xi"].iloc[0]
        for est in est_order:
            row = fam_n[fam_n["estimator"] == est].iloc[0]
            vals = np.asarray(row["raw_vals"], dtype=float)
            sq = (vals - true_xi) ** 2
            mse_f = float(np.mean(sq))
            rmse_f = float(np.sqrt(mse_f))
            # Delta-method MC SE for RMSE = sqrt(MSE):
            #   SE(RMSE) ≈ SE(MSE) / (2 * RMSE) = sd(sq) / (2 * RMSE * sqrt(N))
            if rmse_f > 0:
                se_rmse_f = float(np.std(sq, ddof=1) / (2 * rmse_f * np.sqrt(len(sq))))
            else:
                se_rmse_f = 0.0
            mean_bias[est] += float(row["bias"])
            mean_rmse[est] += rmse_f
            se_rmse_sq[est] += se_rmse_f ** 2
            sq_err_pool[est].extend(sq.tolist())

    if n_fam == 0:
        empty = {e: float("nan") for e in est_order}
        return empty, empty, empty, est_order[0], {e: 1.0 for e in est_order}

    for est in est_order:
        mean_bias[est] /= n_fam
        mean_rmse[est] /= n_fam
    # SE of the mean across n_fam independent families
    se_mean_rmse = {e: float(np.sqrt(se_rmse_sq[e]) / n_fam) for e in est_order}

    best = min(est_order, key=lambda e: mean_rmse[e])
    best_sq = np.asarray(sq_err_pool[best])

    raw_p: dict[str, float] = {}
    for est in est_order:
        if est == best:
            continue
        diff = np.asarray(sq_err_pool[est]) - best_sq
        # one-sided: is `est` significantly worse (larger MSE) than `best`?
        res = stats.ttest_1samp(diff, 0.0, alternative="greater")
        raw_p[est] = float(res.pvalue)
    adj_p = _holm_adjust(raw_p) if raw_p else {}
    adj_p[best] = 0.0
    return mean_bias, mean_rmse, se_mean_rmse, best, adj_p


def _stars(p: float) -> str:
    """Regression-table style significance stars."""
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return r"^{***}"
    if p < 0.05:
        return r"^{**}"
    if p < 0.10:
        return r"^{*}"
    return ""


def _rmse_summary_for_cell(
    n_sub: pd.DataFrame,
    families: list[str],
    est_order: list[str],
    baseline: str = "nn",
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute family-averaged RMSE, MC SE of averaged RMSE, and p-values.

    The p-values test whether each estimator has smaller squared error than
    the baseline estimator, using paired one-sided t-tests pooled over families.
    """
    mean_rmse = {e: 0.0 for e in est_order}
    se_rmse_sq = {e: 0.0 for e in est_order}
    sq_err_pool: dict[str, list[float]] = {e: [] for e in est_order}
    n_fam = 0

    for fam in families:
        fam_n = n_sub[n_sub["family"] == fam]
        if len(fam_n) == 0:
            continue

        n_fam += 1
        true_xi = float(fam_n["true_xi"].iloc[0])

        for est in est_order:
            row = fam_n[fam_n["estimator"] == est]
            if len(row) == 0:
                continue

            vals = np.asarray(row.iloc[0]["raw_vals"], dtype=float)
            sq = (vals - true_xi) ** 2
            mse_f = float(np.mean(sq))
            rmse_f = float(np.sqrt(mse_f))

            if rmse_f > 0:
                se_rmse_f = float(
                    np.std(sq, ddof=1) / (2.0 * rmse_f * np.sqrt(len(sq)))
                )
            else:
                se_rmse_f = 0.0

            mean_rmse[est] += rmse_f
            se_rmse_sq[est] += se_rmse_f ** 2
            sq_err_pool[est].extend(sq.tolist())

    if n_fam == 0:
        nan_dict = {e: float("nan") for e in est_order}
        return nan_dict, nan_dict, nan_dict

    for est in est_order:
        mean_rmse[est] /= n_fam

    se_mean_rmse = {
        est: float(np.sqrt(se_rmse_sq[est]) / n_fam)
        for est in est_order
    }

    pvals = {e: 1.0 for e in est_order}
    if baseline in sq_err_pool and len(sq_err_pool[baseline]) > 0:
        baseline_sq = np.asarray(sq_err_pool[baseline], dtype=float)

        for est in est_order:
            if est == baseline or len(sq_err_pool[est]) == 0:
                continue

            est_sq = np.asarray(sq_err_pool[est], dtype=float)
            diff = baseline_sq - est_sq

            if np.allclose(diff, diff[0]):
                pvals[est] = 0.0 if diff[0] > 0 else 1.0
            else:
                res = stats.ttest_1samp(diff, 0.0, alternative="greater")
                pvals[est] = float(res.pvalue)

    return mean_rmse, se_mean_rmse, pvals

def _rmse_summary_for_cell(
    n_sub: pd.DataFrame,
    families: list[str],
    est_order: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """For one (dependence level, n) cell, compute family-averaged RMSE,
    MC SE of the family-averaged RMSE, and one-sided p-values testing whether
    each estimator has smaller squared error than NN.

    Stars therefore answer: is this estimator significantly better than NN?
    """
    mean_rmse = {e: 0.0 for e in est_order}
    se_rmse_sq = {e: 0.0 for e in est_order}
    sq_err_pool: dict[str, list[float]] = {e: [] for e in est_order}
    n_fam = 0

    for fam in families:
        fam_n = n_sub[n_sub["family"] == fam]
        if len(fam_n) == 0:
            continue
        n_fam += 1
        true_xi = fam_n["true_xi"].iloc[0]

        for est in est_order:
            row = fam_n[fam_n["estimator"] == est].iloc[0]
            vals = np.asarray(row["raw_vals"], dtype=float)
            sq = (vals - true_xi) ** 2
            rmse_f = float(np.sqrt(np.mean(sq)))

            if rmse_f > 0:
                se_rmse_f = float(
                    np.std(sq, ddof=1) / (2 * rmse_f * np.sqrt(len(sq)))
                )
            else:
                se_rmse_f = 0.0

            mean_rmse[est] += rmse_f
            se_rmse_sq[est] += se_rmse_f ** 2
            sq_err_pool[est].extend(sq.tolist())

    for est in est_order:
        mean_rmse[est] /= n_fam

    se_mean_rmse = {
        e: float(np.sqrt(se_rmse_sq[e]) / n_fam)
        for e in est_order
    }

    # Paired one-sided tests against NN:
    # H_A: estimator has smaller squared error than NN.
    pvals = {e: 1.0 for e in est_order}
    nn_sq = np.asarray(sq_err_pool["nn"], dtype=float)

    for est in est_order:
        if est == "nn":
            continue
        est_sq = np.asarray(sq_err_pool[est], dtype=float)
        diff = nn_sq - est_sq
        res = stats.ttest_1samp(diff, 0.0, alternative="greater")
        pvals[est] = float(res.pvalue)

    return mean_rmse, se_mean_rmse, pvals


def make_summary_table(
    results: pd.DataFrame,
    n_reps: int = N_REPS,
) -> str:
    """Generate a compact regression-style LaTeX table.

    Rows are (dependence level, sample size). Columns are estimators.
    Entries are family-averaged RMSEs; parentheses contain MC standard errors.
    Stars indicate significantly lower squared error than NN.
    """
    sel_ns = [100, 500, 1000, 5000]
    dep_levels = ["low", "moderate", "strong"]
    families = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]

    est_order = ["check_avg", "check_pi", "check_min", "nn"]
    est_short = {
        "check_avg": r"\textsc{CheckAvg}",
        "check_pi":  r"\textsc{CheckPi}",
        "check_min": r"\textsc{CheckMin}",
        "nn":        r"\textsc{NN}",
    }

    def compact_cell(value: float, stars: str = "") -> str:
        """Single-line RMSE table cell."""
        return rf"${value:.3f}{stars}$"

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Family-averaged RMSE of Chatterjee $\xi$ estimators with $\kappa=\tfrac{1}{3}$.}")
    lines.append(r"  \label{tab:simulation_summary}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{llcccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Dep. level & $n$ & "
        + " & ".join(est_short[e] for e in est_order)
        + r" \\"
    )
    lines.append(r"    \midrule")

    for di, dep in enumerate(dep_levels):
        for ni, n in enumerate(sel_ns):
            n_sub = results[(results["dep_level"] == dep) & (results["n"] == n)]
            mean_rmse, se_rmse, pvals = _rmse_summary_for_cell(
                n_sub=n_sub,
                families=families,
                est_order=est_order,
            )

            dep_label = dep.capitalize() if ni == 0 else ""
            row_cells = []
            for est in est_order:
                stars = "" if est == "nn" else _stars(pvals.get(est, 1.0))
                row_cells.append(compact_cell(mean_rmse[est], stars))

            lines.append(
                f"    {dep_label} & {n} & "
                + " & ".join(row_cells)
                + r" \\"
            )

        if di < len(dep_levels) - 1:
            lines.append(r"    \addlinespace[0.25em]")
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \begin{tablenotes}")
    lines.append(r"    \footnotesize")
    lines.append(
        rf"    \item Notes: Entries are RMSEs averaged over the Gaussian, Clayton, "
        rf"Gumbel--Hougaard, and Frank copula families. Parentheses report Monte Carlo "
        rf"standard errors of the family-averaged RMSE. Each cell is based on "
        rf"{n_reps} Monte Carlo replications per copula family."
    )
    lines.append(
        r"    Stars indicate significantly smaller squared error than "
        r"\textsc{NN}, based on paired one-sided $t$-tests pooled over families: "
        r"$^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$."
    )
    lines.append(r"  \end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def make_latex_table(results: pd.DataFrame, n_reps: int = N_REPS) -> str:
    sel_ns = [100, 500, 1000, 5000]
    # display order for estimators
    est_order = ["check_avg", "nn", "check_pi", "check_min"]
    est_short = {
        "check_avg": r"\textsc{CheckAvg}",
        "nn":        r"\textsc{NN}",
        "check_pi":  r"\textsc{CheckPi}",
        "check_min": r"\textsc{CheckMin}",
    }

    # column spec: family | dep | n | (Bias RMSE) x4
    ncols_data = len(est_order) * 2
    col_spec = "llr" + "rr" * len(est_order)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")
    # header row 1: estimator names spanning 2 cols each
    est_header_cells = " & ".join(
        r"\multicolumn{2}{c}{" + est_short[e] + "}" for e in est_order
    )
    lines.append(r"    Family & Dep.\ & $n$ & " + est_header_cells + r" \\")
    # header row 2: Bias/RMSE labels
    br_cells = " & ".join("Bias & RMSE" for _ in est_order)
    lines.append(r"    & & & " + br_cells + r" \\")
    lines.append(r"    \midrule")

    families = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]
    for fi, family in enumerate(families):
        fam_sub = results[results["family"] == family]
        params_sorted = sorted(fam_sub["param"].unique())
        for pi, param in enumerate(params_sorted):
            dep = DEP_LEVEL.get((family, param), "")
            true_xi = fam_sub[fam_sub["param"] == param]["true_xi"].iloc[0]
            if family == "Gaussian":
                param_str = rf"$\rho={param}$"
            elif family == "GumbelHougaard":
                param_str = rf"$\vartheta={param:.1f}$"
            else:
                param_str = rf"$\vartheta={param:.1f}$"
            fam_label = FAMILY_DISPLAY[family] if pi == 0 else ""

            for ni, n in enumerate(sel_ns):
                sub = fam_sub[(fam_sub["param"] == param) & (fam_sub["n"] == n)]
                if ni == 0:
                    row_start = f"    {fam_label} & {param_str} & {n}"
                else:
                    row_start = f"    & & {n}"

                cells = []
                for est in est_order:
                    row = sub[sub["estimator"] == est]
                    if len(row) == 0:
                        cells += [r"\multicolumn{2}{c}{--}"]
                    else:
                        bias = row["bias"].values[0]
                        rmse = row["rmse"].values[0]
                        cells += [f"${bias:+.3f}$", f"${rmse:.3f}$"]
                lines.append(row_start + " & " + " & ".join(cells) + r" \\")

            if pi < len(params_sorted) - 1:
                lines.append(r"    \cmidrule{2-" + str(3 + ncols_data) + "}")
        lines.append(r"    \midrule")

    # replace last \midrule with \bottomrule
    lines[-1] = r"    \bottomrule"

    lines.append(r"  \end{tabular}")
    lines.append(r"  \caption{%")
    lines.append(
        r"  Finite-sample performance of Chatterjee $\xi$ estimators with "
        r"$\kappa=\tfrac{1}{3}$"
    )
    lines.append(
        r"  across four copula families (Gaussian, Clayton, Gumbel--Hougaard, Frank),"
    )
    lines.append(
        r"  three dependence levels, and four sample sizes ($n \in \{100,500,1000,5000\}$)."
    )
    lines.append(
        rf"  Bias and RMSE are based on {n_reps} Monte Carlo replications."
    )
    lines.append(
        r"  \textsc{CheckAvg} is the estimator $\xi_n^\kappa$;"
    )
    lines.append(
        r"  \textsc{NN} is the Azadkia--Chatterjee nearest-neighbour estimator;"
    )
    lines.append(
        r"  \textsc{CheckPi} and \textsc{CheckMin} are the lower and upper bounds"
    )
    lines.append(r"  $\underline{\xi}_n^\kappa$ and $\overline{\xi}_n^\kappa$, respectively.}")
    lines.append(r"  \label{tab:simulation_study}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ─── SUMMARY PRINTOUT ─────────────────────────────────────────────────────────


def print_summary(results: pd.DataFrame):
    """Print a concise console summary of key findings."""
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)

    # RMSE at n=1000 for moderate dependence
    mod = results[results["dep_level"] == "moderate"]
    n1k = mod[mod["n"] == 1000].copy()
    print("\nRMSE at n=1000, moderate dependence:")
    pivot = n1k.pivot_table(index=["family", "param"], columns="estimator", values="rmse")
    print(pivot[ESTIMATOR_KEYS].round(4).to_string())

    # Fraction of negative estimates at low dep, n=200
    low = results[(results["dep_level"] == "low") & (results["n"] == 200)]
    print("\nFraction of negative estimates at n=200, low dependence:")
    fneg = low.pivot_table(index="family", columns="estimator", values="frac_neg")
    print((fneg[["check_pi", "check_avg"]] * 100).round(1).to_string())

    # Bias at n=100
    print("\nBias at n=100:")
    n100 = results[results["n"] == 100].pivot_table(
        index=["family", "dep_level", "param"], columns="estimator", values="bias"
    )
    print(n100[ESTIMATOR_KEYS].round(4).to_string())

    # Significance flags for the summary table
    print("\nSummary-table significance flags (best in CAPS, * = not sig. worse):")
    print(f"{'dep':>9} {'n':>5}  " + "  ".join(f"{e:>15}" for e in ESTIMATOR_KEYS))
    families = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]
    for dep in ["low", "moderate", "strong"]:
        dep_sub = results[results["dep_level"] == dep]
        for n in [100, 500, 1000, 5000]:
            n_sub = dep_sub[dep_sub["n"] == n]
            _, mean_rmse, se_rmse, best, adj_p = _summary_significance(
                n_sub, families, ESTIMATOR_KEYS
            )
            cells = []
            for e in ESTIMATOR_KEYS:
                marker = ""
                if e == best:
                    body = f"{mean_rmse[e]:.4f}".upper()  # placeholder
                    cells.append(f"[{mean_rmse[e]:.4f}±{se_rmse[e]:.4f}]")
                elif adj_p.get(e, 1.0) > 0.05:
                    cells.append(f" {mean_rmse[e]:.4f}±{se_rmse[e]:.4f}*")
                else:
                    cells.append(f" {mean_rmse[e]:.4f}±{se_rmse[e]:.4f} ")
            print(f"{dep:>9} {n:>5}  " + "  ".join(f"{c:>15}" for c in cells))

    print("=" * 70)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(out_dir, exist_ok=True)

    calibration_path = os.path.join(out_dir, "simulation_calibrated_design.csv")
    SETTINGS, DEP_LEVEL = calibrate_design(cache_path=calibration_path)

    results = run_simulation()

    # drop raw_vals for CSV (not serialisable as string easily)
    results_save = results.drop(columns=["raw_vals"])
    csv_path = os.path.join(out_dir, "simulation_results.csv")
    results_save.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}  ({len(results_save)} rows)")

    plot_rmse(results, save_path=os.path.join(out_dir, "simulation_rmse.png"))
    plot_bias(results, save_path=os.path.join(out_dir, "simulation_bias.png"))
    plot_negative_fraction(
        results, save_path=os.path.join(out_dir, "simulation_neg_fraction.png")
    )
    plot_boxplots(results, save_path=os.path.join(out_dir, "simulation_boxplots.png"))

    latex = make_latex_table(results)
    tex_path = os.path.join(out_dir, "simulation_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"LaTeX table saved to {tex_path}")

    summary = make_summary_table(results)
    summary_path = os.path.join(out_dir, "simulation_summary.tex")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Summary table saved to {summary_path}")

    print_summary(results)
