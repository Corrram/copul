"""Regenerate all plots from the saved CSV without re-running the full simulation."""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import copul
from copul.chatterjee import xi_ncalculate

# ── paths ──────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "images")
CSV     = os.path.join(OUT_DIR, "simulation_results.csv")

results = pd.read_csv(CSV)

# ── import helpers from main module ────────────────────────────────────────────
import sys
sys.path.insert(0, HERE)
from simulation_study_xi_estimator import (
    ESTIMATOR_KEYS, ESTIMATOR_LABELS, ESTIMATOR_COLORS, ESTIMATOR_STYLES,
    FAMILY_DISPLAY, DEP_LEVEL,
    plot_rmse, plot_bias, plot_negative_fraction, make_latex_table,
    KAPPA, SEED, N_REPS,
)

# ── line plots (use saved CSV directly, no raw_vals needed) ────────────────────
plot_rmse(results, save_path=os.path.join(OUT_DIR, "simulation_rmse.png"))
plot_bias(results, save_path=os.path.join(OUT_DIR, "simulation_bias.png"))
plot_negative_fraction(results, save_path=os.path.join(OUT_DIR, "simulation_neg_fraction.png"))

# ── box plots (collect raw data only for moderate dependence) ──────────────────

def rep_seed(base, fi, ni, rep):
    return (base + fi * 1_000_000 + ni * 100 + rep) % (2**31)

def make_cop(family, param):
    if family == "Gaussian":       return copul.Gaussian(param)
    if family == "Clayton":        return copul.Clayton(param)
    if family == "GumbelHougaard": return copul.GumbelHougaard(param)
    return copul.Frank(param)

def estimate(data):
    df = pd.DataFrame(data, columns=["X","Y"])
    cp = copul.BivCheckPi.from_data(df, kappa=KAPPA)
    cm = copul.BivCheckMin.from_data(df, kappa=KAPPA)
    xi_pi  = float(cp.chatterjees_xi())
    xi_min = float(cm.chatterjees_xi())
    return {"check_avg": (xi_pi+xi_min)/2,
            "nn":        float(xi_ncalculate(data[:,0], data[:,1]))}

SEL_FAMILIES = ["Gaussian", "Clayton", "GumbelHougaard", "Frank"]
SEL_PARAMS   = {"Gaussian": 0.70, "Clayton": 2.00,
                "GumbelHougaard": 3.00, "Frank": 6.00}
SEL_NS       = [100, 500, 2000]

# raw[family][n][estimator] = list of floats
print("Collecting raw values for box plots (moderate dependence) ...")
raw: dict = {}
for fi, family in enumerate(SEL_FAMILIES):
    param = SEL_PARAMS[family]
    cop   = make_cop(family, param)
    raw[family] = {}
    for ni, n in enumerate(SEL_NS):
        vals = {"check_avg": [], "nn": []}
        for rep in range(N_REPS):
            s    = rep_seed(SEED, fi, ni, rep)
            data = cop.rvs(n, random_state=s)
            est  = estimate(data)
            vals["check_avg"].append(est["check_avg"])
            vals["nn"].append(est["nn"])
        raw[family][n] = vals

# build true_xi lookup
true_xi_map = {}
for family in SEL_FAMILIES:
    param = SEL_PARAMS[family]
    sub = results[(results["family"] == family) & (results["param"] == param)]
    true_xi_map[family] = float(sub["true_xi"].iloc[0])

# draw box plots
fig, axes = plt.subplots(len(SEL_FAMILIES), len(SEL_NS), figsize=(12, 10), sharey="row")
fig.suptitle(
    r"Distribution of $\xi$ estimates: CheckAvg vs. $\xi_n$ (moderate dependence)",
    fontsize=12,
)
for row, family in enumerate(SEL_FAMILIES):
    true_xi = true_xi_map[family]
    for col, n in enumerate(SEL_NS):
        ax = axes[row, col]
        data_bp = [raw[family][n]["check_avg"], raw[family][n]["nn"]]
        bp = ax.boxplot(data_bp, patch_artist=True,
                        medianprops={"linewidth": 2}, widths=0.5)
        colors = [ESTIMATOR_COLORS["check_avg"], ESTIMATOR_COLORS["nn"]]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.axhline(true_xi, color="red", linewidth=1.5, linestyle="--")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["CheckAvg", "NN"], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        if col == 0:
            ax.set_ylabel(FAMILY_DISPLAY[family], fontsize=10)
        if row == 0:
            ax.set_title(f"$n = {n}$", fontsize=10)
        if row == 0 and col == len(SEL_NS) - 1:
            ax.plot([], [], color="red", linestyle="--", linewidth=1.5, label="True $\\xi$")
            ax.legend(fontsize=8, loc="upper right")

plt.tight_layout()
bp_path = os.path.join(OUT_DIR, "simulation_boxplots.png")
plt.savefig(bp_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {bp_path}")

# ── LaTeX table ────────────────────────────────────────────────────────────────
latex = make_latex_table(results)
tex_path = os.path.join(OUT_DIR, "simulation_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"LaTeX table saved to {tex_path}")
print("All done.")
