import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, UnivariateSpline
from plot_utils import (
    CorrelationData, find_data_dir, load_family_data,
    get_gaussian_xi_rho_tau, get_marshall_olkin_a1eq1_xi_rho_tau,
)


# ── Analytic curves specific to (ξ, β) ───────────────────────────────────────

def get_gaussian_xi_beta(n_points: int = 300):
    """Gaussian copula: (ξ, β) as a function of the Pearson-r parameter."""
    r    = np.linspace(0, 1.0, n_points)
    beta = (2/np.pi) * np.arcsin(r)
    xi   = (3/np.pi) * np.arcsin((1 + r**2) / 2) - 0.5
    return xi, beta

def get_boundary_checkerboard_xi_beta(n_points: int = 1000):
    """Symmetric 2×2 checkerboard copulas C_a^#: exact lower boundary xi = beta²/2."""
    beta = np.linspace(0, 1.0, n_points)
    return (beta**2) / 2, beta

def get_marshall_olkin_a1eq1_xi_beta(n_points: int = 1000):
    """Marshall-Olkin with α₁ = 1 fixed, α₂ ∈ [0,1]: (ξ, β)."""
    a2   = np.linspace(0, 1, n_points)
    beta = 2**a2 - 1          # = 4·C(½,½)−1 with C(½,½) = (½)^(2−a2)
    xi   = 2*a2 / (3 - a2)
    return xi, beta


# ── Configuration ─────────────────────────────────────────────────────────────

COLORS = {
    "Boundary":       "blue",
    "Gaussian":       "black",
    "MO_a1":          "#17becf",
    "BivClayton":     "#d62728",
    "Clayton":        "#d62728",
    "Frank":          "#2ca02c",
    "GumbelHougaard": "#ff7f0e",
    "Joe":            "#9467bd",
}
LABELS = {
    "GumbelHougaard": "Gumbel",
    "Boundary":       r"$C_a^\#$ (Boundary)",
    "MO_a1":          r"Marshall-Olkin ($\alpha_1=1$)",
    "BivClayton":     "Clayton",
}
FAMILIES = ["BivClayton", "Frank", "GumbelHougaard", "Joe"]


def _sorted_handles_labels(ax):
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    items = sorted(by_label.items(),
                   key=lambda t: (0 if "Boundary" in t[0] else 1, t[0]))
    if not items:
        return [], []
    lbls_s, h_s = zip(*items)
    return list(h_s), list(lbls_s)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = find_data_dir(__file__)

    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 15, "axes.titlesize": 16,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 13,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5),
                                   layout="constrained",
                                   gridspec_kw={"width_ratios": [1, 1.2]})

    BLUE_ENV, FILL_ENV = "#00529B", "#D6EAF8"

    # Exact envelope: ξ ≥ β²/2
    beta_env     = np.linspace(0, 1.0, 300)
    xi_env_lower = beta_env**2 / 2
    ax1.plot(beta_env, xi_env_lower, color=BLUE_ENV, lw=2.5)
    ax1.plot(beta_env, np.ones_like(beta_env), color=BLUE_ENV, lw=2.5)
    ax1.plot([0, 0], [0, 1], color=BLUE_ENV, lw=2.5)
    ax1.plot([1, 1], [0.5, 1], color=BLUE_ENV, lw=2.5)
    ax1.fill_between(beta_env, xi_env_lower, 1, color=FILL_ENV, alpha=0.5, zorder=0)
    ax1.plot([0, 1], [0, 1], color="gray", ls="-", lw=1.5, alpha=0.4)

    # ── Left: analytic + empirical families ──────────────────────────────────
    def plot_left(xi_arr, beta_arr, name, smooth=True):
        if xi_arr is None or len(xi_arr) == 0:
            return
        idx = np.argsort(beta_arr)
        b, x = beta_arr[idx], xi_arr[idx]
        if smooth and len(b) > 50:
            try:
                x = savgol_filter(x, 31, 3)
                cs = CubicSpline(b, x)
                b  = np.linspace(b[0], b[-1], 500)
                x  = np.clip(cs(b), 0.0, 1.0)
            except Exception:
                pass
        c, lbl = COLORS.get(name, "gray"), LABELS.get(name, name)
        ax1.plot(b, x, color=c, lw=2.5, ls="-", label=lbl, alpha=0.9)

    xi_g,     beta_g     = get_gaussian_xi_beta()
    xi_mo,    beta_mo    = get_marshall_olkin_a1eq1_xi_beta()
    xi_bound, beta_bound = get_boundary_checkerboard_xi_beta()
    plot_left(xi_g,     beta_g,     "Gaussian",  smooth=False)
    plot_left(xi_mo,    beta_mo,    "MO_a1",     smooth=False)
    plot_left(xi_bound, beta_bound, "Boundary",  smooth=False)
    for fam in FAMILIES:
        xi_e, beta_e = load_family_data(
            fam, data_dir,
            key1="chatterjees_xi", key2="blomqvist_beta",
            filter_fn=lambda x, b: (b > 0.001) & (b <= 1.0) & (x >= 0.0) & (x <= 1.0),
        )
        plot_left(xi_e, beta_e, fam)

    ax1.set_xlabel(r"Blomqvist's $\beta$")
    ax1.set_ylabel(r"Chatterjee's $\xi$")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_aspect("equal")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.set_title(r"Families within the exact $(\beta, \xi)$-region", pad=15)
    h1, l1 = _sorted_handles_labels(ax1)
    ax1.legend(h1, l1, loc="upper left", framealpha=0.9, edgecolor="none")

    # ── Right: β − ξ differences ─────────────────────────────────────────────
    def plot_right(xi_arr, beta_arr, name, smooth=True):
        if xi_arr is None or len(xi_arr) == 0:
            return
        idx   = np.argsort(xi_arr)
        x     = xi_arr[idx]
        y     = beta_arr[idx] - x
        if smooth and len(x) > 50:
            try:
                spl = UnivariateSpline(x, y, s=len(x)*8e-4, k=4)
                x   = np.linspace(x[0], x[-1], 500)
                y   = spl(x)
            except Exception:
                pass
        c, lbl = COLORS.get(name, "gray"), LABELS.get(name, name)
        ax2.plot(x, y, color=c, lw=2.5, ls="-", label=lbl, alpha=0.9)

    plot_right(xi_g,     beta_g,     "Gaussian",  smooth=False)
    plot_right(xi_mo,    beta_mo,    "MO_a1",     smooth=False)
    plot_right(xi_bound, beta_bound, "Boundary",  smooth=False)
    for fam in FAMILIES:
        xi_e, beta_e = load_family_data(
            fam, data_dir,
            key1="chatterjees_xi", key2="blomqvist_beta",
            filter_fn=lambda x, b: (b > 0.001) & (b <= 1.0) & (x >= 0.0) & (x <= 1.0),
        )
        plot_right(xi_e, beta_e, fam)

    ax2.set_xlabel(r"Chatterjee's $\xi$")
    ax2.set_ylabel(r"$\beta - \xi$")
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 0.52)
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.set_title(r"Rank correlation difference: $(\beta - \xi)$", pad=15)
    h2, l2 = _sorted_handles_labels(ax2)
    ax2.legend(h2, l2, loc="upper right", framealpha=0.9, edgecolor="none", ncol=2)

    Path("images/").mkdir(exist_ok=True)
    plt.savefig("images/combined_region_and_diffs_beta.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
