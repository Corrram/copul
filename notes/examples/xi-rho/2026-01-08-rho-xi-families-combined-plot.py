import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
from plot_utils import (
    CorrelationData, find_data_dir, load_family_data,
    get_gaussian_xi_rho_tau, get_cb_xi_rho_tau, get_marshall_olkin_a1eq1_xi_rho_tau,
)


# ── Upper-boundary helper (ρ-axis) ────────────────────────────────────────────

def _b_from_x_regime1(x: float) -> float:
    if np.isclose(x, 1.0): return np.inf
    if x <= 3/10: return 1.0 if np.isclose(x, 3/10) else np.nan
    numer = 5 + np.sqrt(5 * (6*x - 1))
    denom = 10 * (1 - x)
    return np.inf if np.isclose(denom, 0) else numer / denom

def _b_from_x_regime2(x: float) -> float:
    if np.isclose(x, 0): return 0.0
    if x > 3/10: return 1.0 if np.isclose(x, 3/10) else np.nan
    theta = (1/3) * np.arccos(np.clip(1 - (108/25)*x, -1.0, 1.0))
    return np.clip((5/6) + (5/3)*np.cos(theta - 2*np.pi/3), 0.0, 1.0)

def rho_upper_bound(xi: float) -> float:
    """Upper boundary ρ_max(ξ) for the attainable (ξ, ρ) region."""
    if not (0 <= xi <= 1): return np.nan
    if np.isclose(xi, 0): return 0.0
    if np.isclose(xi, 1): return 1.0
    x_thresh = 3/10
    if xi < x_thresh:
        b = _b_from_x_regime2(xi)
        return b - 3*b**2/10
    if xi > x_thresh:
        b = _b_from_x_regime1(xi)
        return 1.0 if np.isinf(b) else 1 - 1/(2*b**2) + 1/(5*b**3)
    return 1.0 - 3/10


# ── Configuration ─────────────────────────────────────────────────────────────

COLORS = {
    "C_b":            "blue",
    "Gaussian":       "black",
    "MO_a1":          "#17becf",
    "Clayton":        "#d62728",
    "Frank":          "#2ca02c",
    "GumbelHougaard": "#ff7f0e",
    "Joe":            "#9467bd",
}
LABELS = {
    "GumbelHougaard": "Gumbel",
    "C_b":   r"$(C_b)_{b>0}$",
    "MO_a1": r"Marshall-Olkin ($\alpha_1=1$)",
}
FAMILIES = ["Clayton", "Frank", "GumbelHougaard", "Joe"]


def _sorted_handles_labels(ax):
    """Deduplicate and sort legend entries, C_b / Boundary first."""
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    items = sorted(by_label.items(),
                   key=lambda t: (0 if "C_b" in t[0] or "Boundary" in t[0] else 1, t[0]))
    if not items:
        return [], []
    lbls_s, h_s = zip(*items)
    return list(h_s), list(lbls_s)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = find_data_dir(__file__)

    # Envelope data
    eps    = 1e-9
    xi_env = np.unique(np.clip(np.concatenate([
        np.linspace(0.0,        3/10 - eps, 150),
        np.linspace(3/10 - eps, 3/10 + eps, 50),
        np.linspace(3/10 + eps, 1.0,        150),
    ]), 0.0, 1.0))
    rho_env = np.array([rho_upper_bound(x) for x in xi_env])
    valid   = ~np.isnan(rho_env)
    xi_env, rho_env = xi_env[valid], rho_env[valid]

    BLUE_ENV, FILL_ENV = "#00529B", "#D6EAF8"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: attainable region ──────────────────────────────────────────────
    ax1.plot( rho_env, xi_env, color=BLUE_ENV, lw=2.5)
    ax1.plot(-rho_env, xi_env, color=BLUE_ENV, lw=2.5)
    ax1.fill_betweenx(xi_env, -rho_env, rho_env, color=FILL_ENV, alpha=0.5, zorder=0)
    ax1.plot([0, 1], [0, 1], color="gray", ls="-", lw=1, alpha=0.5)

    def plot_left(xi_arr, rho_arr, name, smooth=True):
        if xi_arr is None or len(xi_arr) == 0:
            return
        idx = np.argsort(rho_arr)
        r, x = rho_arr[idx], xi_arr[idx]
        if smooth and len(r) > 50:
            try:
                x = savgol_filter(x, 21, 3)
            except Exception:
                pass
        c, lbl = COLORS.get(name, "gray"), LABELS.get(name, name)
        ax1.plot(r, x,            color=c, lw=2.5, ls="-",  label=lbl)
        ax1.plot(r, np.sqrt(np.clip(x, 0, None)), color=c, lw=2.5, ls=":", alpha=0.8)

    xi_g,  rho_g,  _     = get_gaussian_xi_rho_tau(r_max=1.0)
    xi_mo, rho_mo, _     = get_marshall_olkin_a1eq1_xi_rho_tau()
    xi_cb, rho_cb, _     = get_cb_xi_rho_tau()
    plot_left(xi_g,  rho_g,  "Gaussian", smooth=False)
    plot_left(xi_mo, rho_mo, "MO_a1",   smooth=False)
    plot_left(xi_cb, rho_cb, "C_b",     smooth=False)
    for fam in FAMILIES:
        xi_e, rho_e, _ = load_family_data(
            fam, data_dir,
            key1="chatterjees_xi", key2="spearmans_rho",
            extra_keys=["kendalls_tau"],
            filter_fn=lambda x, r: r > 0.001,
        )
        plot_left(xi_e, rho_e, fam)

    ax1.set_xlabel(r"$\rho$", fontsize=14)
    ax1.set_ylabel(r"$\xi$ (solid)  and  $\sqrt{\xi}$ (dotted)", fontsize=14)
    ax1.set_xlim(-0.1, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect("equal")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.set_title("Copula families within the attainable region", fontsize=14)
    h1, l1 = _sorted_handles_labels(ax1)
    ax1.legend(h1, l1, loc="upper left", fontsize=14, framealpha=0.9)

    # ── Right: differences ───────────────────────────────────────────────────
    def plot_right(xi_arr, rho_arr, tau_arr, name, smooth=True):
        if xi_arr is None or len(xi_arr) == 0:
            return
        idx   = np.argsort(xi_arr)
        x     = xi_arr[idx]
        y_rho = rho_arr[idx] - x
        y_tau = tau_arr[idx] - x if tau_arr is not None else None

        if smooth and len(x) > 50:
            try:
                y_rho = savgol_filter(y_rho, 21, 3)
                if y_tau is not None and not np.all(np.isnan(y_tau)):
                    y_tau = savgol_filter(y_tau, 21, 3)
            except Exception:
                pass

        c, lbl = COLORS.get(name, "gray"), LABELS.get(name, name)
        ax2.plot(x, y_rho, color=c, lw=2.5, ls="-", label=lbl)
        if y_tau is not None and not np.all(np.isnan(y_tau)):
            ax2.plot(x, y_tau, color=c, lw=2.5, ls="--")

    xi_g,  rho_g,  tau_g  = get_gaussian_xi_rho_tau(r_max=1.0)
    xi_mo, rho_mo, tau_mo = get_marshall_olkin_a1eq1_xi_rho_tau()
    xi_cb, rho_cb, tau_cb = get_cb_xi_rho_tau()
    plot_right(xi_g,  rho_g,  tau_g,  "Gaussian", smooth=False)
    plot_right(xi_mo, rho_mo, tau_mo, "MO_a1",    smooth=False)
    plot_right(xi_cb, rho_cb, tau_cb, "C_b",      smooth=False)
    for fam in FAMILIES:
        xi_e, rho_e, tau_e = load_family_data(
            fam, data_dir,
            key1="chatterjees_xi", key2="spearmans_rho",
            extra_keys=["kendalls_tau"],
            filter_fn=lambda x, r: r > 0.001,
        )
        plot_right(xi_e, rho_e, tau_e, fam)

    ax2.set_xlabel(r"$\xi$", fontsize=14)
    ax2.set_ylabel(r"$\rho - \xi$ (solid) and  $\tau - \xi$ (dotted)", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.45)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.set_title(r"Rank correlation differences: $(\rho - \xi)$ and $(\tau - \xi)$", fontsize=14)
    h2, l2 = _sorted_handles_labels(ax2)
    ax2.legend(h2, l2, loc="upper right", fontsize=14, framealpha=0.9, ncol=2)

    Path("images/").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig("images/combined_region_and_diffs.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
