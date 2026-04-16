import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.signal import savgol_filter
from plot_utils import (
    CorrelationData, find_data_dir, load_family_data,
    get_gaussian_xi_rho_tau, get_cb_xi_rho_tau, get_marshall_olkin_a1eq1_xi_rho_tau,
)


# ── Configuration ─────────────────────────────────────────────────────────────

COLORS = {
    "C_b":          "blue",
    "Gaussian":     "black",
    "MarshallOlkin": "#17becf",
    "Clayton":      "#d62728",
    "Frank":        "#2ca02c",
    "GumbelHougaard": "#ff7f0e",
    "Joe":          "#9467bd",
}
LABELS = {
    "GumbelHougaard": "Gumbel",
    "C_b":           r"$(C_b)_{b>0}$",
    "MarshallOlkin": r"Marshall-Olkin ($\alpha_1=1$)",
}
FAMILIES = ["Clayton", "Frank", "GumbelHougaard", "Joe", "MarshallOlkin"]


# ── Smoothed plot helper ──────────────────────────────────────────────────────

def _plot_diff(ax, xi, rho, tau, name, smooth=True):
    if xi is None or len(xi) == 0:
        return
    idx = np.argsort(xi)
    x   = xi[idx]
    y_rho = rho[idx] - x
    y_tau = tau[idx] - x if tau is not None else None

    if smooth and len(x) > 21:
        try:
            y_rho = savgol_filter(y_rho, 21, 3)
            if y_tau is not None and not np.all(np.isnan(y_tau)):
                y_tau = savgol_filter(y_tau, 21, 3)
        except Exception:
            pass

    c   = COLORS.get(name, "gray")
    lbl = LABELS.get(name, name)
    ax.plot(x, y_rho, color=c, lw=2, label=lbl)
    if y_tau is not None and not np.all(np.isnan(y_tau)):
        ax.plot(x, y_tau, color=c, lw=2, ls="--")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = find_data_dir(__file__)

    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # Analytic curves
    for get_fn, name in [
        (get_cb_xi_rho_tau,                    "C_b"),
        (get_gaussian_xi_rho_tau,              "Gaussian"),
        (get_marshall_olkin_a1eq1_xi_rho_tau,  "MarshallOlkin"),
    ]:
        xi, rho, tau = get_fn()
        _plot_diff(ax, xi, rho, tau, name, smooth=False)

    # Empirical families
    for fam in FAMILIES:
        xi, rho, tau = load_family_data(
            fam, data_dir,
            key1="chatterjees_xi", key2="spearmans_rho",
            extra_keys=["kendalls_tau"],
            filter_fn=lambda x, r: r > 0.001,
        )
        _plot_diff(ax, xi, rho, tau, fam, smooth=True)

    plt.xlim(0, 1)
    plt.ylim(0, 0.45)
    plt.xlabel(r"$\xi$", fontsize=12)
    plt.ylabel(r"$\rho - \xi$ (solid)  and  $\tau - \xi$ (dashed)", fontsize=12)
    plt.title(r"Differences between rank correlations", fontsize=14)

    # Two-legend setup: family colours + line-style guide
    from matplotlib.lines import Line2D
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    leg1 = plt.legend(by_label.values(), by_label.keys(),
                      loc="upper right", frameon=True, fontsize=10, ncol=2)
    ax.add_artist(leg1)
    plt.legend(
        handles=[
            Line2D([0], [0], color="gray", lw=2, ls="-",  label=r"$\rho - \xi$"),
            Line2D([0], [0], color="gray", lw=2, ls="--", label=r"$\tau - \xi$"),
        ],
        loc="upper center", fontsize=10, frameon=False,
    )

    plt.grid(True, ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("rho_tau_minus_xi_plot_smoothed.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
