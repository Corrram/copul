import pickle
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
from typing import Dict
from pathlib import Path


# ------------------------------------------------------------------
# 1. Analytic Helper Functions (Bounds & Families for Beta vs Xi)
# ------------------------------------------------------------------


def get_gaussian_curve(n_points=300):
    """Analytic curve for the Gaussian copula."""
    r = np.linspace(0, 1.0, n_points)
    beta = (2 / np.pi) * np.arcsin(r)
    xi = (3 / np.pi) * np.arcsin((1 + r**2) / 2) - 0.5
    return xi, beta


def get_boundary_checkerboard_curve(n_points=1000):
    """
    Analytic curve for the symmetric 2x2 checkerboard copulas C_a^#.
    This traces the exact lower boundary of the attainable region.
    """
    beta = np.linspace(0, 1.0, n_points)
    xi = (beta**2) / 2
    return xi, beta


def get_marshall_olkin_alpha1_1_curve(n_points=1000):
    """
    Marshall-Olkin with fixed alpha_1 = 1, varying alpha_2 in [0, 1].
    """
    a2 = np.linspace(0, 1, n_points)
    # C(1/2, 1/2) = (1/2)^(2 - a2), hence beta = 4*C(1/2,1/2) - 1 = 2^a2 - 1
    beta = 2**a2 - 1
    xi = 2 * a2 / (3 - a2)
    return xi, beta


# ------------------------------------------------------------------
# 2. Data Loader
# ------------------------------------------------------------------


@dataclass
class CorrelationData:
    params: np.ndarray
    values: Dict[str, np.ndarray]


def load_family_data(family: str, data_dir: Path):
    candidates = [
        data_dir / f"{family}_data.pkl",
        data_dir / f"{family}.pkl",
        data_dir / f"{family} Copula_data.pkl",
    ]
    file_path = next((c for c in candidates if c.exists()), None)

    if not file_path:
        return None, None

    try:
        data = pickle.loads(file_path.read_bytes())
        xi = data.values.get("chatterjees_xi")
        beta = data.values.get("blomqvist_beta")

        if xi is None or beta is None:
            return None, None

        # Filter for positive dependence and valid values
        mask = np.isfinite(xi) & np.isfinite(beta) & (beta > 0.001)
        return xi[mask], beta[mask]
    except Exception as e:
        print(f"Error loading {family}: {e}")
        return None, None


# ------------------------------------------------------------------
# 3. Main Plotting
# ------------------------------------------------------------------


def main():
    # --- Data Import Logic ---
    data_dir = Path("rank_correlation_estimates")  # default
    found = False

    try:
        with pkg_resources.path("copul", "docs") as docs_path:
            possible_dir = docs_path / "rank_correlation_estimates"
            if possible_dir.exists():
                data_dir = possible_dir
                found = True
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    if not found:
        # Check relative path
        script_location = Path(__file__).parent
        relative_candidate = (
            script_location / "../../../docs/rank_correlation_estimates"
        )
        if relative_candidate.resolve().exists():
            data_dir = relative_candidate.resolve()
            found = True

    if found:
        print(f"Loading data from: {data_dir.resolve()}")
    else:
        print(
            "Warning: Could not find data directory. Empirical curves may be missing."
        )

    # --- Configuration ---
    colors = {
        "Boundary": "blue",
        "Gaussian": "black",
        "MO_a1": "#17becf",  # Cyan
        "BivClayton": "#d62728",
        "Clayton": "#d62728",
        "Frank": "#2ca02c",
        "GumbelHougaard": "#ff7f0e",
        "Joe": "#9467bd",
    }

    labels = {
        "GumbelHougaard": "Gumbel",
        "Boundary": r"$C_a^\#$ (Boundary)",
        "MO_a1": r"Marshall-Olkin ($\alpha_1=1$)",
        "BivClayton": "Clayton",
        "Clayton": "Clayton",
        "Frank": "Frank",
        "Joe": "Joe",
        "Gaussian": "Gaussian",
    }

    families = ["BivClayton", "Frank", "GumbelHougaard", "Joe"]

    # --- Global Plot Aesthetics ---
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # SINGLE PLOT LAYOUT
    fig, ax = plt.subplots(figsize=(7.5, 7.5), layout="constrained")

    # ----------------------------------------------------------
    # PLOT: Attainable Region
    # ----------------------------------------------------------
    BLUE_ENV = "#00529B"
    FILL_ENV = "#D6EAF8"

    # Exact Envelope for Beta (y^2 <= 2x -> xi >= beta^2 / 2)
    beta_env = np.linspace(0, 1.0, 300)
    xi_env_lower = (beta_env**2) / 2
    xi_env_upper = np.ones_like(beta_env)

    # Draw boundaries
    ax.plot(beta_env, xi_env_lower, color=BLUE_ENV, lw=2.5)
    ax.plot(beta_env, xi_env_upper, color=BLUE_ENV, lw=2.5)
    ax.plot([0, 0], [0, 1], color=BLUE_ENV, lw=2.5)  # y-axis closure
    ax.plot([1, 1], [0.5, 1], color=BLUE_ENV, lw=2.5)  # Right edge closure

    # Fill region
    ax.fill_between(
        beta_env, xi_env_lower, xi_env_upper, color=FILL_ENV, alpha=0.5, zorder=0
    )

    # Diagonal
    ax.plot([0, 1], [0, 1], color="gray", linestyle="-", lw=1.5, alpha=0.4)

    def plot_fam(xi_arr, beta_arr, name, smooth=True):
        if len(xi_arr) == 0:
            return

        # Sort and ensure strictly increasing X for spline fitting
        idx = np.argsort(beta_arr)
        b = beta_arr[idx]
        x = xi_arr[idx]
        b_uniq, u_idx = np.unique(b, return_index=True)
        x_uniq = x[u_idx]

        # Heavy Smoothing: Savitzky-Golay + Smoothing Spline
        if smooth and len(b_uniq) > 50:
            try:
                # 1. Wider window to cut out local noise
                w = min(101, len(b_uniq) - 1 if len(b_uniq) % 2 == 0 else len(b_uniq))
                if w % 2 == 0:
                    w += 1
                x_sg = savgol_filter(x_uniq, w, 2)

                # 2. UnivariateSpline with s > 0 forces a smooth trajectory
                spl = UnivariateSpline(b_uniq, x_sg, s=0.002)

                b_new = np.linspace(b_uniq.min(), b_uniq.max(), 500)
                x = spl(b_new)
                b = b_new
            except Exception:
                pass
        else:
            b = b_uniq
            x = x_uniq

        c = colors.get(name, "gray")
        lbl = labels.get(name, name)
        ax.plot(b, x, color=c, lw=2.5, linestyle="-", label=lbl, alpha=0.9)

    # Analytic (No smoothing to preserve end-points)
    xi_g, beta_g = get_gaussian_curve()
    plot_fam(xi_g, beta_g, "Gaussian", smooth=False)

    xi_mo1, beta_mo1 = get_marshall_olkin_alpha1_1_curve()
    plot_fam(xi_mo1, beta_mo1, "MO_a1", smooth=False)

    xi_bound, beta_bound = get_boundary_checkerboard_curve()
    plot_fam(xi_bound, beta_bound, "Boundary", smooth=False)

    # Empirical (Smoothing enabled)
    for fam in families:
        xi_e, beta_e = load_family_data(fam, data_dir)
        if xi_e is not None:
            plot_fam(xi_e, beta_e, fam, smooth=True)

    ax.set_xlabel(r"Blomqvist's $\beta$")
    ax.set_ylabel(r"Chatterjee's $\xi$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(r"Families within the exact $(\beta, \xi)$-region", pad=15)

    # --- Legend Sorting Helper ---
    def get_sorted_handles_labels(axes):
        handles, lbls = axes.get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))

        def sort_key(item):
            label = item[0]
            priority = 0 if "Boundary" in label else 1
            return (priority, label)

        sorted_items = sorted(by_label.items(), key=sort_key)
        if not sorted_items:
            return [], []

        labels_sorted, handles_sorted = zip(*sorted_items)
        return list(handles_sorted), list(labels_sorted)

    # Apply Legend
    handles, final_labels = get_sorted_handles_labels(ax)
    leg = ax.legend(
        handles, final_labels, loc="upper left", framealpha=0.9, edgecolor="none"
    )
    ax.add_artist(leg)

    # --- Save and Show ---
    Path("images/").mkdir(parents=False, exist_ok=True)
    plt.savefig("images/exact_region_beta_xi.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
