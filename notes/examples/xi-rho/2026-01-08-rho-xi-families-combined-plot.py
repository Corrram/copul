import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from pathlib import Path


# ------------------------------------------------------------------
# 1. Analytic Helper Functions (Bounds & Families)
# ------------------------------------------------------------------

def b_from_x_regime1(x_val: float) -> float:
    """b(x) for x in (3/10, 1] with b > 1."""
    if np.isclose(x_val, 1.0): return np.inf
    if x_val <= 3 / 10: return 1.0 if np.isclose(x_val, 3 / 10) else np.nan
    numer = 5 + np.sqrt(5 * (6 * x_val - 1))
    denom = 10 * (1 - x_val)
    return np.inf if np.isclose(denom, 0) else numer / denom


def b_from_x_regime2(x_val: float) -> float:
    """b(x) for x in (0, 3/10] with 0 < b <= 1."""
    if np.isclose(x_val, 0): return 0.0
    if x_val > 3 / 10: return 1.0 if np.isclose(x_val, 3 / 10) else np.nan
    theta = (1 / 3) * np.arccos(np.clip(1 - (108 / 25) * x_val, -1.0, 1.0))
    return np.clip((5 / 6) + (5 / 3) * np.cos(theta - 2 * np.pi / 3), 0.0, 1.0)


def M_x_upper_bound_corrected(x_val: float) -> float:
    """Corrected upper bound M_xi(rho)."""
    if x_val < 0 or x_val > 1: return np.nan
    if np.isclose(x_val, 0): return 0.0
    if np.isclose(x_val, 1): return 1.0

    x_thresh = 3 / 10
    if x_val < x_thresh and not np.isclose(x_val, x_thresh):
        b = b_from_x_regime2(x_val)
        return b - (3 * b ** 2) / 10
    if x_val > x_thresh and not np.isclose(x_val, x_thresh):
        b = b_from_x_regime1(x_val)
        return 1.0 if np.isinf(b) else 1 - 1 / (2 * b ** 2) + 1 / (5 * b ** 3)
    return 1.0 - (3 / 10)


def get_gaussian_curve(n_points=300):
    # Go all the way to 1.0 to ensure the curve closes at the corner
    r = np.linspace(0, 1.0, n_points)
    rho_s = (6 / np.pi) * np.arcsin(r / 2)
    tau = (2 / np.pi) * np.arcsin(r)
    xi = (3 / np.pi) * np.arcsin((1 + r ** 2) / 2) - 0.5
    return xi, rho_s, tau


def get_cb_curve(n_points=1000):
    # Focus on positive dependence part
    b_vals = np.linspace(0, 100, 2000)  # Extend b to ensure coverage near xi=1

    # Use errstate to ignore division by zero in the unused branch of np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        def vec_xi(b):
            return np.where(b <= 1, (b ** 2 / 10) * (5 - 2 * b), 1 - 1 / b + 3 / (10 * b ** 2))

        def vec_rho(b):
            return np.where(b <= 1, b - 3 * b ** 2 / 10, 1 - 1 / (2 * b ** 2) + 1 / (5 * b ** 3))

        def vec_tau(b):
            return np.where(b <= 1, 2 * b / 3 - b ** 2 / 6, 1 - 2 / (3 * b) + 1 / (6 * b ** 2))

    xi, rho, tau = vec_xi(b_vals), vec_rho(b_vals), vec_tau(b_vals)
    mask = np.isfinite(xi) & np.isfinite(rho)
    return xi[mask], rho[mask], tau[mask]


def get_marshall_olkin_alpha1_1_curve(n_points=1000):
    """
    Marshall-Olkin with fixed alpha_1 = 1, varying alpha_2 in [0, 1].
    """
    a2 = np.linspace(0, 1, n_points)
    rho = 3 * a2 / (2 + a2)
    tau = a2
    xi = 2 * a2 / (3 - a2)
    return xi, rho, tau


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
        data_dir / f"{family}.pkl"
    ]
    file_path = next((c for c in candidates if c.exists()), None)

    if not file_path:
        return None, None, None

    try:
        data = pickle.loads(file_path.read_bytes())
        xi = data.values.get("chatterjees_xi")
        rho = data.values.get("spearmans_rho")
        tau = data.values.get("kendalls_tau")
        if xi is None or rho is None: return None, None, None

        if tau is None:
            tau = np.full_like(xi, np.nan)

        # Filter for positive dependence and valid values
        mask = np.isfinite(xi) & np.isfinite(rho) & (rho > 0.001)
        return xi[mask], rho[mask], tau[mask]
    except Exception as e:
        print(f"Error loading {family}: {e}")
        return None, None, None


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
        relative_candidate = script_location / "../../../docs/rank_correlation_estimates"
        if relative_candidate.resolve().exists():
            data_dir = relative_candidate.resolve()
            found = True

    if found:
        print(f"Loading data from: {data_dir.resolve()}")
    else:
        print("Warning: Could not find data directory. Empirical curves may be missing.")

    # --- Setup Envelope Data (Left Plot) ---
    eps = 1e-9
    xi_env_in = np.concatenate([
        np.linspace(0.0, 3 / 10 - eps, 150),
        np.linspace(3 / 10 - eps, 3 / 10 + eps, 50),
        np.linspace(3 / 10 + eps, 1.0, 150),
    ])
    xi_env_in = np.unique(np.clip(xi_env_in, 0.0, 1.0))
    rho_env = np.array([M_x_upper_bound_corrected(x) for x in xi_env_in])
    valid = ~np.isnan(rho_env)
    xi_env = xi_env_in[valid]
    rho_env = rho_env[valid]

    # --- Configuration ---
    colors = {
        "C_b": "blue",
        "Gaussian": "black",
        "MO_a1": "#17becf",  # Cyan
        "Clayton": "#d62728",
        "Frank": "#2ca02c",
        "GumbelHougaard": "#ff7f0e",
        "Joe": "#9467bd",
    }

    labels = {
        "GumbelHougaard": "Gumbel",
        "C_b": r"$(C_b)_{b>0}$",
        "MO_a1": r"Marshall-Olkin ($\alpha_1=1$)",
        "Clayton": "Clayton",
        "Frank": "Frank",
        "Joe": "Joe",
        "Gaussian": "Gaussian"
    }

    families = ["Clayton", "Frank", "GumbelHougaard", "Joe"]

    # --- Create Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # ----------------------------------------------------------
    # PLOT 1: Attainable Region (Left)
    # ----------------------------------------------------------
    BLUE_ENV = "#00529B"
    FILL_ENV = "#D6EAF8"

    # Envelope
    ax1.plot(rho_env, xi_env, color=BLUE_ENV, lw=2.5)
    ax1.plot(-rho_env, xi_env, color=BLUE_ENV, lw=2.5)
    ax1.fill_betweenx(xi_env, -rho_env, rho_env, color=FILL_ENV, alpha=0.5, zorder=0)

    # Diagonal
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="-", lw=1, alpha=0.5)

    def plot_fam_left(xi_arr, rho_arr, name, smooth=True):
        if len(xi_arr) == 0: return
        idx = np.argsort(rho_arr)
        r = rho_arr[idx]
        x = xi_arr[idx]

        # Only smooth empirical data
        if smooth and len(r) > 50:
            try:
                x = savgol_filter(x, 21, 3)
            except:
                pass

        c = colors.get(name, "gray")
        lbl = labels.get(name, name)

        # Solid: xi vs rho
        ax1.plot(r, x, color=c, lw=2.5, linestyle="-", label=lbl)
        # Dotted: sqrt(xi) vs rho
        x_sqrt = np.sqrt(np.clip(x, 0, None))
        ax1.plot(r, x_sqrt, color=c, lw=2.5, linestyle=":", alpha=0.8)

    # Analytic (No smoothing to preserve end-points)
    xi_g, rho_g, _ = get_gaussian_curve()
    plot_fam_left(xi_g, rho_g, "Gaussian", smooth=False)

    xi_mo1, rho_mo1, _ = get_marshall_olkin_alpha1_1_curve()
    plot_fam_left(xi_mo1, rho_mo1, "MO_a1", smooth=False)

    xi_cb, rho_cb, _ = get_cb_curve()
    plot_fam_left(xi_cb, rho_cb, "C_b", smooth=False)

    # Empirical (Smoothing enabled)
    for fam in families:
        xi_e, rho_e, _ = load_family_data(fam, data_dir)
        if xi_e is not None:
            plot_fam_left(xi_e, rho_e, fam, smooth=True)

    ax1.set_xlabel(r"$\rho$", fontsize=14)
    ax1.set_ylabel(r"$\xi$ (solid)  and  $\sqrt{\xi}$ (dotted)", fontsize=14)
    ax1.set_xlim(-0.1, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_title("Copula families within the attainable region", fontsize=14)

    # --- Legend Sorting Helper ---
    def get_sorted_handles_labels(ax):
        handles, lbls = ax.get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))

        def sort_key(item):
            label = item[0]
            # Priority: 0 if label has "C_b", 1 otherwise, then alphabetical
            priority = 0 if "C_b" in label else 1
            return (priority, label)

        sorted_items = sorted(by_label.items(), key=sort_key)
        if not sorted_items:
            return [], []

        # Unzip: sorted_items is [(label, handle), ...]
        labels_sorted, handles_sorted = zip(*sorted_items)
        return list(handles_sorted), list(labels_sorted)

    # Apply Legend Left
    handles1, labels1 = get_sorted_handles_labels(ax1)
    leg1 = ax1.legend(handles1, labels1, loc='upper left', fontsize=14, framealpha=0.9)
    ax1.add_artist(leg1)

    # Style Legend Left
    from matplotlib.lines import Line2D
    style_lines_left = [
        Line2D([0], [0], color='gray', lw=2.5, linestyle='-', label=r'$\xi$ vs $\rho$'),
        Line2D([0], [0], color='gray', lw=2.5, linestyle=':', label=r'$\sqrt{\xi}$ vs $\rho$')
    ]
    # ax1.legend(handles=style_lines_left, loc='lower right', fontsize=14, title="Curve Type", framealpha=0.9)

    # ----------------------------------------------------------
    # PLOT 2: Differences (Right)
    # ----------------------------------------------------------

    def plot_fam_right(xi_arr, rho_arr, tau_arr, name, smooth=True):
        if len(xi_arr) == 0: return
        idx = np.argsort(xi_arr)
        x = xi_arr[idx]
        y_rho = rho_arr[idx] - x
        y_tau = tau_arr[idx] - x

        # Only smooth empirical data
        if smooth and len(x) > 50:
            try:
                y_rho = savgol_filter(y_rho, 21, 3)
                if not np.all(np.isnan(y_tau)):
                    y_tau = savgol_filter(y_tau, 21, 3)
            except:
                pass

        c = colors.get(name, "gray")
        lbl = labels.get(name, name)

        # Solid: rho - xi
        ax2.plot(x, y_rho, color=c, lw=2.5, linestyle="-", label=lbl)

        # Dashed: tau - xi
        if not np.all(np.isnan(y_tau)):
            ax2.plot(x, y_tau, color=c, lw=2.5, linestyle="--")

    # Analytic (No smoothing)
    xi_g, rho_g, tau_g = get_gaussian_curve()
    plot_fam_right(xi_g, rho_g, tau_g, "Gaussian", smooth=False)

    xi_mo1, rho_mo1, tau_mo1 = get_marshall_olkin_alpha1_1_curve()
    plot_fam_right(xi_mo1, rho_mo1, tau_mo1, "MO_a1", smooth=False)

    xi_cb, rho_cb, tau_cb = get_cb_curve()
    plot_fam_right(xi_cb, rho_cb, tau_cb, "C_b", smooth=False)

    # Empirical (Smoothing enabled)
    for fam in families:
        xi_e, rho_e, tau_e = load_family_data(fam, data_dir)
        if xi_e is not None:
            plot_fam_right(xi_e, rho_e, tau_e, fam, smooth=True)

    ax2.set_xlabel(r"$\xi$", fontsize=14)
    ax2.set_ylabel(r"$\rho - \xi$ (solid) and  $\tau - \xi$ (dotted)", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.45)
    ax2.grid(True, linestyle=":", alpha=0.6)
    # Raw string for LaTeX
    ax2.set_title(r"Rank correlation differences: $(\rho - \xi)$ and $(\tau - \xi)$", fontsize=14)

    # Apply Legend Right
    handles2, labels2 = get_sorted_handles_labels(ax2)
    leg2 = ax2.legend(handles2, labels2, loc='upper right', fontsize=14, framealpha=0.9, ncol=2)
    ax2.add_artist(leg2)

    # Style Legend Right
    style_lines_right = [
        Line2D([0], [0], color='gray', lw=2.5, linestyle='-', label=r'$\rho - \xi$'),
        Line2D([0], [0], color='gray', lw=2.5, linestyle='--', label=r'$\tau - \xi$')
    ]
    # ax2.legend(handles=style_lines_right, loc='upper center', fontsize=14, frameon=False)

    # --- Save and Show ---
    plt.tight_layout()
    Path("images/").mkdir(parents=False, exist_ok=True)
    plt.savefig("images/combined_region_and_diffs.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()