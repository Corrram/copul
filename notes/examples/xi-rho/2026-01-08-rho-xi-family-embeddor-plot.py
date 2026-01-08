import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
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
    r = np.linspace(0, 0.9999, n_points)
    rho_s = (6 / np.pi) * np.arcsin(r / 2)
    xi = (3 / np.pi) * np.arcsin((1 + r ** 2) / 2) - 0.5
    return xi, rho_s


def get_cb_curve(n_points=1000):
    # Focus on positive dependence part
    b_vals = np.linspace(0, 10, 2000)

    # Use errstate to ignore division by zero in the unused branch of np.where
    with np.errstate(divide='ignore', invalid='ignore'):
        def vec_xi(b):
            # When b <= 1 (first branch), the second branch (containing 1/b)
            # is evaluated but discarded. We allow the inf/nan there.
            return np.where(b <= 1, (b ** 2 / 10) * (5 - 2 * b), 1 - 1 / b + 3 / (10 * b ** 2))

        def vec_rho(b):
            return np.where(b <= 1, b - 3 * b ** 2 / 10, 1 - 1 / (2 * b ** 2) + 1 / (5 * b ** 3))

        xi = vec_xi(b_vals)
        rho = vec_rho(b_vals)

    # Filter only positive rho and finite values
    mask = (rho >= 0) & np.isfinite(rho) & np.isfinite(xi)
    return xi[mask], rho[mask]


def get_marshall_olkin_alpha1_1_curve(n_points=1000):
    """
    Returns the curve for Marshall-Olkin with fixed alpha_1 = 1
    and alpha_2 varying in [0, 1].

    Formulas:
      rho = 3*a2 / (2 + a2)
      xi  = 2*a2 / (3 - a2)
    """
    a2 = np.linspace(0, 1, n_points)
    rho = 3 * a2 / (2 + a2)
    xi = 2 * a2 / (3 - a2)
    return xi, rho


# ------------------------------------------------------------------
# 2. Data Loader
# ------------------------------------------------------------------

@dataclass
class CorrelationData:
    params: np.ndarray
    values: Dict[str, np.ndarray]


def load_family_data(family: str, data_dir: Path):
    # Try multiple naming conventions
    candidates = [
        data_dir / f"{family}_data.pkl",
        data_dir / f"{family}.pkl"
    ]

    file_path = next((c for c in candidates if c.exists()), None)

    if not file_path:
        return None, None

    try:
        data = pickle.loads(file_path.read_bytes())
        xi = data.values.get("chatterjees_xi")
        rho = data.values.get("spearmans_rho")
        if xi is None or rho is None: return None, None

        # Filter for positive dependence and valid values
        mask = np.isfinite(xi) & np.isfinite(rho) & (rho > 0.001)
        return xi[mask], rho[mask]
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

    # 1. Try using the package resource
    try:
        with pkg_resources.path("copul", "docs") as docs_path:
            possible_dir = docs_path / "rank_correlation_estimates"
            if possible_dir.exists():
                data_dir = possible_dir
                found = True
    except (ImportError, ModuleNotFoundError, AttributeError):
        pass

    # 2. If not found via package, try relative to this script's location
    if not found:
        # Based on path: notes/examples/xi-rho/script.py -> ../../../docs/rank_correlation_estimates
        script_location = Path(__file__).parent
        relative_candidate = script_location / "../../../docs/rank_correlation_estimates"
        if relative_candidate.resolve().exists():
            data_dir = relative_candidate.resolve()
            found = True

    if found:
        print(f"Loading data from: {data_dir.resolve()}")
    else:
        print(
            f"Warning: Could not find data directory. Looked in package 'copul.docs' and relative path.")

    # --- Setup Envelope Data ---
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

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))

    BLUE = "#00529B"
    FILL = "#D6EAF8"

    # 1. Draw Envelope
    ax.plot(rho_env, xi_env, color=BLUE, lw=2)
    ax.plot(-rho_env, xi_env, color=BLUE, lw=2)
    ax.fill_betweenx(xi_env, -rho_env, rho_env, color=FILL, alpha=0.5, zorder=0)

    # 2. Draw Diagonal (y=x) for reference
    ax.plot([0, 1], [0, 1], color="gray", linestyle="-", lw=1, alpha=0.5)

    # 3. Plot Families
    colors = {
        "C_b": "blue",
        "Gaussian": "black",
        "MarshallOlkin": "#17becf",  # Cyan
        "Clayton": "#d62728",
        "Frank": "#2ca02c",
        "GumbelHougaard": "#ff7f0e",
        "Joe": "#9467bd",
    }

    labels = {
        "GumbelHougaard": "Gumbel",
        "C_b": r"$(C_b)_{b>0}$",
        "MarshallOlkin": r"Marshall-Olkin ($\alpha_1=1$)",
        "Clayton": "Clayton",
        "Frank": "Frank",
        "Joe": "Joe",
        "Gaussian": "Gaussian"
    }

    def plot_fam(xi_arr, rho_arr, name):
        if len(xi_arr) == 0: return

        # Sort by rho for drawing clean lines
        idx = np.argsort(rho_arr)
        r = rho_arr[idx]
        x = xi_arr[idx]

        # Smooth if empirical (points > 50)
        if len(r) > 50:
            try:
                x = savgol_filter(x, 21, 3)
            except:
                pass

        c = colors.get(name, "gray")
        lbl = labels.get(name, name)

        # Plot (rho, xi)
        ax.plot(r, x, color=c, lw=2, linestyle="-", label=lbl)

        # Plot (rho, sqrt(xi))
        # Note: If xi < 0 due to numerical noise, clip it
        x_sqrt = np.sqrt(np.clip(x, 0, None))
        ax.plot(r, x_sqrt, color=c, lw=2, linestyle=":", alpha=0.8)

    # --- Analytic Families ---
    xi_g, rho_g = get_gaussian_curve()
    plot_fam(xi_g, rho_g, "Gaussian")

    xi_mo, rho_mo = get_marshall_olkin_alpha1_1_curve()
    plot_fam(xi_mo, rho_mo, "MarshallOlkin")

    xi_cb, rho_cb = get_cb_curve()
    plot_fam(xi_cb, rho_cb, "C_b")

    # --- Empirical Families ---
    for fam in ["Clayton", "Frank", "GumbelHougaard", "Joe"]:
        xi_e, rho_e = load_family_data(fam, data_dir)
        if xi_e is not None:
            plot_fam(xi_e, rho_e, fam)
        else:
            if found:
                print(f"Warning: File for {fam} not found in {data_dir}")

    # --- Formatting ---
    ax.set_xlabel(r"Spearman's $\rho$", fontsize=14)
    ax.set_ylabel(r"$\xi$ (solid)  and  $\sqrt{\xi}$ (dotted)", fontsize=14)
    ax.set_xlim(-0.1, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=":", alpha=0.6)

    # Legend Handling
    handles, plot_labels = ax.get_legend_handles_labels()
    by_label = dict(zip(plot_labels, handles))

    # Custom handles for line styles
    from matplotlib.lines import Line2D
    style_lines = [
        Line2D([0], [0], color='gray', lw=2, linestyle='-', label=r'$\xi$ vs $\rho$'),
        Line2D([0], [0], color='gray', lw=2, linestyle=':', label=r'$\sqrt{\xi}$ vs $\rho$')
    ]

    leg1 = ax.legend(handles=list(by_label.values()), labels=list(by_label.keys()),
                     loc='upper left', fontsize=10, framealpha=0.9)
    # ax.legend(handles=style_lines, loc='lower right', fontsize=11, title="Curve Type", framealpha=0.9)
    ax.add_artist(leg1)

    ax.set_title(
        "Copula families within the attainable region",
        fontsize=14)

    Path("images/").mkdir(parents=False, exist_ok=True)
    plt.savefig("images/attainable_region_embedded_curves.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()