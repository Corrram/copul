import pickle
import importlib.resources as pkg_resources
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from scipy.signal import savgol_filter


# ------------------------------------------------------------------
# 1. Define the Data Class
# ------------------------------------------------------------------
@dataclass
class CorrelationData:
    params: np.ndarray
    values: Dict[str, np.ndarray]


# ------------------------------------------------------------------
# 2. Data Loaders
# ------------------------------------------------------------------
def load_family_data(family: str, data_dir: Path):
    """Loads CorrelationData object from the pickle file."""
    candidates = [data_dir / f"{family}_data.pkl"]

    file = None
    for cand in candidates:
        if cand.exists():
            file = cand
            break

    if file is None:
        print(f"Warning: File for {family} not found. Skipped.")
        return None, None

    try:
        data = pickle.loads(file.read_bytes())
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None, None

    try:
        xi = data.values.get("chatterjees_xi")
        rho = data.values.get("spearmans_rho")
    except AttributeError:
        return None, None

    # Filter: Valid xi/rho and removing extremely low rho (artifacts)
    mask = np.isfinite(xi) & np.isfinite(rho) & (rho > 0.01)

    return xi[mask], rho[mask]


# ------------------------------------------------------------------
# 3. Analytic Models
# ------------------------------------------------------------------
def get_gaussian_curve(n_points=300):
    r = np.linspace(0, 0.9999, n_points)
    # rho_s = (6 / pi) * arcsin(r / 2)
    rho_s = (6 / np.pi) * np.arcsin(r / 2)
    # xi = (3 / pi) * arcsin((1 + r^2) / 2) - 0.5
    xi = (3 / np.pi) * np.arcsin((1 + r**2) / 2) - 0.5
    return xi, rho_s


def get_cb_curve(n_points=1000):
    # Parameter b ranges from 0 to infinity
    b_vals = np.concatenate([np.linspace(0, 1, 1_000), np.linspace(1, 100, 1_000)[1:]])

    def vec_xi(b):
        return np.where(b <= 1, (b**2 / 10) * (5 - 2 * b), 1 - 1 / b + 3 / (10 * b**2))

    def vec_rho(b):
        return np.where(b <= 1, b - 3 * b**2 / 10, 1 - 1 / (2 * b**2) + 1 / (5 * b**3))

    return vec_xi(b_vals), vec_rho(b_vals)


# ------------------------------------------------------------------
# 4. Plotting
# ------------------------------------------------------------------
def main():
    try:
        with pkg_resources.path("copul", "docs") as docs_path:
            data_dir = docs_path / "rank_correlation_estimates"
    except (ImportError, ModuleNotFoundError, AttributeError):
        data_dir = Path("rank_correlation_estimates")

    families = ["Clayton", "Frank", "GumbelHougaard", "Joe"]

    colors = {
        "C_b": "blue",
        "Gaussian": "black",
        "Clayton": "#d62728",
        "Frank": "#2ca02c",
        "GumbelHougaard": "#ff7f0e",
        "Joe": "#9467bd",
    }
    display_labels = {"GumbelHougaard": "Gumbel", "C_b": r"$(C_b)_{b>0}$"}

    plt.figure(figsize=(9, 7))

    # --- Analytic Curves (Already smooth) ---
    # 1. C_b
    xi_cb, rho_cb = get_cb_curve()

    # y1: rho - xi
    plt.plot(
        xi_cb,
        rho_cb - xi_cb,
        color=colors["C_b"],
        linewidth=2,
        label=display_labels["C_b"],
    )
    # y2: rho - sqrt(xi)
    # Use max(xi, 0) to avoid warnings if analytic float precision drops slightly below 0
    plt.plot(
        xi_cb,
        rho_cb - np.sqrt(np.maximum(xi_cb, 0)),
        color=colors["C_b"],
        linewidth=2,
        linestyle="--",
    )

    # 2. Gaussian
    xi_g, rho_g = get_gaussian_curve()
    plt.plot(
        xi_g, rho_g - xi_g, color=colors["Gaussian"], linewidth=2, label="Gaussian"
    )
    plt.plot(
        xi_g,
        rho_g - np.sqrt(np.maximum(xi_g, 0)),
        color=colors["Gaussian"],
        linewidth=2,
        linestyle="--",
    )

    # --- Empirical Families (Apply Smoothing) ---
    for fam in families:
        xi_raw, rho_raw = load_family_data(fam, data_dir)

        if xi_raw is not None and len(xi_raw) > 0:
            label_name = display_labels.get(fam, fam)
            c = colors.get(fam, "gray")

            # 1. Sort by xi (Critical for line plots)
            sort_idx = np.argsort(xi_raw)
            x_plot = xi_raw[sort_idx]

            # y1: rho - xi
            y_rho_diff_xi = rho_raw[sort_idx] - x_plot

            # y2: rho - sqrt(xi)
            y_rho_diff_sqrt_xi = rho_raw[sort_idx] - np.sqrt(np.maximum(x_plot, 0))

            # 2. Smooth the curves
            window = 21
            poly = 3

            if len(x_plot) > window:
                try:
                    y_rho_diff_xi = savgol_filter(
                        y_rho_diff_xi, window_length=window, polyorder=poly
                    )
                    y_rho_diff_sqrt_xi = savgol_filter(
                        y_rho_diff_sqrt_xi, window_length=window, polyorder=poly
                    )
                except Exception:
                    pass  # Fallback to raw data if smoothing fails

            # 3. Plot
            plt.plot(x_plot, y_rho_diff_xi, label=label_name, color=c, linewidth=2)
            plt.plot(x_plot, y_rho_diff_sqrt_xi, color=c, linewidth=2, linestyle="--")

    # --- Formatting ---
    plt.xlim(0, 1)

    # Adjusted limits: rho - sqrt(xi) will often be negative
    plt.ylim(0, 0.4)

    plt.xlabel(r"$\xi$", fontsize=12)
    plt.ylabel(r"Solid: $\rho - \xi$ | Dashed: $\rho - \sqrt{\xi}$", fontsize=12)
    plt.title(r"Differences: $(\rho - \xi)$ and $(\rho - \sqrt{\xi})$", fontsize=14)

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, loc="lower right", frameon=True, fontsize=12, ncol=2)

    plt.grid(True, linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("rho_minus_xi_and_sqrt_xi.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
