import pickle
import importlib.resources as pkg_resources
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from scipy.signal import savgol_filter  # <--- Import for smoothing


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

    # Look for files (e.g. Clayton_data.pkl)
    candidates = [data_dir / f"{family}_data.pkl"]

    file = None
    for cand in candidates:
        if cand.exists():
            file = cand
            break

    if file is None:
        print(f"Warning: File for {family} not found. Skipped.")
        return None, None, None

    try:
        data = pickle.loads(file.read_bytes())
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None, None, None

    try:
        xi = data.values.get("chatterjees_xi")
        rho = data.values.get("spearmans_rho")
        tau = data.values.get("kendalls_tau")
    except AttributeError:
        return None, None, None

    if tau is None:
        tau = np.full_like(xi, np.nan)

    # Filter: Valid xi/rho and removing extremely low rho (artifacts)
    mask = np.isfinite(xi) & np.isfinite(rho) & (rho > 0.01)

    return xi[mask], rho[mask], tau[mask]


# ------------------------------------------------------------------
# 3. Analytic Models
# ------------------------------------------------------------------


def get_gaussian_curve(n_points=300):
    r = np.linspace(0, 0.9999, n_points)
    rho_s = (6 / np.pi) * np.arcsin(r / 2)
    tau = (2 / np.pi) * np.arcsin(r)
    xi = (3 / np.pi) * np.arcsin((1 + r**2) / 2) - 0.5
    return xi, rho_s, tau


def get_cb_curve(n_points=1000):
    b_vals = np.concatenate([np.linspace(0, 1, 1_000), np.linspace(1, 100, 1_000)[1:]])

    def vec_xi(b):
        return np.where(b <= 1, (b**2 / 10) * (5 - 2 * b), 1 - 1 / b + 3 / (10 * b**2))

    def vec_rho(b):
        return np.where(b <= 1, b - 3 * b**2 / 10, 1 - 1 / (2 * b**2) + 1 / (5 * b**3))

    def vec_tau(b):
        return np.where(b <= 1, 2 * b / 3 - b**2 / 6, 1 - 2 / (3 * b) + 1 / (6 * b**2))

    return vec_xi(b_vals), vec_rho(b_vals), vec_tau(b_vals)


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

    # --- Analytic Curves (Already smooth, no filter needed) ---
    xi_cb, rho_cb, tau_cb = get_cb_curve()
    plt.plot(
        xi_cb,
        rho_cb - xi_cb,
        color=colors["C_b"],
        linewidth=2,
        label=display_labels["C_b"],
    )
    plt.plot(xi_cb, tau_cb - xi_cb, color=colors["C_b"], linewidth=2, linestyle="--")

    xi_g, rho_g, tau_g = get_gaussian_curve()
    plt.plot(
        xi_g, rho_g - xi_g, color=colors["Gaussian"], linewidth=2, label="Gaussian"
    )
    plt.plot(xi_g, tau_g - xi_g, color=colors["Gaussian"], linewidth=2, linestyle="--")

    # --- Empirical Families (Apply Smoothing) ---
    for fam in families:
        xi_raw, rho_raw, tau_raw = load_family_data(fam, data_dir)

        if xi_raw is not None and len(xi_raw) > 0:
            label_name = display_labels.get(fam, fam)
            c = colors.get(fam, "gray")

            # 1. Sort by xi (Critical for line plots)
            sort_idx = np.argsort(xi_raw)
            x_plot = xi_raw[sort_idx]
            y_rho = rho_raw[sort_idx] - x_plot
            y_tau = tau_raw[sort_idx] - x_plot

            # 2. Smooth the curves
            # Window length needs to be odd and <= len(x).
            # 21 is usually a good balance for ~200 data points.
            window = 21
            poly = 3

            if len(x_plot) > window:
                try:
                    y_rho = savgol_filter(y_rho, window_length=window, polyorder=poly)
                    # Only smooth tau if it's not full of NaNs
                    if not np.all(np.isnan(y_tau)):
                        y_tau = savgol_filter(
                            y_tau, window_length=window, polyorder=poly
                        )
                except Exception:
                    pass  # Fallback to raw data if smoothing fails

            # 3. Plot
            plt.plot(x_plot, y_rho, label=label_name, color=c, linewidth=2)

            if not np.all(np.isnan(y_tau)):
                plt.plot(x_plot, y_tau, color=c, linewidth=2, linestyle="--")

    # --- Formatting ---
    plt.xlim(0, 1)
    plt.ylim(0, 0.45)
    plt.xlabel(r"$\xi$", fontsize=12)
    plt.ylabel(r"$\rho - \xi$ and $\tau - \xi$", fontsize=12)
    plt.title(r"Differences between rank correlations", fontsize=14)

    # Legend
    from matplotlib.lines import Line2D

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, loc="upper right", frameon=True, fontsize=12, ncol=2)

    plt.grid(True, linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("rho_tau_minus_xi_plot_smoothed.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
