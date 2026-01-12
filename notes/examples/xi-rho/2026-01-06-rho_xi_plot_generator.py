import pickle
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from pathlib import Path
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import Dict


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

    candidates = [data_dir / f"{family}_data.pkl", data_dir / f"{family}.pkl"]

    file_path = next((c for c in candidates if c.exists()), None)

    if not file_path:
        return None, None, None

    try:
        data = pickle.loads(file_path.read_bytes())
    except Exception as e:
        print(f"Error loading {family}: {e}")
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
    mask = np.isfinite(xi) & np.isfinite(rho) & (rho > 0.001)

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

    with np.errstate(divide="ignore", invalid="ignore"):

        def vec_xi(b):
            return np.where(
                b <= 1, (b**2 / 10) * (5 - 2 * b), 1 - 1 / b + 3 / (10 * b**2)
            )

        def vec_rho(b):
            return np.where(
                b <= 1, b - 3 * b**2 / 10, 1 - 1 / (2 * b**2) + 1 / (5 * b**3)
            )

        def vec_tau(b):
            return np.where(
                b <= 1, 2 * b / 3 - b**2 / 6, 1 - 2 / (3 * b) + 1 / (6 * b**2)
            )

    xi, rho, tau = vec_xi(b_vals), vec_rho(b_vals), vec_tau(b_vals)
    mask = np.isfinite(xi) & np.isfinite(rho)
    return xi[mask], rho[mask], tau[mask]


def get_marshall_olkin_alpha1_1_curve(n_points=1000):
    """
    Returns the curve for Marshall-Olkin with fixed alpha_1 = 1
    and alpha_2 varying in [0, 1].

    Based on formulas:
      rho = (3 * a1 * a2) / (2*a1 - a1*a2 + 2*a2)
      tau = (a1 * a2) / (a1 - a1*a2 + a2)
      xi  = (2 * a1^2 * a2) / (3*a1 + a2 - 2*a1*a2)

    Substituting a1 = 1, a2 = x:
      rho = 3x / (2 + x)
      tau = x
      xi  = 2x / (3 - x)
    """
    # alpha_2 ranges from 0 to 1
    a2 = np.linspace(0, 1, n_points)

    rho = 3 * a2 / (2 + a2)
    tau = a2
    xi = 2 * a2 / (3 - a2)

    return xi, rho, tau


# ------------------------------------------------------------------
# 4. Plotting
# ------------------------------------------------------------------


def main():
    # --- Robust Data Path Finding ---
    data_dir = Path("rank_correlation_estimates")
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
            "Warning: Could not find 'rank_correlation_estimates'. Plotting analytic curves only."
        )

    families = ["Clayton", "Frank", "GumbelHougaard", "Joe", "MarshallOlkin"]

    colors = {
        "C_b": "blue",
        "Gaussian": "black",
        "MarshallOlkin": "#17becf",  # Cyan
        "Clayton": "#d62728",
        "Frank": "#2ca02c",
        "GumbelHougaard": "#ff7f0e",
        "Joe": "#9467bd",
    }

    display_labels = {
        "GumbelHougaard": "Gumbel",
        "C_b": r"$(C_b)_{b>0}$",
        "MarshallOlkin": r"Marshall-Olkin ($\alpha_1=1$)",
    }

    plt.figure(figsize=(9, 7))

    # --- 1. Analytic Curves ---

    # C_b
    xi_cb, rho_cb, tau_cb = get_cb_curve()
    plt.plot(
        xi_cb,
        rho_cb - xi_cb,
        color=colors["C_b"],
        linewidth=2,
        label=display_labels["C_b"],
    )
    plt.plot(xi_cb, tau_cb - xi_cb, color=colors["C_b"], linewidth=2, linestyle="--")

    # Gaussian
    xi_g, rho_g, tau_g = get_gaussian_curve()
    plt.plot(
        xi_g, rho_g - xi_g, color=colors["Gaussian"], linewidth=2, label="Gaussian"
    )
    plt.plot(xi_g, tau_g - xi_g, color=colors["Gaussian"], linewidth=2, linestyle="--")

    # Marshall-Olkin (alpha_1 = 1)
    xi_mo, rho_mo, tau_mo = get_marshall_olkin_alpha1_1_curve()
    plt.plot(
        xi_mo,
        rho_mo - xi_mo,
        color=colors["MarshallOlkin"],
        linewidth=2,
        label=display_labels["MarshallOlkin"],
    )
    plt.plot(
        xi_mo,
        tau_mo - xi_mo,
        color=colors["MarshallOlkin"],
        linewidth=2,
        linestyle="--",
    )

    # --- 2. Empirical Families (Apply Smoothing) ---
    for fam in families:
        xi_raw, rho_raw, tau_raw = load_family_data(fam, data_dir)

        if xi_raw is not None and len(xi_raw) > 0:
            label_name = display_labels.get(fam, fam)
            c = colors.get(fam, "gray")

            sort_idx = np.argsort(xi_raw)
            x_plot = xi_raw[sort_idx]
            y_rho = rho_raw[sort_idx] - x_plot
            y_tau = tau_raw[sort_idx] - x_plot

            window = 21
            poly = 3

            # Only smooth if enough points
            if len(x_plot) > window:
                try:
                    y_rho_smooth = savgol_filter(
                        y_rho, window_length=window, polyorder=poly
                    )
                    plt.plot(
                        x_plot, y_rho_smooth, label=label_name, color=c, linewidth=2
                    )
                except Exception:
                    plt.plot(x_plot, y_rho, label=label_name, color=c, linewidth=2)

                if not np.all(np.isnan(y_tau)):
                    try:
                        y_tau_smooth = savgol_filter(
                            y_tau, window_length=window, polyorder=poly
                        )
                        plt.plot(
                            x_plot, y_tau_smooth, color=c, linewidth=2, linestyle="--"
                        )
                    except Exception:
                        plt.plot(x_plot, y_tau, color=c, linewidth=2, linestyle="--")
            else:
                plt.plot(x_plot, y_rho, label=label_name, color=c, linewidth=2)
                if not np.all(np.isnan(y_tau)):
                    plt.plot(x_plot, y_tau, color=c, linewidth=2, linestyle="--")

    # --- Formatting ---
    plt.xlim(0, 1)
    plt.ylim(0, 0.45)
    plt.xlabel(r"$\xi$", fontsize=12)
    plt.ylabel(r"$\rho - \xi$ (solid)  and  $\tau - \xi$ (dashed)", fontsize=12)
    plt.title(r"Differences between rank correlations", fontsize=14)

    # Legend Logic
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    leg1 = plt.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        frameon=True,
        fontsize=10,
        ncol=2,
    )
    plt.gca().add_artist(leg1)

    # Style legend
    from matplotlib.lines import Line2D

    style_handles = [
        Line2D([0], [0], color="gray", lw=2, linestyle="-", label=r"$\rho - \xi$"),
        Line2D([0], [0], color="gray", lw=2, linestyle="--", label=r"$\tau - \xi$"),
    ]
    plt.legend(handles=style_handles, loc="upper center", fontsize=10, frameon=False)

    plt.grid(True, linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("rho_tau_minus_xi_plot_smoothed.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
