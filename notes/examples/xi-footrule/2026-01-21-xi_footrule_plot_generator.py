import pickle
import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as pkg_resources
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal, norm
from scipy.integrate import quad
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
def load_family_data(family: str, data_dir: Path, param_range=(None, None)):
    """Loads CorrelationData object from the pickle file."""
    candidates = [data_dir / f"{family}_data.pkl", data_dir / f"{family}.pkl"]
    file_path = next((c for c in candidates if c.exists()), None)
    if not file_path:
        return None, None
    try:
        data = pickle.loads(file_path.read_bytes())
    except Exception as e:
        print(f"Error loading {family}: {e}")
        return None, None
    try:
        xi = data.values.get("chatterjees_xi")
        phi = data.values.get("spearmans_footrule")
        if phi is None:
            phi = data.values.get("phi")
    except AttributeError:
        return None, None
    if param_range is not None:
        pmin, pmax = param_range
        mask = np.ones_like(data.params, dtype=bool)
        if pmin is not None:
            mask &= data.params >= pmin
        if pmax is not None:
            mask &= data.params <= pmax
        xi = xi[mask]
        phi = phi[mask]
    if phi is None:
        return None, None
    mask = np.isfinite(xi) & np.isfinite(phi)
    return xi[mask], phi[mask]


# ------------------------------------------------------------------
# 3. Analytic Models
# ------------------------------------------------------------------
def calculate_gaussian_footrule(rho_param):
    def integrand(u):
        if u <= 0 or u >= 1:
            return 0
        z = norm.ppf(u)
        return multivariate_normal.cdf(
            [z, z], mean=[0, 0], cov=[[1, rho_param], [rho_param, 1]]
        )

    integral, _ = quad(integrand, 0, 1, points=[0.5])
    return 6 * integral - 2


def get_gaussian_curve_phi_xi(n_points=50):
    # Plotting negative dependence branch to match user intent
    r_vals = np.linspace(-0.01, -0.999, n_points)
    xi = (3 / np.pi) * np.arcsin((1 + r_vals**2) / 2) - 0.5
    phi = np.array([calculate_gaussian_footrule(r) for r in r_vals])
    return xi, phi


def get_upper_frechet_curve(n_points=100):
    """Upper Boundary: psi = sqrt(xi)"""
    xi = np.linspace(0, 1, n_points)
    phi = np.sqrt(xi)
    return xi, phi


def get_lower_frechet_curve(n_points=100):
    """
    Lower Fréchet Family: Mixture of Independence (Pi) and Lower Bound (W).
    C_theta = (1-theta)*Pi + theta*W  for theta in [0, 1].

    Analytic properties:
    xi = theta^2
    psi = -0.5 * theta

    Resulting curve: psi = -0.5 * sqrt(xi)
    """
    theta = np.linspace(0, 1, n_points)
    xi = theta**2
    phi = -0.5 * theta
    return xi, phi


def get_jensen_lower_bound_curve(n_points=100):
    """Jensen Lower Bound (Proposition 3.4)"""
    v1 = np.linspace(0.5, 1.0, n_points)
    phi = -2 * v1**2 + 6 * v1 - 5 + (1 / v1)
    xi = -4 * v1**2 + 20 * v1 - 17 + (2 / v1) - (1 / v1**2) - 12 * np.log(v1)
    return xi, phi


# --- C_mu Implementation (Section 3.2) ---
def psi_s_func(s, alpha, beta):
    """Eq (23): Lower boundary of the zero density strip."""
    if alpha >= 0.5:
        return (1 - beta) if s > 0.5 else 0.0
    if 0 <= s <= alpha:
        return 0.0
    elif alpha < s < 1 - alpha:
        return ((1 - beta) / (1 - 2 * alpha)) * (s - alpha)
    elif 1 - alpha <= s <= 1:
        return 1 - beta
    return 0.0


def F_T_inv_func(y, alpha, beta):
    """Proposition 3.5: Inverse CDF of the transformed variable."""
    if beta >= 1.0:
        return y
    a = (1 - alpha) / (1 - beta)
    denom = (1 - beta) ** 2
    kappa = (1 - 2 * alpha) / denom
    c = (1 - 2 * beta + 2 * alpha * beta) / denom
    u1 = a * beta - 0.5 * kappa * beta**2
    res = np.zeros_like(y)

    # Handle limit case kappa -> 0 (when alpha -> 0.5)
    if abs(kappa) < 1e-7:
        mask1 = y < u1
        mask2 = (y >= u1) & (y <= 1 - u1)
        mask3 = y > 1 - u1
        res[mask1] = y[mask1] / a
        res[mask2] = beta + (y[mask2] - u1) / c
        res[mask3] = 1 - (1 - y[mask3]) / a
    else:
        mask1 = y < u1
        mask2 = (y >= u1) & (y <= 1 - u1)
        mask3 = y > 1 - u1

        val1 = a**2 - 2 * kappa * y[mask1]
        val1[val1 < 0] = 0
        res[mask1] = (a - np.sqrt(val1)) / kappa

        res[mask2] = beta + (y[mask2] - u1) / c

        val3 = a**2 - 2 * kappa * (1 - y[mask3])
        val3[val3 < 0] = 0
        res[mask3] = 1 - (a - np.sqrt(val3)) / kappa
    return res


def calc_cmu_point(mu, grid_size=200):
    """Numerically calculates xi and psi for C_mu."""
    # Eq (27): Parameter path
    if mu <= 2:
        alpha = 0.15 * mu
        beta = 0.25 * mu
    else:
        alpha = 0.5 - 0.4 / mu
        beta = 0.5

    # Numerical safety for alpha=0.5
    if alpha > 0.49999:
        alpha = 0.49999
    if beta > 0.999:
        beta = 0.999

    # 1. Calculate Psi: 6 * int C(u,u) - 2
    def get_C_uu(u_val):
        t = F_T_inv_func(np.array([u_val]), alpha, beta)[0]
        res, _ = quad(
            lambda s: min(beta, max(0, t - psi_s_func(s, alpha, beta))),
            0,
            u_val,
            points=[alpha, 1 - alpha],
        )
        return (u_val * t - res) / (1 - beta)

    int_psi, _ = quad(get_C_uu, 0, 1, points=[alpha, 1 - alpha])
    psi = 6 * int_psi - 2

    # 2. Calculate Xi: 6 * int (dC/du)^2 - 2
    u_grid = np.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size)
    v_grid = np.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size)

    T_vals = F_T_inv_func(v_grid, alpha, beta)
    T_matrix = T_vals[:, np.newaxis]  # Shape (Grid, 1) -> T(v)

    Psi_vals = np.array([psi_s_func(u, alpha, beta) for u in u_grid])
    Psi_matrix = Psi_vals[np.newaxis, :]  # Shape (1, Grid) -> Psi(u)

    Term = np.maximum(0, T_matrix - Psi_matrix)
    Term = np.minimum(beta, Term)
    Partial = (T_matrix - Term) / (1 - beta)

    xi = 6 * np.mean(Partial**2) - 2

    return xi, psi


def get_cmu_curve():
    mus = np.concatenate([np.linspace(0, 2, 15), np.linspace(2.5, 10, 10)])
    xis = []
    psis = []
    print("Calculating C_mu curve (this may take a moment)...")
    for mu in mus:
        x, p = calc_cmu_point(mu)
        xis.append(x)
        psis.append(p)
    return np.array(xis), np.array(psis)


# ------------------------------------------------------------------
# 4. Main Plotting Routine
# ------------------------------------------------------------------
def main():
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

    if found:
        print(f"Loading data from: {data_dir}")
    else:
        print(
            "Warning: Could not find 'rank_correlation_estimates'. Plotting analytic curves only."
        )

    families = ["Clayton", "Frank", "Joe"]

    colors = {
        "Gaussian": "black",
        "Clayton": "#d62728",
        "Frank": "#2ca02c",
        "GumbelHougaard": "#ff7f0e",
        "Joe": "#9467bd",
        "JensenBound": "#bcbd22",
        "LowerFrechet": "gray",
        "CMu": "blue",
    }

    # Restricting to negative dependence where possible
    param_ranges = {"Clayton": (None, 0), "Frank": (None, 0)}
    display_labels = {
        "GumbelHougaard": "Gumbel",
        "LowerFrechet": r"Lower Fréchet",
    }

    plt.figure(figsize=(9, 7))

    # --- 1. Analytic Curves ---

    # A. Jensen Bound
    xi_jl, phi_jl = get_jensen_lower_bound_curve()
    plt.plot(
        xi_jl,
        phi_jl,
        color=colors["JensenBound"],
        linewidth=2.5,
        linestyle="-",
        label=r"$C^{\searrow}_\mu$",
    )

    # C. C_mu Curve
    xi_cm, phi_cm = get_cmu_curve()
    plt.plot(
        xi_cm,
        phi_cm,
        color=colors["CMu"],
        linewidth=2.5,
        linestyle="-",
        label=r"$C_\mu$",
    )

    # B. Lower Fréchet Curve (New)
    xi_lf, phi_lf = get_lower_frechet_curve()
    plt.plot(
        xi_lf,
        phi_lf,
        color=colors["LowerFrechet"],
        linewidth=2.5,
        linestyle="-",
        label=display_labels["LowerFrechet"],
    )

    # D. Gaussian
    print("Calculating Gaussian...")
    xi_g, phi_g = get_gaussian_curve_phi_xi(n_points=30)
    plt.plot(xi_g, phi_g, color=colors["Gaussian"], linewidth=2.5, label="Gaussian")

    # --- 2. Empirical Families ---
    for fam in families:
        param_range = param_ranges.get(fam, (None, None))
        xi_raw, phi_raw = load_family_data(fam, data_dir, param_range)

        if xi_raw is not None and len(xi_raw) > 0:
            label_name = display_labels.get(fam, fam)
            c = colors.get(fam, "gray")

            sort_idx = np.argsort(xi_raw)
            x_plot = xi_raw[sort_idx]
            y_plot = phi_raw[sort_idx]

            # Smoothing
            window = 21
            poly = 3
            if len(x_plot) > window:
                try:
                    y_smooth = savgol_filter(
                        y_plot, window_length=window, polyorder=poly
                    )
                    plt.plot(x_plot, y_smooth, label=label_name, color=c, linewidth=2.5)
                except Exception:
                    plt.plot(x_plot, y_plot, label=label_name, color=c, linewidth=2.5)
            else:
                plt.plot(x_plot, y_plot, label=label_name, color=c, linewidth=2.5)

    # --- Formatting ---
    plt.xlim(0, 1)
    plt.ylim(-0.55, 0.05)  # Zoomed in on negative Psi values as requested
    plt.xlabel(r"Chatterjee's rank correlation $\xi$", fontsize=12)
    plt.ylabel(r"Spearman's footrule $\psi$", fontsize=12)
    plt.title(r"$(\xi, \psi)$ for different copula families", fontsize=14)
    plt.legend(loc="upper right", frameon=True, fontsize=12, ncol=2)
    plt.grid(True, linestyle="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("psi_vs_xi_lower_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
