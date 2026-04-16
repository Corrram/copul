import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal, norm
from scipy.integrate import quad
from plot_utils import (
    CorrelationData, find_data_dir, load_family_data,
    get_gaussian_xi_rho_tau,
)


# ── Analytic curves specific to (ξ, ψ) ───────────────────────────────────────

def _gaussian_footrule(rho_param: float) -> float:
    def integrand(u):
        if u <= 0 or u >= 1:
            return 0
        z = norm.ppf(u)
        return multivariate_normal.cdf(
            [z, z], mean=[0, 0], cov=[[1, rho_param], [rho_param, 1]]
        )
    integral, _ = quad(integrand, 0, 1, points=[0.5])
    return 6 * integral - 2

def get_gaussian_xi_footrule(n_points: int = 50):
    r_vals = np.linspace(-0.01, -0.999, n_points)
    xi  = (3/np.pi) * np.arcsin((1 + r_vals**2) / 2) - 0.5
    phi = np.array([_gaussian_footrule(r) for r in r_vals])
    return xi, phi

def get_upper_frechet_xi_footrule(n_points: int = 100):
    xi  = np.linspace(0, 1, n_points)
    phi = np.sqrt(xi)
    return xi, phi

def get_lower_frechet_xi_footrule(n_points: int = 100):
    """C_θ = (1−θ)Π + θW,  ξ = θ²,  ψ = −θ/2."""
    theta = np.linspace(0, 1, n_points)
    return theta**2, -0.5 * theta

def get_jensen_lower_bound(n_points: int = 100):
    v1  = np.linspace(0.5, 1.0, n_points)
    phi = -2*v1**2 + 6*v1 - 5 + (1/v1)
    xi  = -4*v1**2 + 20*v1 - 17 + (2/v1) - (1/v1**2) - 12*np.log(v1)
    return xi, phi


# ── C_μ family (lower boundary arc) ──────────────────────────────────────────

def _psi_s(s, alpha, beta):
    if alpha >= 0.5:
        return (1 - beta) if s > 0.5 else 0.0
    if   s <= alpha:      return 0.0
    elif s <  1 - alpha:  return ((1-beta)/(1-2*alpha)) * (s - alpha)
    else:                 return 1 - beta

def _F_T_inv(y, alpha, beta):
    if beta >= 1.0:
        return y
    a     = (1 - alpha) / (1 - beta)
    denom = (1 - beta)**2
    kappa = (1 - 2*alpha) / denom
    c     = (1 - 2*beta + 2*alpha*beta) / denom
    u1    = a*beta - 0.5*kappa*beta**2
    res   = np.zeros_like(y)
    if abs(kappa) < 1e-7:
        m1, m2, m3 = y < u1, (y >= u1) & (y <= 1-u1), y > 1-u1
        res[m1] = y[m1] / a
        res[m2] = beta + (y[m2] - u1) / c
        res[m3] = 1 - (1 - y[m3]) / a
    else:
        m1, m2, m3 = y < u1, (y >= u1) & (y <= 1-u1), y > 1-u1
        v1 = np.maximum(0, a**2 - 2*kappa*y[m1])
        res[m1] = (a - np.sqrt(v1)) / kappa
        res[m2] = beta + (y[m2] - u1) / c
        v3 = np.maximum(0, a**2 - 2*kappa*(1 - y[m3]))
        res[m3] = 1 - (a - np.sqrt(v3)) / kappa
    return res

def _calc_cmu_point(mu, grid_size=200):
    alpha = 0.15*mu if mu <= 2 else 0.5 - 0.4/mu
    beta  = 0.25*mu if mu <= 2 else 0.5
    alpha = min(alpha, 0.49999)
    beta  = min(beta,  0.99900)

    def C_uu(u_val):
        t   = _F_T_inv(np.array([u_val]), alpha, beta)[0]
        res, _ = quad(lambda s: min(beta, max(0, t - _psi_s(s, alpha, beta))),
                      0, u_val, points=[alpha, 1-alpha])
        return (u_val*t - res) / (1 - beta)

    int_psi, _ = quad(C_uu, 0, 1, points=[alpha, 1-alpha])
    psi = 6*int_psi - 2

    u_g = np.linspace(0.5/grid_size, 1 - 0.5/grid_size, grid_size)
    v_g = np.linspace(0.5/grid_size, 1 - 0.5/grid_size, grid_size)
    T   = _F_T_inv(v_g, alpha, beta)[:, np.newaxis]
    P   = np.array([_psi_s(u, alpha, beta) for u in u_g])[np.newaxis, :]
    term = np.minimum(beta, np.maximum(0, T - P))
    xi   = 6 * np.mean(((T - term) / (1 - beta))**2) - 2
    return xi, psi

def get_cmu_curve():
    mus  = np.concatenate([np.linspace(0, 2, 15), np.linspace(2.5, 10, 10)])
    print("Calculating C_μ curve (this may take a moment)…")
    pts  = [_calc_cmu_point(m) for m in mus]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


# ── Configuration ─────────────────────────────────────────────────────────────

COLORS = {
    "Gaussian":    "black",
    "Clayton":     "#d62728",
    "Frank":       "#2ca02c",
    "GumbelHougaard": "#ff7f0e",
    "Joe":         "#9467bd",
    "JensenBound": "#bcbd22",
    "LowerFrechet": "gray",
    "CMu":         "blue",
}
LABELS = {
    "GumbelHougaard": "Gumbel",
    "LowerFrechet":  r"Lower Fréchet",
}
FAMILIES = ["Clayton", "Frank", "Joe"]
PARAM_RANGES = {"Clayton": (None, 0), "Frank": (None, 0)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir = find_data_dir(__file__)

    plt.figure(figsize=(9, 7))

    # Analytic curves
    xi_jl, phi_jl = get_jensen_lower_bound()
    plt.plot(xi_jl, phi_jl, color=COLORS["JensenBound"], lw=2.5,
             label=r"$C^{\searrow}_\mu$")

    xi_cm, phi_cm = get_cmu_curve()
    plt.plot(xi_cm, phi_cm, color=COLORS["CMu"], lw=2.5, label=r"$C_\mu$")

    xi_lf, phi_lf = get_lower_frechet_xi_footrule()
    plt.plot(xi_lf, phi_lf, color=COLORS["LowerFrechet"], lw=2.5,
             label=LABELS["LowerFrechet"])

    print("Calculating Gaussian…")
    xi_g, phi_g = get_gaussian_xi_footrule(n_points=30)
    plt.plot(xi_g, phi_g, color=COLORS["Gaussian"], lw=2.5, label="Gaussian")

    # Empirical families
    for fam in FAMILIES:
        pmin, pmax = PARAM_RANGES.get(fam, (None, None))

        def _mask(x, phi, pmin=pmin, pmax=pmax):
            return np.ones_like(x, dtype=bool)   # param range handled below

        xi_raw, phi_raw = load_family_data(
            fam, data_dir,
            key1="chatterjees_xi", key2="spearmans_footrule",
        )
        if xi_raw is None:
            # fallback key name
            xi_raw, phi_raw = load_family_data(fam, data_dir,
                                               key1="chatterjees_xi", key2="phi")
        if xi_raw is None or len(xi_raw) == 0:
            continue

        lbl = LABELS.get(fam, fam)
        c   = COLORS.get(fam, "gray")
        idx = np.argsort(xi_raw)
        x, y = xi_raw[idx], phi_raw[idx]
        window = 21
        if len(x) > window:
            try:
                y = savgol_filter(y, window, 3)
            except Exception:
                pass
        plt.plot(x, y, label=lbl, color=c, lw=2.5)

    plt.xlim(0, 1)
    plt.ylim(-0.55, 0.05)
    plt.xlabel(r"Chatterjee's rank correlation $\xi$", fontsize=12)
    plt.ylabel(r"Spearman's footrule $\psi$", fontsize=12)
    plt.title(r"$(\xi, \psi)$ for different copula families", fontsize=14)
    plt.legend(loc="upper right", frameon=True, fontsize=12, ncol=2)
    plt.grid(True, ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("psi_vs_xi_lower_plot.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
