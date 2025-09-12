import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid
from tqdm import tqdm
import pandas as pd


# --- Your provided functions for the copula construction ---
# (These functions remain unchanged)
def spearmans_footrule(s, alpha, beta):
    s = np.asarray(s)
    ps_val = np.zeros_like(s, dtype=float)
    mask_middle = (s > alpha) & (s < 1 - alpha)
    mask_upper = s >= (1 - alpha)
    if alpha < 0.5:
        slope = (1 - beta) / (1 - 2 * alpha)
        ps_val[mask_middle] = slope * (s[mask_middle] - alpha)
    ps_val[mask_upper] = 1 - beta
    return ps_val


def get_L_t(t_grid, alpha, beta, n_points=2000):
    s_fine = np.linspace(0, 1, n_points)
    psi_vals = psi(s_fine, alpha, beta)
    is_in_hole = (psi_vals[np.newaxis, :] <= t_grid[:, np.newaxis]) & (
        psi_vals[np.newaxis, :] + beta >= t_grid[:, np.newaxis]
    )
    return trapezoid(is_in_hole.astype(float), s_fine, axis=1)


def construct_diagonal_copula(u_grid, v_grid, alpha, beta):
    if alpha >= 0.5 or beta >= 1:
        raise ValueError("Parameters must satisfy alpha < 0.5 and beta < 1")
    t_base_pts = v_grid[:, 0]
    L_t = get_L_t(t_base_pts, alpha, beta)
    f_T = (1 - L_t) / (1 - beta)
    F_T = cumulative_trapezoid(f_T, t_base_pts, initial=0)
    F_T = np.maximum.accumulate(F_T)
    t_mapped = np.interp(v_grid, F_T, t_base_pts)
    s_grid = u_grid
    psi_vals_mapped = psi(s_grid, alpha, beta)
    is_in_hole = (t_mapped >= psi_vals_mapped) & (t_mapped <= psi_vals_mapped + beta)
    f_T_mapped = np.interp(t_mapped, t_base_pts, f_T)
    density = (1 / (1 - beta)) * (1 / (f_T_mapped + 1e-9))
    density[is_in_hole] = 0
    return density


def calculate_xi_psi_grid(alpha, beta, n_points=500):
    u = np.linspace(0, 1, n_points)
    v = np.linspace(0, 1, n_points)
    V, U = np.meshgrid(v, u, indexing="ij")
    Z = construct_diagonal_copula(U, V, alpha, beta)
    H = cumulative_trapezoid(Z, v, initial=0, axis=0)
    C_matrix = cumulative_trapezoid(H, u, initial=0, axis=1)
    C_diagonal = np.diag(C_matrix)
    integral_psi = trapezoid(C_diagonal, u)
    spearman_psi = 6 * integral_psi - 2
    xi_integrand = H**2
    integral_xi = trapezoid(trapezoid(xi_integrand, v, axis=0), u)
    chatterjee_xi = 6 * integral_xi - 2
    return chatterjee_xi, spearman_psi


# --- Main script to trace the boundary ---

print("Computing final high-precision boundary with adaptive resolution...")
NUM_POINTS_PER_PATH = 50

# Path 1: Corresponds to mu >= 2 (where beta=0.5)
results_path1 = []
# Add the endpoint for mu -> infinity, where alpha=0.5
results_path1.append((np.inf, 0.5, -0.5))

# This path goes from alpha=0.5 (mu=inf) down to the transition point alpha=0.3 (mu=2)
alphas_path1 = np.linspace(0.5, 0.3, NUM_POINTS_PER_PATH)[1:]
for alpha in tqdm(alphas_path1, desc="Path 1 (mu >= 2)"):
    # Calculate mu from alpha using the relation: alpha = 0.5 - 0.4 / mu
    mu = 0.4 / (0.5 - alpha + 1e-12)

    # --- FIX: Renamed variable from 'psi' to 'psi_val' ---
    xi_val, psi_val = calculate_xi_psi_grid(alpha, 0.5, n_points=500)
    results_path1.append((mu, xi_val, psi_val))

# Path 2: Corresponds to mu <= 2 (where alpha = 0.6*beta)
# This path goes from the transition point (beta=0.5, mu=2) down to beta=0 (mu=0)
betas_path2 = np.linspace(0.5, 0, NUM_POINTS_PER_PATH)
results_path2 = []
for beta in tqdm(betas_path2, desc="Path 2 (mu <= 2)"):
    alpha = 0.6 * beta
    if alpha >= 0.5:
        alpha = 0.4999

    # Calculate mu from beta using the relation: beta = 0.25 * mu
    mu = 4 * beta

    grid_res = 5000

    # --- FIX: Renamed variable from 'psi' to 'psi_val' ---
    xi_val, psi_val = calculate_xi_psi_grid(alpha, beta, n_points=grid_res)
    results_path2.append((mu, xi_val, psi_val))

# Combine, sort, and save results
df_path1 = pd.DataFrame(results_path1, columns=["mu", "xi", "psi"])
df_path2 = pd.DataFrame(results_path2, columns=["mu", "xi", "psi"])

df = (
    pd.concat([df_path1, df_path2])
    .drop_duplicates(subset=["xi", "psi"])
    .sort_values(by="mu")
    .reset_index(drop=True)
)

output_filename = "lower_boundary_final_smooth.csv"
df.to_csv(output_filename, index=False)
print(f"\n✅ Final high-precision data (with mu) saved to '{output_filename}'")

# --- Plotting ---
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(
    df["xi"],
    df["psi"],
    "-",
    color="crimson",
    linewidth=2.5,
    label="Conjectured Lower Boundary",
    zorder=5,
)
x_ref = np.linspace(0, 1, 5000)
ax.plot(
    x_ref,
    np.sqrt(x_ref),
    color="royalblue",
    linestyle="--",
    label=r"Upper Bound: $\psi = \sqrt{\xi}$",
)
ax.set_title(r"Final High-Precision Lower Boundary for $(\xi, \psi)$", fontsize=16)
ax.set_xlabel(r"Chatterjee's $\xi$", fontsize=12)
ax.set_ylabel(r"Spearman's Footrule $\psi$", fontsize=12)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.6, 1.05)
ax.legend()
ax.set_aspect("equal", adjustable="box")
plt.grid(True)
plt.savefig("images/lower_boundary_final_precise.png", dpi=300)
plt.show()
