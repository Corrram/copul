import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

# --- Plot Style Configuration ---
# Use settings that mimic a professional LaTeX document
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (7, 7),
    }
)


def get_alpha_beta_path(mu_values):
    """
    Calculates the continuous path (alpha(mu), beta(mu)) for the new model
    with a transition at mu = 2.
    """
    alphas = np.zeros_like(mu_values)
    betas = np.zeros_like(mu_values)

    # Regime 1: mu <= 2 (Path from (0,0) to (0.3, 0.5))
    mask1 = mu_values <= 2
    alphas[mask1] = 0.15 * mu_values[mask1]
    betas[mask1] = 0.25 * mu_values[mask1]

    # Regime 2: mu > 2 (Path from (0.3, 0.5) towards (0.5, 0.5))
    mask2 = mu_values > 2
    alphas[mask2] = 0.5 - 0.4 / mu_values[mask2]
    betas[mask2] = 0.5

    return alphas, betas


# --- Data Generation ---
# Start mu slightly above zero to avoid log(0) issues.
mu = np.linspace(0.01, 15, 1000)
alphas, betas = get_alpha_beta_path(mu)

# --- Plotting ---
fig, ax = plt.subplots()

# --- Create a Gradient-Colored Line ---
points = np.array([alphas, betas]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Use LogNorm for a logarithmic color scale.
lc = LineCollection(
    segments,
    cmap="cividis",
    norm=colors.LogNorm(vmin=mu[0], vmax=2.5),  # vmax=2.5 still works well
)

mu_midpoints = (mu[:-1] + mu[1:]) / 2
lc.set_array(mu_midpoints)
lc.set_linewidth(2.5)
line = ax.add_collection(lc)

# --- Add Color Bar ---
cbar = fig.colorbar(line, ax=ax, shrink=0.8)
cbar.set_label(r"Parameter $\mu$", rotation=270, labelpad=20)

# --- Mark Key Points ---
# Starting point at mu = 0 (Circle)
ax.plot(
    0.0,
    0.0,
    "o",
    markersize=8,
    color="white",
    markeredgecolor="black",
    label=r"$\mu = 0$: Start",
)
# Transition point at mu = 2 (Square)
ax.plot(
    0.3,
    0.5,
    "s",
    markersize=8,
    color="white",
    markeredgecolor="black",
    label=r"$\mu = 2$: Transition",
)

# --- Plot Formatting ---
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel(r"$\beta$")
ax.set_title(
    r"Visualization of the Parameter Path $(\alpha(\mu), \beta(\mu))$",
    fontsize=16,
    pad=15,
)
ax.set_xlim(-0.02, 0.52)
ax.set_ylim(-0.02, 0.52)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="lower right")
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(f"images/mu_path_visualizer.png", dpi=300, bbox_inches="tight")
plt.show()
