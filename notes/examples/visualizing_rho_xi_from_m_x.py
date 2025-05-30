import numpy as np
import matplotlib.pyplot as plt

# Apply a style for a cleaner look
plt.style.use("seaborn-v0_8-whitegrid")
# You can also try to set a global font if you have a specific preference
# and know it's available on your system, e.g.:
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial'] # Or 'DejaVu Sans', 'Helvetica' etc.


# --- Define b(x) functions (same as before) ---
def b_from_x_regime1(x_val):
    """
    Calculates b(x) for x in (3/10, 1).
    This is for b > 1.
    """
    if x_val == 1.0:  # As x -> 1, b -> infinity
        return np.inf
    if x_val <= 3 / 10:  # Should be handled by regime 2
        if np.isclose(x_val, 3 / 10):
            return 1.0
        return np.nan

    discriminant_term = 5 * (6 * x_val - 1)
    numerator = 5 + np.sqrt(discriminant_term)
    denominator = 10 * (1 - x_val)
    if denominator == 0:
        return np.inf
    return numerator / denominator


def b_from_x_regime2(x_val):
    """
    Calculates b(x) for x in (0, 3/10].
    This is for 0 < b <= 1.
    """
    if x_val == 0:
        return 0.0
    if x_val > 3 / 10:
        if np.isclose(x_val, 3 / 10):
            return 1.0
        return np.nan

    term_inside_arccos = 1 - (108 / 25) * x_val
    term_inside_arccos = np.clip(term_inside_arccos, -1.0, 1.0)

    theta = (1 / 3) * np.arccos(term_inside_arccos)
    cos_term = np.cos(theta - (2 * np.pi / 3))

    b_val = (5 / 6) + (5 / 3) * cos_term
    return np.clip(b_val, 0.0, 1.0)


# --- Define M_x (upper bound for rho) with corrected formula (same as before) ---
def M_x_upper_bound_corrected(x_val):
    """
    Calculates M_x based on the *corrected* derived formulas.
    x_val here represents xi.
    """
    if x_val < 0 or x_val > 1:
        return np.nan
    if np.isclose(x_val, 0):
        return 0.0
    if np.isclose(x_val, 1):
        return 1.0

    x_threshold = 3 / 10

    if x_val < x_threshold and not np.isclose(x_val, x_threshold):
        b_val = b_from_x_regime2(x_val)
        m_x = b_val - (3 * b_val**2) / 10
    elif x_val > x_threshold and not np.isclose(x_val, x_threshold):
        b_val = b_from_x_regime1(x_val)
        if b_val == np.inf:
            m_x = 1.0
        elif np.isnan(b_val) or b_val == 0:
            m_x = np.nan
        else:
            m_x = 1 - (1 / (2 * b_val**2)) + (1 / (5 * b_val**3))
            # m_x = 12*(1/3-1/(24*b_val**2)+1/(60*b_val**3)) -3
    elif np.isclose(x_val, x_threshold):
        b_val = 1.0
        m_x = b_val - (3 * b_val**2) / 10
    else:
        return np.nan
    return m_x


# --- Generate xi values for plotting (same as before) ---
epsilon = 1e-9
xi_coords = np.linspace(0.0, 3 / 10 - epsilon, 150)
xi_coords = np.append(xi_coords, np.linspace(3 / 10 - epsilon, 3 / 10 + epsilon, 50))
xi_coords = np.append(xi_coords, np.linspace(3 / 10 + epsilon, 1.0, 150))
xi_coords_full = np.unique(np.clip(xi_coords, 0.0, 1.0))

# --- Calculate rho_upper (M_x) and rho_lower (mirrored) values (same as before) ---
rho_upper_corrected = np.array([M_x_upper_bound_corrected(xi) for xi in xi_coords_full])
valid_indices = ~np.isnan(rho_upper_corrected)
xi_coords_valid = xi_coords_full[valid_indices]
rho_upper_valid = rho_upper_corrected[valid_indices]
rho_lower_mirrored = -rho_upper_valid

# --- Create the plot ---
fig, ax = plt.subplots(
    figsize=(6, 8)
)  # Adjusted figsize to be taller, closer to example

# Plot the corrected M_x upper bound
ax.plot(
    xi_coords_valid,
    rho_upper_valid,
    color="#00529B",
    linewidth=2.5,
    label=r"$\pm M_\xi$"        # Gemeinsamer Label‐Eintrag
)
ax.plot(
    xi_coords_valid,
    rho_lower_mirrored,
    color="#00529B",
    linewidth=2.5               # Solid line statt '--'
)

# Mark key points (without text labels as in example)
key_points_xi = [0, 1, 0]
key_points_rho = [0, 1, -1]
ax.scatter(
    key_points_xi, key_points_rho, color="black", s=60, zorder=5
)  # Slightly smaller points

# Fill the attainable region between -M_x and M_x
ax.fill_between(
    xi_coords_valid,
    rho_lower_mirrored,
    rho_upper_valid,
    where=rho_upper_valid >= rho_lower_mirrored,
    color="#D6EAF8",
    alpha=0.7,
    interpolate=True,
    label=r"Attainable Region",
    zorder=1,
)  # Adjusted color and alpha


# --- Plot settings to match example image ---
# ax.set_title(r"Attainable Region for (Chatterjee's $\xi$, Spearman's $\rho$)", fontsize=18, pad=20, fontweight='bold') # Title removed
ax.set_xlabel(r"Chatterjee's $\xi$", fontsize=16, labelpad=10)
ax.set_ylabel(r"Spearman's $\rho$", fontsize=16, labelpad=10)

ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-1.05, 1.05])

# Customize ticks
ax.tick_params(axis="both", which="major", labelsize=13)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))


# Grid appearance
ax.grid(True, linestyle=":", alpha=0.6, color="darkgray", zorder=0)  # Darker grid

# Axes lines
ax.axhline(0, color="black", linewidth=0.8, zorder=2)
ax.axvline(0, color="black", linewidth=0.8, zorder=2)

# Legend inside the plot
legend = ax.legend(
    loc="center",
    ncol=1,
    fontsize=12,
    frameon=True,
    fancybox=False,
    shadow=False,
    borderpad=0.5,
)
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_facecolor("white")


# ax.set_aspect('equal', adjustable='box') # Removed to allow figsize to dictate shape more like example
fig.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make space for legend and title

plt.show()
