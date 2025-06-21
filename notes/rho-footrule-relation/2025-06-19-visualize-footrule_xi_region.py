import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------- envelope √ξ
xi_grid = np.linspace(0.0, 1.0, 400)
psi_up  = np.sqrt(xi_grid)
psi_lo  = -psi_up

# -------------------------------------------------- optimum of  ψ−ξ
b_star   = 0.5              # derivative 1−2b=0
xi_star  = b_star ** 2      # = 0.25
psi_star = b_star           # = 0.5
gain_max = psi_star - xi_star   # = 0.25

# -------------------------------------------------- figure
BLUE = "#00529B"
FILL = "#D6EAF8"
RED  = "#C70039"

fig, ax = plt.subplots(figsize=(6, 8))

ax.plot(xi_grid,  psi_up, color=BLUE, lw=2.5, label=r"$\psi_{\max}=\gamma_{\max}$")
ax.plot(xi_grid,  psi_lo, color=BLUE, lw=2.5)
ax.fill_between(xi_grid, psi_lo, psi_up,
                color=FILL, alpha=0.7, label="attainable")

# ---------- optimal point  (xi*, psi*)
ax.scatter([xi_star], [psi_star], s=90,
           color=RED, zorder=6, label=r"$b^{\star}=0.5$")
ax.annotate(r"$(\xi^{\star},\psi^{\star})$",
            (xi_star, psi_star),
            xytext=(12, 8), textcoords="offset points",
            color=RED, fontsize=14, ha="left", va="bottom")

# ---------- independence & symmetry point for reference
ax.scatter([0], [0], s=60, color="black")
ax.annotate(r"$\Pi$", (0, 0),
            xytext=(10, 0), textcoords="offset points",
            fontsize=18, ha="left", va="center")

# ---------- axes / grid / legend
ax.set_xlabel(r"Chatterjee's $\xi$", fontsize=16)
ax.set_ylabel(r"Spearman's foot-rule $\psi$ (= Gini’s $\gamma$)",
              fontsize=16)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
ax.tick_params(axis="both", labelsize=13)
ax.grid(True, linestyle=":", alpha=0.6)
ax.axhline(0, color="black", lw=0.8)

ax.legend(loc="upper left", fontsize=11)
fig.tight_layout()
plt.show()
