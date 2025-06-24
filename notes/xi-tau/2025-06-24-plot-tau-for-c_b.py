import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# helper: inverse slope  b(xi)
# ------------------------------------------------------------------
def b_from_xi(x):
    """Return b = b_x(x) for an array-like xi=x∈(0,1)."""
    x = np.asarray(x)
    b = np.empty_like(x)

    mask = x <= 0.3                      # first branch 0 < x ≤ 3/10
    # branch 1
    if np.any(mask):
        sqrt6x = np.sqrt(6 * x[mask])
        b[mask] = sqrt6x / (2 * np.cos((1/3) * np.arccos(-3 * sqrt6x / 5)))

    # branch 2   3/10 < x < 1
    mask2 = ~mask
    if np.any(mask2):
        b[mask2] = (5 + np.sqrt(5 * (6 * x[mask2] - 1))) / (10 * (1 - x[mask2]))

    return b


# ------------------------------------------------------------------
# closed‑form τ(b) and ρ(b)  (b > 0)
# ------------------------------------------------------------------
def tau_from_b(b):
    b = np.asarray(b)
    tau = np.where(b <= 1,
                   b * (4 - b) / 6,
                   (6 * b**2 - 4 * b + 1) / (6 * b**2))
    return tau


def rho_from_b(b):
    b = np.asarray(b)
    rho = np.where(b <= 1,
                   b - 3 * b**2 / 10,
                   1 - 1 / (2 * b**2) + 1 / (5 * b**3))
    return rho


# ------------------------------------------------------------------
# dense grid in ξ ∈ (0,1)
# ------------------------------------------------------------------
xi = np.linspace(1e-6, 0.999, 4000)
b  = b_from_xi(xi)

tau = tau_from_b(b)
rho = rho_from_b(b)

# ------------------------------------------------------------------
# plot
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(xi, tau, label=r'$(\xi,\tau)$', linewidth=2)
plt.plot(xi, rho, label=r'$(\xi,\rho)$', linewidth=2)
plt.axvline(0.3, color='k', linestyle='--', alpha=0.3)     # breakpoint ξ=0.3 (b=1)
plt.xlabel(r'$\xi(C_b)$')
plt.ylabel(r'$\tau(C_b)$ / $\rho(C_b)$')
plt.title(r'Full range $\,\xi\in[0,1)$ with $b=b_{\xi}$')
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()
plt.show()
