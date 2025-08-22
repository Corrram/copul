import numpy as np
from scipy.optimize import minimize_scalar


def calculate_correlations(mu):
    """
    Calculates psi and xi for a given mu based on the formulas
    from Proposition 3.2.
    """
    if mu < 0:
        return float("inf"), float("inf")  # Invalid domain for mu

    # Handle the mu=0 case to avoid division by zero
    if mu == 0:
        v1 = 1.0
        # For mu=0, the copula is Pi, so psi=0, xi=0
        return 0.0, 0.0

    v1 = 2 / (2 + mu)

    # Formula for psi
    psi = -2 * v1**2 + 6 * v1 - 5 + (1 / v1)

    # Formula for xi
    # The term (6*v1**2 - 4*v1**3 - 1) / mu**2 can be tricky.
    # We substitute mu = (2/v1) - 2 to make it robust.
    mu_from_v1 = (2 / v1) - 2

    xi = (
        4 * v1**3
        - 18 * v1**2
        + 36 * v1
        - 22
        - 12 * np.log(v1)
        + (6 * v1**2 - 4 * v1**3 - 1) / mu_from_v1**2
    )

    return psi, xi


def objective_function(mu):
    """
    The function we want to minimize, which is the sum of psi and xi.
    """
    psi, xi = calculate_correlations(mu)
    return psi + xi


# --- Main execution ---
# Find the minimum of the objective function within the valid bounds [0, 2]
result = minimize_scalar(objective_function, bounds=(0, 2), method="bounded")

# Extract the optimal mu and the minimum value of the sum
optimal_mu = result.x
min_sum = result.fun

# Calculate the corresponding psi and xi at the optimal mu
final_psi, final_xi = calculate_correlations(optimal_mu)

# --- Print the results ---
print("=" * 45)
print("🔎 Minimization Results for ψ(μ) + ξ(μ) 🔎")
print("=" * 45)
print(f"  Optimal Parameter (μ): {optimal_mu:.4f}")
print(f"  Spearman's Footrule (ψ): {final_psi:.4f}")
print(f"  Chatterjee's Correlation (ξ): {final_xi:.4f}")
print("-" * 35)
print(f"  Minimum Sum (ψ + ξ):  {min_sum:.4f}")
print("=" * 45)
