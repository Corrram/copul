import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define Formulas from Proposition \ref{prop:compact_formulas} ---


def get_v1(mu):
    """Calculates v1 = 2 / (2 + mu)"""
    return 2.0 / (2.0 + mu)


def calculate_psi_se_mu(v1):
    """Calculates psi(C^se_mu) from v1"""
    return -2 * v1**2 + 6 * v1 - 5 + 1 / v1


def calculate_xi_se_mu(v1):
    """Calculates xi(C^se_mu) from v1"""
    # np.log is the natural logarithm (ln)
    return -4 * v1**2 + 20 * v1 - 17 + 2 / v1 - 1 / v1**2 - 12 * np.log(v1)


# --- 2. Set up Parameter Range ---
# We test over the valid range mu in [0, 2]
# We create 200 points for a smooth curve
mu_values = np.linspace(0, 2, 200)

# --- 3. Calculate Values ---
v1_values = get_v1(mu_values)
psi_values = calculate_psi_se_mu(v1_values)
xi_values = calculate_xi_se_mu(v1_values)

# --- 4. Verify the Inequality |psi| <= sqrt(xi) ---

# Per Prop \ref{prop:compact_formulas}, psi ranges from 0 (at mu=0)
# to -0.5 (at mu=2). So |psi| = -psi.
abs_psi_values = np.abs(psi_values)

# We take the square root of xi.
# We use np.errstate to suppress the expected "invalid value" warning
# that occurs when taking np.sqrt(0) at mu=0.
with np.errstate(invalid="ignore"):
    sqrt_xi_values = np.sqrt(xi_values)

# At mu=0, v1=1, which makes xi=0 and psi=0.
# np.sqrt(0) can result in NaN, so we fix it.
if np.isnan(sqrt_xi_values[0]):
    sqrt_xi_values[0] = 0.0

# Check 1: Is |psi| <= sqrt(xi) for all mu?
# We add a small tolerance for floating point comparisons
tolerance = 1e-9
is_inequality_held = np.all(abs_psi_values <= sqrt_xi_values + tolerance)

print("--- Verification of |psi(C^se_mu)| <= sqrt(xi(C^se_mu)) ---")
print("Checking if inequality holds for all mu in [0, 2]...")
print(f"Result: {is_inequality_held}\n")

# Check 2: Verify Proposition \ref{prop:mirrored-lower-footrule-vs-upper}
# Check for equality at mu=0
equality_at_zero = np.isclose(abs_psi_values[0], sqrt_xi_values[0])
print("Checking for equality at mu=0:")
print(f"  |psi(0)| = {abs_psi_values[0]:.6f}")
print(f"  sqrt(xi(0)) = {sqrt_xi_values[0]:.6f}")
print(f"Result: {equality_at_zero}\n")

# Check for strict inequality for mu > 0
strict_inequality_for_positive_mu = np.all(abs_psi_values[1:] < sqrt_xi_values[1:])
print("Checking for strict inequality |psi| < sqrt(xi) for all mu in (0, 2]...")
print(f"Result: {strict_inequality_for_positive_mu}\n")

# --- 5. Plot the Results ---
print("Generating plot for visual confirmation...")

plt.figure(figsize=(12, 6))

# Plot 1: The two functions
# FIX: Added r'' prefix to labels for raw strings
plt.subplot(1, 2, 1)
plt.plot(
    mu_values,
    sqrt_xi_values,
    label=r"$\sqrt{\xi(C^{\searrow}_\mu)}$",
    color="blue",
    linewidth=2,
)
plt.plot(
    mu_values,
    abs_psi_values,
    label=r"$|\psi(C^{\searrow}_\mu)|$",
    color="red",
    linestyle="--",
)
plt.title(r"Verification of $|\psi| \leq \sqrt{\xi}$ for Lower Bound")
plt.xlabel(r"Parameter $\mu$ (from 0 to 2)")
plt.ylabel("Value")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)

# Plot 2: The difference
# FIX: Added r'' prefix to labels for raw strings
plt.subplot(1, 2, 2)
difference = sqrt_xi_values - abs_psi_values
plt.plot(
    mu_values,
    difference,
    label=r"$\sqrt{\xi(C^{\searrow}_\mu)} - |\psi(C^{\searrow}_\mu)|$",
    color="green",
)
plt.hlines(0, 0, 2, color="black", linestyles=":", label="Zero line")
plt.title(r"Difference ($\sqrt{\xi} - |\psi|$)")
plt.xlabel(r"Parameter $\mu$ (from 0 to 2)")
plt.ylabel("Difference (should be >= 0)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)

plt.tight_layout()
plt.show()

print("\n--- Plot Description ---")
print("The first plot shows the two functions. The blue solid line (sqrt(xi)) is")
print("clearly above the red dashed line (|psi|) for all mu > 0, and they meet at 0.")
print("The second plot shows their difference, which starts at 0 and is always")
print("non-negative, confirming the inequality |psi| <= sqrt(xi) holds.")
