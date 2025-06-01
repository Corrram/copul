import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def calculate_xi_opt_for_spearman_rho(rho_s_input):
    """
    Calculates the optimal Chatterjee's xi (xi_opt) for a given Spearman's rho (rho_s_input).
    rho_s_input is Spearman's rho. The calculation uses formulas derived for abs(rho_s) in [0, 1].
    Thus, effectively computes xi_opt(|rho_s_input|).
    """
    
    # Handle inputs outside [-1,1] by returning NaN for robustness, though we'll plot for [-1,1]
    if not (-1 - 1e-9 <= rho_s_input <= 1 + 1e-9):
        return np.nan
        
    rho_s_calc = abs(rho_s_input) # Use absolute value for calculation based on [0,1] model
    rho_s_calc = np.clip(rho_s_calc, 0, 1) # Ensure strictly within [0,1] for calculations

    # Case: rho_s_calc = 0 (Independence)
    if np.isclose(rho_s_calc, 0):
        b_star = 0.0
        # xi_opt from xi_val,B(b) = b^2/2 - b^3/5
        xi_opt_val = (b_star**2 / 2.0) - (b_star**3 / 5.0) # Will be 0
        return xi_opt_val

    # Case: rho_s_calc = 1 (Frechet-Hoeffding Upper Bound for positive dependence)
    if np.isclose(rho_s_calc, 1.0):
        # b_star -> infinity conceptually
        # xi_opt from xi_val,A(b) = 1 - 1/b + 3/(10b^2)
        xi_opt_val = 1.0 
        return xi_opt_val

    # Case: rho_s_calc = 3/10 (Transition point)
    rho_transition = 3.0/10.0
    if np.isclose(rho_s_calc, rho_transition):
        b_star = 1.0
        # Can use either formula, e.g., from Regime 1 (xi_val,B)
        xi_opt_val = (b_star**2 / 2.0) - (b_star**3 / 5.0) # 0.5 - 0.2 = 0.3
        return xi_opt_val

    # Regime 1: rho_s_calc in (0, 3/10)
    # (Spearman constraint: 12K_B(b) - 3 = rho_s_calc)
    # (K_B(b) = b^2/24 - b^3/60 + 1/4)
    # Equation for b_star: 2*b^3 - 5*b^2 + 10*rho_s_calc = 0
    if 0 < rho_s_calc < rho_transition:
        def cubic_func_b(b, x_val): # x_val here is rho_s_calc
            return 2*b**3 - 5*b**2 + 10*x_val
        try:
            # f(0) = 10*rho_s_calc > 0. f(1) = 10*rho_s_calc - 3 < 0. Root is in (0,1).
            b_star = brentq(cubic_func_b, 0, 1, args=(rho_s_calc,)) # brentq should handle endpoints if they are roots
        except ValueError as e:
            # This might happen if due to float precision f(0)*f(1) is not < 0 near boundaries
            # Fallback or print warning
            print(f"Warning: brentq failed for rho_s_calc = {rho_s_calc} in (0, {rho_transition}). Error: {e}")
            # Try numpy.roots as a fallback for debugging, then select
            coeffs = [2, -5, 0, 10*rho_s_calc]
            roots = np.roots(coeffs)
            b_star_candidate = [r.real for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1]
            if not b_star_candidate: return np.nan
            b_star = b_star_candidate[0]

        # xi_opt from xi_val,B(b) = b^2/2 - b^3/5
        xi_opt_val = (b_star**2 / 2.0) - (b_star**3 / 5.0)
        return xi_opt_val

    # Regime 2: rho_s_calc in (3/10, 1)
    # (Spearman constraint: 12K_A(b) - 3 = rho_s_calc)
    # (K_A(b) = 1/3 - 1/(12b) + 1/(40b^2))
    # Equation for b_star: (10-10*rho_s_calc)*b^2 - 10*b + 3 = 0
    elif rho_transition < rho_s_calc < 1:
        discriminant_val = 120*rho_s_calc - 20 
        
        if discriminant_val < -1e-9: # Allow small negative due to precision
            print(f"Warning: Negative discriminant {discriminant_val} for rho_s_calc = {rho_s_calc} in ({rho_transition}, 1)")
            return np.nan
        discriminant_val = max(0, discriminant_val) # Ensure non-negative for sqrt

        denominator_val = 20 * (1 - rho_s_calc) # Positive for rho_s_calc < 1
        if np.isclose(denominator_val, 0): # Should be caught by rho_s_calc < 1
             print(f"Warning: Denominator near zero for rho_s_calc = {rho_s_calc}")
             return np.nan

        b_star = (10 + np.sqrt(discriminant_val)) / denominator_val
        
        # xi_opt from xi_val,A(b) = 1 - 1/b + 3/(10b^2)
        xi_opt_val = 1.0 - (1.0 / b_star) + (3.0 / (10.0 * b_star**2))
        return xi_opt_val
        
    else: 
        # This path should ideally not be taken if rho_s_calc is one of the explicitly checked boundary values.
        # It might be hit if rho_s_calc is slightly outside [0,1] due to input linspace inaccuracies
        # before the np.clip, or if a category was missed.
        print(f"Warning: rho_s_calc = {rho_s_calc} (from original input {rho_s_input}) did not fall into expected analytic regimes after explicit checks.")
        return np.nan

# Generate x (Spearman's rho) values for plotting
spearman_rho_inputs = np.linspace(-1.0, 1.0, 401) 

differences = []
valid_spearman_inputs_for_plot = []

for x_plot_val in spearman_rho_inputs:
    # calculate_xi_opt_for_spearman_rho internally computes xi_opt(|x_plot_val|)
    xi_calculated_val = calculate_xi_opt_for_spearman_rho(x_plot_val) 
    
    if not np.isnan(xi_calculated_val):
        difference = x_plot_val - xi_calculated_val 
        differences.append(difference)
        valid_spearman_inputs_for_plot.append(x_plot_val)

# Plotting: (Spearman's rho, Difference)
plt.figure(figsize=(9, 6))
plt.plot(valid_spearman_inputs_for_plot, differences, linestyle='-', color='dodgerblue')

# For clarity, plot the theoretical expectation
x_theory_neg = np.linspace(-1, 0, 50)
y_theory_neg = 2 * x_theory_neg
x_theory_pos = np.linspace(0, 1, 50)
y_theory_pos = 0 * x_theory_pos
plt.plot(x_theory_neg, y_theory_neg, color='red', linestyle='--', linewidth=2, label=r'Expected: $2x$ for $x \in [-1,0]$')
plt.plot(x_theory_pos, y_theory_pos, color='red', linestyle='--', linewidth=2, label=r'Expected: $0$ for $x \in [0,1]$')


plt.xlabel(r"Given Spearman's $\rho_S$ ($x$)")
plt.ylabel(r"Difference: $x - \xi_{opt}(|x|)$")
plt.title(r"Difference between Spearman's $\rho_S$ and Maximal Chatterjee's $\xi_{opt}$")
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.7)
plt.axvline(0, color='black', linewidth=0.7)

# Adjust y-limits based on expected range
min_expected_diff = -2.0
max_expected_diff = 0.0
plt.ylim([min_expected_diff - 0.1, max_expected_diff + 0.1])
plt.xlim([-1.05, 1.05])
plt.legend()
plt.show()