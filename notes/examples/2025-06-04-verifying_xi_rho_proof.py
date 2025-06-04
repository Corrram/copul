import numpy as np
from scipy.integrate import quad
import sympy as sp

# --- Define the integrand J_v (numeric) ---
def J_v_integrand(v, b_val):
    """
    Calculates J_v = 1/2 - (1-v)*sqrt(2*(1-v)/b_val)/3
    """
    if b_val <= 0:
        raise ValueError("b must be positive.")
    term_sqrt = np.sqrt(2 * (1 - v) / b_val)
    return 0.5 - ((1 - v) * term_sqrt) / 3

# --- Define the two candidate formulas for I_3(b) ---
def I3b_candidate1(b_val):
    """Candidate 1: (15*b - 2*b^2) / 60"""
    return (15 * b_val - 2 * b_val**2) / 60

def I3b_candidate2(b_val):
    """Candidate 2: (15*b + 2*b^2) / 60"""
    return (15 * b_val + 2 * b_val**2) / 60

# --- Symbolic computation with SymPy ---
v, b = sp.symbols('v b', positive=True, real=True)
Jv_sym = sp.Rational(1, 2) - (1 - v) * sp.sqrt(2 * (1 - v) / b) / 3

# Integrate Jv_sym with respect to v from (1 - b/2) to 1
I3b_sym = sp.integrate(Jv_sym, (v, 1 - b/2, 1))
I3b_sym_simplified = sp.simplify(I3b_sym)

# Create a numeric function from the symbolic result
I3b_sym_func = sp.lambdify(b, I3b_sym_simplified, 'numpy')

print("Symbolic result for I_3(b):")
sp.pprint(I3b_sym_simplified)
print("\n")

# --- Values of b to test ---
b_values_to_test = [1.0, 0.8, 0.5, 0.25, 0.1]

print("--- Numerical vs. Symbolic Check for I_3(b) ---")
print("Integrand J_v = 1/2 - (1-v)*sqrt(2*(1-v)/b)/3")
print("Integral I_3(b) = integral_{1-b/2}^{1} J_v dv\n")

for b_test in b_values_to_test:
    print(f"--- Testing for b = {b_test:.4f} ---")
    lower_limit = 1 - b_test / 2
    upper_limit = 1

    # Numerical integration
    numerical_result, numerical_error = quad(J_v_integrand, lower_limit, upper_limit, args=(b_test,))
    print(f"Numerical integration result: {numerical_result:.8f} (error estimate: {numerical_error:.2e})")

    # Evaluate candidates
    candidate1_value = I3b_candidate1(b_test)
    candidate2_value = I3b_candidate2(b_test)
    print(f"Candidate 1 ((15b - 2b^2)/60): {candidate1_value:.8f}")
    print(f"Candidate 2 ((15b + 2b^2)/60): {candidate2_value:.8f}")

    # Symbolic evaluation
    symbolic_value = I3b_sym_func(b_test)
    print(f"Symbolic integral value: {symbolic_value:.8f}")

    # Compute differences
    diff_num_cand1 = abs(numerical_result - candidate1_value)
    diff_num_cand2 = abs(numerical_result - candidate2_value)
    diff_num_sym = abs(numerical_result - symbolic_value)

    print(f"Difference (numerical vs. candidate 1): {diff_num_cand1:.2e}")
    print(f"Difference (numerical vs. candidate 2): {diff_num_cand2:.2e}")
    print(f"Difference (numerical vs. symbolic):   {diff_num_sym:.2e}")

    # Conclusion
    if diff_num_sym < 1e-7:
        print(f"Conclusion: Symbolic result matches numerical for b = {b_test:.4f}")
    else:
        print(f"Conclusion: Symbolic result does NOT match numerical for b = {b_test:.4f}")

    if diff_num_cand1 < 1e-7:
        print(f"            Candidate 1 is also CORRECT for b = {b_test:.4f}")
    if diff_num_cand2 < 1e-7:
        print(f"            Candidate 2 is CORRECT for b = {b_test:.4f}")
    print()

print("--- Script Finished ---")
