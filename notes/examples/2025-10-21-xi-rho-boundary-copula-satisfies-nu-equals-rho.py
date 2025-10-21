import sympy as sp

# --- 1. Define Symbols ---
v, b = sp.symbols("v b", real=True, positive=True)
u = sp.symbols("u", real=True)
s, a = sp.symbols("s a", real=True)  # Placeholders for sv, av in integrals

# Use sp.Rational for precision
R_0 = sp.Rational(0)
R_1 = sp.Rational(1)
R_2 = sp.Rational(2)
R_1_2 = sp.Rational(1, 2)
R_1_3 = sp.Rational(1, 3)
R_1_4 = sp.Rational(1, 4)
R_1_5 = sp.Rational(1, 5)
R_1_6 = sp.Rational(1, 6)
R_1_10 = sp.Rational(1, 10)
R_1_12 = sp.Rational(1, 12)
R_3_10 = sp.Rational(3, 10)

# --- 2. Define the Four Inner Integral Formulas I_nu(v) ---

# Indefinite integrals needed
integrand_ramp = b * (s - u) * (1 - u) ** 2
integrand_plateau = (1 - u) ** 2

I_ramp_indef = sp.integrate(integrand_ramp, u)
I_plateau_indef = sp.integrate(integrand_plateau, u)


# Function to evaluate F(upper) - F(lower)
def eval_integral(F_indef, lower, upper):
    # Need to handle potential NaN/complex for sqrt, ensure real results
    # Simplification helps sympy manage expressions
    res = F_indef.subs(u, upper) - F_indef.subs(u, lower)
    # Aggressively simplify, assuming b>0 and 0<=v<=1
    return sp.simplify(res)


# Case 1a (a_v <= 0, s_v <= 1): Ramp from 0 to s_v
I_nu_1a = eval_integral(I_ramp_indef, 0, s)
# Case 1b (a_v <= 0, s_v > 1): Ramp from 0 to 1
I_nu_1b = eval_integral(I_ramp_indef, 0, 1)
# Case 2a (a_v > 0, s_v <= 1): Plateau from 0 to a_v, Ramp from a_v to s_v
I_nu_2a_part1 = eval_integral(I_plateau_indef, 0, a)
I_nu_2a_part2 = eval_integral(I_ramp_indef, a, s)
I_nu_2a = sp.simplify(I_nu_2a_part1 + I_nu_2a_part2)
# Case 2b (a_v > 0, s_v > 1): Plateau from 0 to a_v, Ramp from a_v to 1
I_nu_2b_part1 = eval_integral(I_plateau_indef, 0, a)
I_nu_2b_part2 = eval_integral(I_ramp_indef, a, 1)
I_nu_2b = sp.simplify(I_nu_2b_part1 + I_nu_2b_part2)

# --- 3. Define rho(C_b^xi,rho) and xi(C_b^xi,rho) ---
rho_small_b = b - R_3_10 * b**2
rho_large_b = 1 - R_1_2 / b**2 + R_1_5 / b**3
xi_small_b = b**2 / 10 * (5 - 2 * b)
xi_large_b = 1 - 1 / b + 3 / (10 * b**2)

# --- 4. Case 0 < b <= 1 ---
print("--- Running Case 0 < b <= 1 ---")
# Define integration limits for v
v_lim1_s = b / R_2
v_lim2_s = R_1 - b / R_2

# Define s_v pieces from eq (4.22)
sv_p1_s = sp.sqrt(R_2 * v / b)
sv_p2_s = v / b + R_1_2
sv_p3_s = R_1 + R_1 / b - sp.sqrt(R_2 * (R_1 - v) / b)

# Define a_v pieces
av_p1_s = sv_p1_s - R_1 / b
av_p2_s = sv_p2_s - R_1 / b
av_p3_s = sv_p3_s - R_1 / b

# Define I_nu(v) using Piecewise based on conditions derived in thought process
I_nu_v_small = sp.Piecewise(
    (I_nu_1a.subs(s, sv_p1_s), v <= v_lim1_s),  # Range 1: a_v <= 0, s_v <= 1 -> Use 1a
    (
        I_nu_1b.subs(s, sv_p2_s),
        v <= v_lim2_s,
    ),  # Range 2: a_v <= 0, s_v > 1 -> Use 1b
    (
        I_nu_2b.subs({s: sv_p3_s, a: av_p3_s}),
        v > v_lim2_s,
    ),  # Range 3: a_v > 0, s_v > 1 -> Use 2b
)

print("Calculating sub-integrals for 0 < b <= 1...")
# Sub-integral 1: v from 0 to v_lim1_s (b/2)
I_nu_s_func1 = I_nu_v_small.args[0][0]
I_nu_s_val1 = sp.integrate(I_nu_s_func1, (v, R_0, v_lim1_s))
I_nu_s_val1_simp = sp.ratsimp(I_nu_s_val1)
print(f"  Sub-integral 1 (v=0 to b/2) = {I_nu_s_val1_simp}")

# Sub-integral 2: v from v_lim1_s (b/2) to v_lim2_s (1-b/2)
I_nu_s_func2 = I_nu_v_small.args[1][0]
I_nu_s_val2 = sp.integrate(I_nu_s_func2, (v, v_lim1_s, v_lim2_s))
I_nu_s_val2_simp = sp.ratsimp(I_nu_s_val2)
print(f"  Sub-integral 2 (v=b/2 to 1-b/2) = {I_nu_s_val2_simp}")

# Sub-integral 3: v from v_lim2_s (1-b/2) to 1
I_nu_s_func3 = I_nu_v_small.args[2][0]
I_nu_s_val3 = sp.integrate(I_nu_s_func3, (v, v_lim2_s, R_1))
I_nu_s_val3_simp = sp.ratsimp(I_nu_s_val3)
print(f"  Sub-integral 3 (v=1-b/2 to 1) = {I_nu_s_val3_simp}")

Total_I_nu_small_calc = I_nu_s_val1 + I_nu_s_val2 + I_nu_s_val3
print("...Integration complete.")

# Simplify the result (can be very slow)
print("Simplifying Total_I_nu_small...")
Total_I_nu_small_calc_simp = sp.ratsimp(Total_I_nu_small_calc)
print("...Simplification complete.")

nu_small_calc = sp.ratsimp(12 * Total_I_nu_small_calc_simp - 2)
rho_small_calc = sp.ratsimp(rho_small_b)
xi_small_calc = sp.ratsimp(xi_small_b)

print(f"\nRecalculated Total Integral I_nu = {Total_I_nu_small_calc_simp}")
print(f"Recalculated nu(C_b) = {nu_small_calc}")
print(f"             rho(C_b) = {rho_small_calc}")
print(f"             xi(C_b)  = {xi_small_calc}")

diff_nu_rho_small_calc = sp.ratsimp(nu_small_calc - rho_small_calc)
diff_nu_xi_small_calc = sp.ratsimp(nu_small_calc - xi_small_calc)
print(f"Recalculated Diff (nu-rho) = {diff_nu_rho_small_calc}")
print(f"Recalculated Diff (nu-xi)  = {diff_nu_xi_small_calc}")

# --- 5. Case b >= 1 ---
print("\n--- Running Case b >= 1 ---")
# Define integration limits for v
v_lim1_l = R_1 / (R_2 * b)
v_lim2_l = R_1 - R_1 / (R_2 * b)

# Define s_v pieces from eq (4.22)
sv_p1_l = sp.sqrt(R_2 * v / b)
sv_p2_l = v + R_1 / (R_2 * b)
sv_p3_l = R_1 + R_1 / b - sp.sqrt(R_2 * (R_1 - v) / b)

# Define a_v pieces
av_p1_l = sv_p1_l - R_1 / b
av_p2_l = sv_p2_l - R_1 / b
av_p3_l = sv_p3_l - R_1 / b

# Define I_nu(v) using Piecewise based on conditions derived in thought process
I_nu_v_large = sp.Piecewise(
    (
        I_nu_1a.subs(s, sv_p1_l),
        v <= v_lim1_l,
    ),  # Range 1: a_v <= 0, s_v <= 1/b <= 1 -> Use 1a
    (
        I_nu_2a.subs({s: sv_p2_l, a: av_p2_l}),
        v <= v_lim2_l,
    ),  # Range 2: a_v > 0, s_v <= 1 -> Use 2a
    (
        I_nu_2b.subs({s: sv_p3_l, a: av_p3_l}),
        v > v_lim2_l,
    ),  # Range 3: a_v > 0, s_v > 1 -> Use 2b
)

print("Calculating sub-integrals for b >= 1...")
# Sub-integral 1: v from 0 to v_lim1_l (1/(2*b))
I_nu_l_func1 = I_nu_v_large.args[0][0]
I_nu_l_val1 = sp.integrate(I_nu_l_func1, (v, R_0, v_lim1_l))
I_nu_l_val1_simp = sp.ratsimp(I_nu_l_val1)
print(f"  Sub-integral 1 (v=0 to 1/(2*b)) = {I_nu_l_val1_simp}")

# Sub-integral 2: v from v_lim1_l (1/(2*b)) to v_lim2_l (1-1/(2*b))
I_nu_l_func2 = I_nu_v_large.args[1][0]
I_nu_l_val2 = sp.integrate(I_nu_l_func2, (v, v_lim1_l, v_lim2_l))
I_nu_l_val2_simp = sp.ratsimp(I_nu_l_val2)
print(f"  Sub-integral 2 (v=1/(2*b) to 1-1/(2*b)) = {I_nu_l_val2_simp}")

# Sub-integral 3: v from v_lim2_l (1-1/(2*b)) to 1
I_nu_l_func3 = I_nu_v_large.args[2][0]
I_nu_l_val3 = sp.integrate(I_nu_l_func3, (v, v_lim2_l, R_1))
I_nu_l_val3_simp = sp.ratsimp(I_nu_l_val3)
print(f"  Sub-integral 3 (v=1-1/(2*b) to 1) = {I_nu_l_val3_simp}")

Total_I_nu_large_calc = I_nu_l_val1 + I_nu_l_val2 + I_nu_l_val3
print("...Integration complete.")

print("Simplifying Total_I_nu_large...")
Total_I_nu_large_calc_simp = sp.ratsimp(Total_I_nu_large_calc)
print("...Simplification complete.")

nu_large_calc = sp.ratsimp(12 * Total_I_nu_large_calc_simp - 2)
rho_large_calc = sp.ratsimp(rho_large_b)
xi_large_calc = sp.ratsimp(xi_large_b)

print(f"\nRecalculated Total Integral I_nu = {Total_I_nu_large_calc_simp}")
print(f"Recalculated nu(C_b) = {nu_large_calc}")
print(f"             rho(C_b) = {rho_large_calc}")
print(f"             xi(C_b)  = {xi_large_calc}")

diff_nu_rho_large_calc = sp.ratsimp(nu_large_calc - rho_large_calc)
diff_nu_xi_large_calc = sp.ratsimp(nu_large_calc - xi_large_calc)
print(f"Recalculated Diff (nu-rho) = {diff_nu_rho_large_calc}")
print(f"Recalculated Diff (nu-xi)  = {diff_nu_xi_large_calc}")

# --- 6. Final Check at b=1 ---
print("\n--- Final Check at b=1 ---")
# Use the b>=1 formulas as they seem more likely correct based on previous script run
nu_at_1 = nu_large_calc.subs(b, 1)
rho_at_1 = rho_large_calc.subs(b, 1)
xi_at_1 = xi_large_calc.subs(b, 1)
diff_nu_rho_at_1 = diff_nu_rho_large_calc.subs(b, 1)
diff_nu_xi_at_1 = diff_nu_xi_large_calc.subs(b, 1)

print("Using b>=1 formulas at b=1:")
print(f"  nu(C_1) = {nu_at_1} (Decimal: {nu_at_1.evalf()})")
print(f"  rho(C_1)= {rho_at_1} (Decimal: {rho_at_1.evalf()})")
print(f"  xi(C_1) = {xi_at_1} (Decimal: {xi_at_1.evalf()})")
print(f"  nu-rho = {diff_nu_rho_at_1} (Decimal: {diff_nu_rho_at_1.evalf()})")
print(f"  nu-xi  = {diff_nu_xi_at_1} (Decimal: {diff_nu_xi_at_1.evalf()})")

print("\nCompare nu-xi with claimed max from Thm 4.18:")
claimed_max_diff = sp.Rational(44, 105)
print(
    f"  Calculated nu(C_1^xi,rho) - xi(C_1^xi,rho) = {diff_nu_xi_at_1} approx {diff_nu_xi_at_1.evalf()}"
)
print(
    f"  Claimed max nu-xi (Thm 4.18)           = {claimed_max_diff} approx {claimed_max_diff.evalf()}"
)
print(f"  Is calculated value > claimed max? {diff_nu_xi_at_1 > claimed_max_diff}")
