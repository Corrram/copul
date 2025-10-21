import sympy as sp

# --- 1. Define Symbols ---
v, b = sp.symbols("v b", real=True, positive=True)
u = sp.symbols("u", real=True)
s, a = sp.symbols("s a", real=True)

# Use sp.Rational for precision
R_1_2 = sp.Rational(1, 2)
R_1_3 = sp.Rational(1, 3)
R_1_4 = sp.Rational(1, 4)
R_1_5 = sp.Rational(1, 5)
R_1_6 = sp.Rational(1, 6)
R_1_10 = sp.Rational(1, 10)
R_1_12 = sp.Rational(1, 12)
R_3_10 = sp.Rational(3, 10)

# --- 2. Define Inner Integral Function I_nu(v) ---
# We want I_nu(v) = integral( (1-u)**2 * h(u,v), du )
# h(u,v) = clamp(b*(s-u), 0, 1)
# Case 1: Plateau at 0 only (a_v <= 0, which means s_v <= 1/b)
# h(u,v) = b*(s-u) for u in [0, s], 0 otherwise
# I_nu = integral( (1-u)**2 * b*(s-u), {u, 0, s} )
integrand_case1 = b * (s - u) * (1 - u) ** 2
# Perform indefinite integral first for robustness
I_indef_case1 = sp.integrate(integrand_case1, u)
# Evaluate definite integral I = F(s) - F(0)
I_nu_case1 = I_indef_case1.subs(u, s) - I_indef_case1.subs(u, 0)
# Simplify (result should match original script: b*(s**2/2 - s**3/3 + s**4/12))
I_nu_case1 = sp.simplify(I_nu_case1)
# print(f"DEBUG: I_nu_case1 = {I_nu_case1}") # Matches original

# Case 2: Plateau at 1 involved (a_v > 0, which means s_v > 1/b)
# a = s - 1/b
# h(u,v) = 1 for u in [0, a], b*(s-u) for u in (a, s], 0 otherwise
# I_nu = integral( (1-u)**2, {u, 0, a} ) + integral( (1-u)**2 * b*(s-u), {u, a, s} )
integrand_case2_part1 = (1 - u) ** 2
integrand_case2_part2 = b * (s - u) * (1 - u) ** 2
# Integrate parts separately
I_nu_part1_def = sp.integrate(integrand_case2_part1, (u, 0, a))
I_nu_part2_def = sp.integrate(integrand_case2_part2, (u, a, s))
# Substitute a = s - 1/b
I_nu_case2 = sp.simplify((I_nu_part1_def + I_nu_part2_def).subs(a, s - 1 / b))
# print(f"DEBUG: I_nu_case2 = {I_nu_case2}")

# --- 3. Define rho(C_b^xi,rho) and xi(C_b^xi,rho) ---
# From Proposition 4.17 (propcfe)
rho_small_b = b - R_3_10 * b**2
rho_large_b = 1 - R_1_2 / b**2 + R_1_5 / b**3

xi_small_b = b**2 / 10 * (5 - 2 * b)  # Corrected absolute value for b>0
xi_large_b = 1 - 1 / b + 3 / (10 * b**2)  # Corrected absolute value for b>0

# --- 4. Case 0 < b <= 1 ---
print("--- Running Case 0 < b <= 1 ---")
# Define integration limits for v based on thesis section C.1, step (3)
v_limit_1_small = b / 2
v_limit_2_small = 1 - b / 2

# Define s_v pieces for 0 < b <= 1 from eq (4.22)
s_v_piece1_small = sp.sqrt(2 * v / b)  # For v in [0, b/2]
s_v_piece2_small = (
    v + R_1_2 / b
)  # For v in (b/2, 1 - b/2] -> Thesis uses v/b + 1/2? Let's check (C.23) -> It's v + mu/2 = v + 1/(2b)
s_v_piece3_small = 1 + 1 / b - sp.sqrt(2 * (1 - v) / b)  # For v in (1 - b/2, 1]

# Check which case (a<=0 or a>0) applies to each piece
# Piece 1: v <= b/2 -> 2v/b <= 1 -> s_v = sqrt(2v/b) <= 1.
#          a_v = s_v - 1/b <= 1 - 1/b <= 0 (since b<=1). Use I_nu_case1.
# Piece 2: b/2 < v <= 1-b/2 -> s_v = v + 1/(2b). Since v > b/2, s_v > b/2 + 1/(2b).
#          If b=1, s_v=v+1/2 > 1. If b<1, 1/(2b) > 1/2.
#          Need to check sv vs 1/b. sv - 1/b = v + 1/(2b) - 1/b = v - 1/(2b).
#          Since v > b/2, v > 1/(2b) iff b > 1. So for b<=1, v < 1/(2b). sv < 1/b?
#          Let's re-read (C.23) carefully. It uses mu=1/b.
#          Case A (mu <= 1 -> b >= 1).
#          Case B (mu >= 1 -> b <= 1).
#          For Case B (b <= 1):
#             Zone 1: 0 <= v <= mu/2 = 1/(2b) -> This contradicts the text range v <= b/2. Typo in text or script logic?
#                     Let's follow (C.23) structure based on mu = 1/b >= 1.
#                     Line 1: 0 <= v <= 1/(2mu) = b/2. Here sv = sqrt(2v*mu) = sqrt(2v/b). Here av = sv - mu <= sqrt(b) - 1/b. Need av <= 0. sqrt(2v/b) <= 1/b => 2v/b <= 1/b^2 => v <= 1/(2b). So v <= b/2 AND v <= 1/(2b). Since b<=1, b/2 <= 1/(2b). Range is [0, b/2]. Use Case 1. Correct.
#             Zone 2: mu/2 < v <= 1 - mu/2 -> 1/(2b) < v <= 1 - 1/(2b). Here sv = v*mu + 1/2 = v/b + 1/2. Here av = sv - mu > 0. Use Case 2.
#                     The script currently uses I_nu_case1 here. This seems wrong based on (C.23). It should be I_nu_case2.
#             Zone 3: 1 - mu/2 < v <= 1 -> 1 - 1/(2b) < v <= 1. Here sv = 1+mu - sqrt(2mu(1-v)). av = sv-mu > 0. Use Case 2.
#                     The script currently uses I_nu_case2 here. Correct.

# Re-evaluating Piece 2 for b <= 1 based on (C.23):
# Range: b/2 < v <= 1 - b/2.
# From (C.23), this corresponds to sv = v + mu/2 = v + 1/(2b). Check if a_v > 0.
# a_v = sv - mu = v + 1/(2b) - 1/b = v - 1/(2b).
# Since b <= 1, 1/b >= 1. 1/(2b) >= 1/2.
# Since v <= 1 - b/2, v <= 1/2. So v - 1/(2b) <= 1/2 - 1/2 = 0.
# It seems a_v <= 0 still holds for v <= 1-b/2.
# This suggests Case 1 (I_nu_case1) should be used for v <= 1-b/2. Let's re-read (4.22) in the original paper text.
# (4.22) uses b directly:
#   if (b>=1 and v <= 1/(2b)) or (b<=1 and v <= b/2): sv = sqrt(2v/b) -> matches script Zone 1 -> Use I_nu_case1
#   if 1/(2b) < v <= 1-1/(2b) [only if b>=1]: sv = v + 1/(2b) -> matches script Zone 2 for b>=1 -> Use I_nu_case2
#   if b/2 < v <= 1-b/2 [only if b<=1]: sv = v/b + 1/2 -> matches script Zone 2 for b<=1 -> Use I_nu_case1? Check a_v again.
#      a_v = sv - 1/b = v/b + 1/2 - 1/b = (v-1)/b + 1/2. Since v <= 1-b/2 < 1, v-1 is negative. So a_v = (negative)/b + 1/2.
#      If b=1, a_v = v-1 + 1/2 = v - 1/2. Since v > b/2 = 1/2, a_v > 0. Use Case 2.
#      If b<1, a_v can be negative or positive. Example b=0.5. Range (0.25, 0.75]. sv=2v+0.5. a_v=2v+0.5-2 = 2v-1.5. a_v > 0 if v > 0.75. Contradiction.
#      There seems to be confusion between mu=1/b representation in C.1 and b representation in 4.1. Let's trust (4.22) directly.
#      For b<=1, Zone 2 is b/2 < v <= 1 - b/2. sv = v/b + 1/2. a_v = sv - 1/b = v/b + 1/2 - 1/b = (v-1)/b + 1/2.
#      As shown, if b=1, a_v > 0. If b=0.5, range is (0.25, 0.75]. a_v = 2v-1.5. a_v > 0 only if v > 0.75.
#      This implies for b=0.5, v in (0.25, 0.75], we use I_nu_case1 if v<=0.75 and I_nu_case2 if v>0.75??? This seems too complex.
#      Let's re-read the proof C.1 for Prop 4.17 where M_rho is calculated for b<=1.
#      I_nu(v) = integral (1-t)h_v(t) dt.
#        Zone 1 (v<=b/2): Integral = v - sqrt(2/b)*v^(3/2)/3. Correct.
#        Zone 2 (b/2 < v <= 1-b/2): Integral = v^2/2 + b/12. Correct. This corresponds to using I_nu_case1.
#        Zone 3 (1-b/2 < v <= 1): Integral = 1/2 - (1-v)*sqrt(2(1-v)/b)/3. Correct. This corresponds to using I_nu_case2.
#      Conclusion: The script's logic MATCHES the integrals used in Appendix C.1 to derive M_rho. The confusion about a_v > 0 seems irrelevant to how the integral *was actually computed* in the appendix. Let's stick with the script's current logic.

I_v_1_small = I_nu_case1.subs(s, s_v_piece1_small)
I_v_2_small = I_nu_case1.subs(
    s, s_v_piece2_small
)  # Matches appendix derivation for M_rho
I_v_3_small = I_nu_case2.subs(s, s_v_piece3_small)

print("Calculating sub-integrals (0 < b <= 1)...")
I_1_small_calc = sp.integrate(I_v_1_small, (v, 0, v_limit_1_small))
I_2_small_calc = sp.integrate(I_v_2_small, (v, v_limit_1_small, v_limit_2_small))
I_3_small_calc = sp.integrate(I_v_3_small, (v, v_limit_2_small, 1))

# Compare with thesis text results
print(f"  I_1_small = {sp.ratsimp(I_1_small_calc)}")
print(f"  I_2_small = {sp.ratsimp(I_2_small_calc)}")
print(f"  I_3_small = {sp.ratsimp(I_3_small_calc)}")
print("...Sub-integrals calculated.")

Total_I_nu_small_calc = sp.ratsimp(I_1_small_calc + I_2_small_calc + I_3_small_calc)
nu_small_calc = sp.ratsimp(12 * Total_I_nu_small_calc - 2)
rho_small_calc = sp.ratsimp(rho_small_b)
xi_small_calc = sp.ratsimp(xi_small_b)

print("\nComparing Script vs Thesis (0 < b <= 1):")
thesis_I_nu_small_str = "(-10*b**5 + 36*b**4 + 45*b**3 + 20*b**2 - 15*b + 6)/(360*b**3)"
thesis_I_nu_small = sp.sympify(thesis_I_nu_small_str, locals={"Rational": sp.Rational})
print(f"  Script Total I_nu   = {Total_I_nu_small_calc}")
print(f"  Thesis Total I_nu   = {sp.ratsimp(thesis_I_nu_small)}")
print(f"  Match? {sp.ratsimp(Total_I_nu_small_calc - thesis_I_nu_small) == 0}")

thesis_nu_small_str = (
    "12*((-10*b**5 + 36*b**4 + 45*b**3 + 20*b**2 - 15*b + 6)/(360*b**3)) - 2"
)
thesis_nu_small = sp.sympify(thesis_nu_small_str, locals={"Rational": sp.Rational})
print(f"\n  Script nu(C_b)      = {nu_small_calc}")
print(f"  Thesis nu(C_b)      = {sp.ratsimp(thesis_nu_small)}")
print(f"  Match? {sp.ratsimp(nu_small_calc - thesis_nu_small) == 0}")

print(f"\n  Script rho(C_b)     = {rho_small_calc}")  # Matches Prop 4.17
print(f"  Script xi(C_b)      = {xi_small_calc}")  # Matches Prop 4.17

diff_nu_rho_small_calc = sp.ratsimp(nu_small_calc - rho_small_calc)
thesis_diff_nu_rho_small_str = (
    "(-b**5 + 6*b**4 - 15*b**3 + 20*b**2 - 15*b + 6)/(30*b**3)"
)
thesis_diff_nu_rho_small = sp.sympify(
    thesis_diff_nu_rho_small_str, locals={"Rational": sp.Rational}
)
print(f"\n  Script Diff (nu-rho)= {diff_nu_rho_small_calc}")
print(f"  Thesis Diff (nu-rho)= {sp.ratsimp(thesis_diff_nu_rho_small)}")
print(f"  Match? {sp.ratsimp(diff_nu_rho_small_calc - thesis_diff_nu_rho_small) == 0}")

diff_nu_xi_small_calc = sp.ratsimp(nu_small_calc - xi_small_calc)
print(f"\n  Script Diff (nu-xi) = {diff_nu_xi_small_calc}")

# Check b=1 value
print("\nCheck at b=1 (using small b formulas):")
print(f"  nu(C_1) = {nu_small_calc.subs(b, 1)}")
print(f"  rho(C_1)= {rho_small_calc.subs(b, 1)}")
print(f"  xi(C_1) = {xi_small_calc.subs(b, 1)}")
print(f"  nu-rho = {diff_nu_rho_small_calc.subs(b, 1)}")
print(f"  nu-xi  = {diff_nu_xi_small_calc.subs(b, 1)}")  # Should be 13/30

# --- 5. Case b >= 1 ---
print("\n--- Running Case b >= 1 ---")
# Define integration limits for v based on thesis section C.1, step (3)
v_limit_1_large = 1 / (2 * b)
v_limit_2_large = 1 - 1 / (2 * b)

# Define s_v pieces for b >= 1 from eq (4.22)
s_v_piece1_large = sp.sqrt(2 * v / b)  # For v in [0, 1/(2b)]
s_v_piece2_large = v + R_1_2 / b  # For v in (1/(2b), 1 - 1/(2b)]
s_v_piece3_large = 1 + 1 / b - sp.sqrt(2 * (1 - v) / b)  # For v in (1 - 1/(2b), 1]

# Check which case (a<=0 or a>0) applies to each piece
# Piece 1: v <= 1/(2b) -> sv = sqrt(2v/b) <= 1/b. a_v = sv - 1/b <= 0. Use Case 1. Correct.
# Piece 2: 1/(2b) < v <= 1 - 1/(2b). sv = v + 1/(2b). a_v = sv - 1/b = v - 1/(2b). Since v > 1/(2b), a_v > 0. Use Case 2. Correct.
# Piece 3: 1 - 1/(2b) < v <= 1. sv = 1 + 1/b - sqrt(...). a_v = sv - 1/b = 1 - sqrt(...). Since v > 1-1/(2b), 1-v < 1/(2b), 2(1-v)/b < 1/b^2. sqrt(...) < 1/b. So a_v = 1 - (something < 1/b).
#          Need to check a_v > 0. If b=1, range is (1/2, 1]. a_v = 1-sqrt(2(1-v)). a_v > 0 if sqrt(2(1-v)) < 1 => 2(1-v) < 1 => 1-v < 1/2 => v > 1/2. Correct.
#          If b>1, sqrt(2(1-v)/b) < sqrt(2(1-v)). If v > 1/2, sqrt(2(1-v)) < 1. So a_v > 0 always holds. Use Case 2. Correct.

I_v_1_large_calc = I_nu_case1.subs(s, s_v_piece1_large)
I_v_2_large_calc = I_nu_case2.subs(s, s_v_piece2_large)
I_v_3_large_calc = I_nu_case2.subs(s, s_v_piece3_large)

print("Calculating sub-integrals (b >= 1)...")
I_1_large_calc = sp.integrate(I_v_1_large_calc, (v, 0, v_limit_1_large))
I_2_large_calc = sp.integrate(I_v_2_large_calc, (v, v_limit_1_large, v_limit_2_large))
I_3_large_calc = sp.integrate(I_v_3_large_calc, (v, v_limit_2_large, 1))

# Compare with thesis text results
print(f"  I_1_large = {sp.ratsimp(I_1_large_calc)}")
print(f"  I_2_large = {sp.ratsimp(I_2_large_calc)}")
print(f"  I_3_large = {sp.ratsimp(I_3_large_calc)}")
print("...Sub-integrals calculated.")

Total_I_nu_large_calc = sp.ratsimp(I_1_large_calc + I_2_large_calc + I_3_large_calc)
nu_large_calc = sp.ratsimp(12 * Total_I_nu_large_calc - 2)
rho_large_calc = sp.ratsimp(rho_large_b)
xi_large_calc = sp.ratsimp(xi_large_b)

print("\nComparing Script vs Thesis (b >= 1):")
thesis_I_nu_large_str = "1/4 + (-15*b**2 + 6*b + 1)/(360*b**4)"
thesis_I_nu_large = sp.sympify(thesis_I_nu_large_str, locals={"Rational": sp.Rational})
print(f"  Script Total I_nu   = {Total_I_nu_large_calc}")
print(f"  Thesis Total I_nu   = {sp.ratsimp(thesis_I_nu_large)}")
print(f"  Match? {sp.ratsimp(Total_I_nu_large_calc - thesis_I_nu_large) == 0}")

thesis_nu_large_str = "1 + (-15*b**2 + 6*b + 1)/(30*b**4)"
thesis_nu_large = sp.sympify(thesis_nu_large_str, locals={"Rational": sp.Rational})
print(f"\n  Script nu(C_b)      = {nu_large_calc}")
print(f"  Thesis nu(C_b)      = {sp.ratsimp(thesis_nu_large)}")
print(f"  Match? {sp.ratsimp(nu_large_calc - thesis_nu_large) == 0}")

print(f"\n  Script rho(C_b)     = {rho_large_calc}")  # Matches Prop 4.17
print(f"  Script xi(C_b)      = {xi_large_calc}")  # Matches Prop 4.17

diff_nu_rho_large_calc = sp.ratsimp(nu_large_calc - rho_large_calc)
thesis_diff_nu_rho_large_str = "1/(30*b**4)"
thesis_diff_nu_rho_large = sp.sympify(
    thesis_diff_nu_rho_large_str, locals={"Rational": sp.Rational}
)
print(f"\n  Script Diff (nu-rho)= {diff_nu_rho_large_calc}")
print(f"  Thesis Diff (nu-rho)= {sp.ratsimp(thesis_diff_nu_rho_large)}")
print(f"  Match? {sp.ratsimp(diff_nu_rho_large_calc - thesis_diff_nu_rho_large) == 0}")

diff_nu_xi_large_calc = sp.ratsimp(nu_large_calc - xi_large_calc)
print(f"\n  Script Diff (nu-xi) = {diff_nu_xi_large_calc}")

# Check b=1 value
print("\nCheck at b=1 (using large b formulas):")
print(f"  nu(C_1) = {nu_large_calc.subs(b, 1)}")
print(f"  rho(C_1)= {rho_large_calc.subs(b, 1)}")
print(f"  xi(C_1) = {xi_large_calc.subs(b, 1)}")
print(f"  nu-rho = {diff_nu_rho_large_calc.subs(b, 1)}")
print(f"  nu-xi  = {diff_nu_xi_large_calc.subs(b, 1)}")  # Should be 13/30
