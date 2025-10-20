import sympy as sp

# --- 1. Define Symbols and Inner Integrals ---
v, b = sp.symbols('v b', real=True, positive=True)
u = sp.symbols('u', real=True)
s, a = sp.symbols('s a', real=True)

# We want to compute I_nu(v) = integral( (1-u)**2 * h(u,v), du )
# Use sp.Rational for all coefficients to avoid floats
R_1_2 = sp.Rational(1, 2)
R_1_3 = sp.Rational(1, 3)
R_1_12 = sp.Rational(1, 12)

# Case 1: a_v = 0 (i.e., s_v <= 1/b)
# I_nu(v) = integral( (1-u)**2 * b*(s-u), {u, 0, s} )
I_nu_case1 = b * (R_1_2 * s**2 - R_1_3 * s**3 + R_1_12 * s**4)

# Case 2: a_v > 0 (i.e., s_v > 1/b), where a = s - 1/b
# I_nu(v) = integral( (1-u)**2, {u, 0, a} ) + integral( (1-u)**2 * b*(s-u), {u, a, s} )
I_nu_part1 = sp.integrate((1-u)**2, (u, 0, a))
I_nu_part2 = sp.integrate((1-u)**2 * b*(s-u), (u, a, s))
# We pre-calculate this integral and substitute a = s - 1/b
I_nu_case2 = (I_nu_part1 + I_nu_part2).subs(a, s - 1/b)


# --- 2. Define M_rho(b) for comparison ---
R_3_10 = sp.Rational(3, 10)
R_1_5 = sp.Rational(1, 5)

M_rho_small = b - R_3_10 * b**2
M_rho_large = 1 - R_1_2 / b**2 + R_1_5 / b**3


# --- 3. Case 0 < b <= 1 ---
print(f"--- Running Case 0 < b <= 1 ---")

# Define s_v for 0 < b <= 1
s_v_1_small = sp.sqrt(2*v/b)
s_v_2_small = v/b + R_1_2  # Use sp.Rational(1, 2) instead of 1/2
s_v_3_small = 1 + 1/b - sp.sqrt(2*(1-v)/b)

# Substitute s_v into the correct inner integral I_nu(v)
I_v_1_small = I_nu_case1.subs(s, s_v_1_small)
I_v_2_small = I_nu_case1.subs(s, s_v_2_small)
I_v_3_small = I_nu_case2.subs(s, s_v_3_small)

# Compute the three outer integrals
print("Calculating sub-integrals (0 < b <= 1)...")
I_1_small = sp.integrate(I_v_1_small, (v, 0, b/2))
print(f"  I_1_small = {sp.ratsimp(I_1_small)}")

I_2_small = sp.integrate(I_v_2_small, (v, b/2, 1 - b/2))
print(f"  I_2_small = {sp.ratsimp(I_2_small)}")

I_3_small = sp.integrate(I_v_3_small, (v, 1 - b/2, 1))
print(f"  I_3_small = {sp.ratsimp(I_3_small)}")
print("...Sub-integrals calculated.")


# Sum them to get the total integral I_nu
# Use ratsimp() for robust simplification
Total_I_nu_small = sp.ratsimp(I_1_small + I_2_small + I_3_small)

# Calculate nu and rho
nu_small = sp.ratsimp(12 * Total_I_nu_small - 2)
rho_small = sp.ratsimp(M_rho_small)

print(f"\nTotal Integral I_nu = {Total_I_nu_small}")
print(f"nu(C_b) = {nu_small}")
print(f"rho(C_b) = {rho_small}")
# Use ratsimp() one more time on the final difference
print(f"Difference (nu - rho) = {sp.ratsimp(nu_small - rho_small)}")


# --- 4. Case b >= 1 ---
print(f"\n--- Running Case b >= 1 ---")

# Define s_v for b >= 1
s_v_1_large = sp.sqrt(2*v/b)
s_v_2_large = v + 1 / (2*b)  # This was already correct from the last fix
s_v_3_large = 1 + 1/b - sp.sqrt(2*(1-v)/b)

# Substitute s_v into the correct inner integral I_nu(v)
I_v_1_large = I_nu_case1.subs(s, s_v_1_large)
I_v_2_large = I_nu_case2.subs(s, s_v_2_large)
I_v_3_large = I_nu_case2.subs(s, s_v_3_large)

# Compute the three outer integrals
print("Calculating sub-integrals (b >= 1)...")
I_1_large = sp.integrate(I_v_1_large, (v, 0, 1/(2*b)))
print(f"  I_1_large = {sp.ratsimp(I_1_large)}")

I_2_large = sp.integrate(I_v_2_large, (v, 1/(2*b), 1 - 1/(2*b)))
print(f"  I_2_large = {sp.ratsimp(I_2_large)}")

I_3_large = sp.integrate(I_v_3_large, (v, 1 - 1/(2*b), 1))
print(f"  I_3_large = {sp.ratsimp(I_3_large)}")
print("...Sub-integrals calculated.")

# Sum them to get the total integral I_nu
Total_I_nu_large = sp.ratsimp(I_1_large + I_2_large + I_3_large)

# Calculate nu and rho
nu_large = sp.ratsimp(12 * Total_I_nu_large - 2)
rho_large = sp.ratsimp(M_rho_large)

print(f"\nTotal Integral I_nu = {Total_I_nu_large}")
print(f"nu(C_b) = {nu_large}")
print(f"rho(C_b) = {rho_large}")
print(f"Difference (nu - rho) = {sp.ratsimp(nu_large - rho_large)}")