# -*- coding: utf-8 -*-
import sympy as sp

# ------------------------------------------------------------
# Symbols and helper functions
# ------------------------------------------------------------
b = sp.symbols('b', positive=True)

# gamma and A (as in the theorem)
gamma = sp.sqrt((b - 1)/b)
A = sp.acosh(sp.sqrt(b))

# Derivatives
gamma_prime = sp.diff(gamma, b)
A_prime = sp.diff(A, b)

# ------------------------------------------------------------
# Case 1: 0 < b <= 1  (polynomial case)
# ------------------------------------------------------------
Xi_A = (8/sp.Integer(15))*b**2 - (8/sp.Integer(35))*b**3
Nu_A = (16/sp.Integer(15))*b - (12/sp.Integer(35))*b**2

lhs_poly = sp.diff(Nu_A, b)
rhs_poly = sp.diff(Xi_A, b) / b

print("=== Case 0 < b <= 1 ===")
print("Xi'(b) =", sp.simplify(sp.diff(Xi_A, b)))
print("Nu'(b) =", sp.simplify(sp.diff(Nu_A, b)))
print("Difference (should be 0):", sp.simplify(lhs_poly - rhs_poly))


# ------------------------------------------------------------
# Case 2: b > 1  (nonalgebraic case)
# ------------------------------------------------------------
# Polynomial pieces from theorem
U  = 183 - 38*b - 88*b**2 + 48*b**3
V  = 112*b**2 - 48*b**3
W  = -105/b

U_t  = 87/b + 250 - 376*b + 144*b**2
V_t  = 448*b - 144*b**2
W_t  = -105/b**2

# Definitions of Xi(b) and Nu(b)
Xi_B = (U*gamma + V + W*A) / 210
Nu_B = (U_t*gamma + V_t + W_t*A) / 420

# Derivatives
Xi_B_prime = sp.diff(Xi_B, b)
Nu_B_prime = sp.diff(Nu_B, b)

# Check derivative identity N'(b) = Xi'(b)/b
diff_B = sp.simplify(Nu_B_prime - Xi_B_prime/b)

print("\n=== Case b > 1 ===")
print("γ' =", sp.simplify(gamma_prime))
print("A' =", sp.simplify(A_prime))
print("Difference (should be 0):")
sp.pprint(diff_B.simplify())
