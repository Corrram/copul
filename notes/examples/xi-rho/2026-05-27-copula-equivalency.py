import sympy as sp

# Symbols
u, b, s = sp.symbols("u b s", positive=True, real=True)

# a_v = s_v - 1/b
a = s - 1 / b

# Old/arXiv version:
# C_old = u - b/2 * (u - a)^2 + b/2 * (a ∧ 0)^2
#
# Thesis version:
# C_new = a^+ + b/2 * ((s - a^+)^2 - (s - u)^2)
#
# Since min/max are piecewise, verify the two possible cases separately.

# Case 1: a >= 0, so a^+ = a and a ∧ 0 = 0
C_old_a_nonneg = u - b / 2 * (u - a) ** 2
C_new_a_nonneg = a + b / 2 * ((s - a) ** 2 - (s - u) ** 2)

diff_a_nonneg = sp.simplify(C_old_a_nonneg - C_new_a_nonneg)

# Case 2: a < 0, so a^+ = 0 and a ∧ 0 = a
C_old_a_neg = u - b / 2 * (u - a) ** 2 + b / 2 * a**2
C_new_a_neg = b / 2 * (s**2 - (s - u) ** 2)

diff_a_neg = sp.simplify(C_old_a_neg - C_new_a_neg)

print("Case a >= 0 difference:", diff_a_nonneg)
print("Case a < 0 difference: ", diff_a_neg)

assert diff_a_nonneg == 0
assert diff_a_neg == 0

print("Verified: the two formulas are algebraically identical for b > 0.")
