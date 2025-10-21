# -*- coding: utf-8 -*-
import sympy as sp

# ------------------------------------------------------------
# Symbols
# ------------------------------------------------------------
b = sp.symbols("b", positive=True)  # parameter b>0
R = sp.symbols("R", positive=True)  # R in regimes (i),(ii),(iii)
r = sp.symbols("r", positive=True)  # r in regime (iv)
q = sp.symbols("q")  # auxiliary q

# Regime relations
r_of_R = sp.sqrt(R**2 - 1 / b)  # r = sqrt(R^2 - 1/b) in (iii)
Rmid = 1 / sp.sqrt(b)

# Primitives from Lemma 1D
F = sp.Lambda(
    (sp.Symbol("x"), sp.Symbol("qq")),
    sp.Rational(1, 5) * sp.Symbol("x") ** 5
    - sp.Rational(2, 3) * sp.Symbol("qq") * sp.Symbol("x") ** 3
    + sp.Symbol("qq") ** 2 * sp.Symbol("x"),
)
S = sp.Lambda(
    (sp.Symbol("x"), sp.Symbol("qq")),
    sp.Rational(1, 5) * sp.Symbol("x") ** 5
    - sp.Rational(1, 3) * sp.Symbol("qq") * sp.Symbol("x") ** 3,
)


# ------------------------------------------------------------
# Kernels G, H (generic expressions before inserting regime data)
#   G = a + b^2 ( F(Xa;q) - F(Xs;q) )
#   H = (1-(1-a)^3)/3 + b ( S(Xa;q) - S(Xs;q) )
# ------------------------------------------------------------
def G_expr(a, Xa, Xs, qsym):
    return a + b**2 * (F(Xa, qsym) - F(Xs, qsym))


def H_expr(a, Xa, Xs, qsym):
    return sp.Rational(1, 3) * (1 - (1 - a) ** 3) + b * (S(Xa, qsym) - S(Xs, qsym))


# ------------------------------------------------------------
# Regime integrands for each case
# Case A: 0 < b <= 1  → regimes (i), (ii), (iv)
# ------------------------------------------------------------
# (i) Upper–clamped: a=1-R, Xa=R, Xs=0, q=R^2-1/b, weight: (-v')dq = 2 b R^2 dR
q_i = R**2 - 1 / b
G_i = sp.simplify(G_expr(1 - R, R, 0, q_i))
H_i = sp.simplify(H_expr(1 - R, R, 0, q_i))
w_i = 2 * b * R**2

# (ii) Unclamped: a=0, Xa=1, Xs=0, q=R^2-1/b, weight: (-v')dq = 2 b R dR
q_ii = R**2 - 1 / b
G_ii = sp.simplify(G_expr(0, 1, 0, q_ii))
H_ii = sp.simplify(H_expr(0, 1, 0, q_ii))
w_ii = 2 * b * R

# (iv) Lower–clamped: a=0, Xa=1, Xs=r, q=r^2, weight: (-v')dq = 2 b r(1-r) dr
q_iv = r**2
G_iv = sp.simplify(G_expr(0, 1, r, q_iv))
H_iv = sp.simplify(H_expr(0, 1, r, q_iv))
w_iv = 2 * b * r * (1 - r)

# ------------------------------------------------------------
# Case B: b > 1  → regimes (i), (iii), (iv)
# (i) is same as above but with upper limit 1/sqrt(b)
# (iii) Double–clamped: a=1-R, Xa=R, Xs=r, q=R^2-1/b, r=sqrt(R^2-1/b),
#       weight: (-v')dq = (1 + b(R-r)^2) dR
# ------------------------------------------------------------
q_iii = R**2 - 1 / b
G_iii = sp.simplify(G_expr(1 - R, R, r_of_R, q_iii))
H_iii = sp.simplify(H_expr(1 - R, R, r_of_R, q_iii))
w_iii = sp.simplify(1 + b * (R - r_of_R) ** 2)

# ------------------------------------------------------------
# “Acosh” piece, shown explicitly
# ∫_{1/√b}^1 dR / [ R sqrt(1 - 1/(bR^2)) ] = acosh(sqrt(b))
# ------------------------------------------------------------
J = sp.integrate(1 / (R * sp.sqrt(1 - 1 / (b * R**2))), (R, Rmid, 1))
J_simplified = sp.simplify(J)  # -> acosh(sqrt(b))

# ------------------------------------------------------------
# Assemble integrals for each case
# ------------------------------------------------------------
# Case A: 0<b<=1
Xi_A = sp.simplify(
    6
    * (
        sp.integrate(G_i * w_i, (R, 0, 1))
        + sp.integrate(G_ii * w_ii, (R, 1, 1 / sp.sqrt(b)))
        + sp.integrate(G_iv * w_iv, (r, 0, 1))
    )
    - 2
)
Nu_A = sp.simplify(
    12
    * (
        sp.integrate(H_i * w_i, (R, 0, 1))
        + sp.integrate(H_ii * w_ii, (R, 1, 1 / sp.sqrt(b)))
        + sp.integrate(H_iv * w_iv, (r, 0, 1))
    )
    - 2
)

# Case B: b>1
Xi_B = sp.simplify(
    6
    * (
        sp.integrate(G_i * w_i, (R, 0, 1 / sp.sqrt(b)))
        + sp.integrate(G_iii * w_iii, (R, 1 / sp.sqrt(b), 1))
        + sp.integrate(G_iv * w_iv, (r, sp.sqrt(1 - 1 / b), 1))
    )
    - 2
)
Nu_B = sp.simplify(
    12
    * (
        sp.integrate(H_i * w_i, (R, 0, 1 / sp.sqrt(b)))
        + sp.integrate(H_iii * w_iii, (R, 1 / sp.sqrt(b), 1))
        + sp.integrate(H_iv * w_iv, (r, sp.sqrt(1 - 1 / b), 1))
    )
    - 2
)

# ------------------------------------------------------------
# Pretty print results
# ------------------------------------------------------------
print("Nonalgebraic piece J(b) =", J_simplified)  # expected acosh(sqrt(b))

print("\nCase 0<b<=1:")
print("xi(C_b) =")
sp.pprint(Xi_A)
print("nu(C_b) =")
sp.pprint(Nu_A)

print("\nCase b>1:")
print("xi(C_b) =")
sp.pprint(Xi_B)
print("nu(C_b) =")
sp.pprint(Nu_B)

gamma = sp.sqrt((b - 1) / b)

A = sp.acosh(sp.sqrt(b))

Xi_target = (
    183 * gamma
    - 38 * b * gamma
    - 88 * b**2 * gamma
    + 112 * b**2
    + 48 * b**3 * gamma
    - 48 * b**3
    - 105 * A / b
) / 210

Nu_target = (
    (87 * gamma) / b
    + 250 * gamma
    - 376 * b * gamma
    + 448 * b
    + 144 * b**2 * gamma
    - 144 * b**2
    - 105 * A / b**2
) / 420

# Help SymPy see acosh: replace logs with acosh form
R = sp.sqrt(b * (b - 1))
Xi_rewritten = Xi_B.xreplace({sp.log(b) - 2 * sp.log(b + R): -2 * sp.acosh(sp.sqrt(b))})
Nu_rewritten = Nu_B.xreplace({sp.log(b) - 2 * sp.log(b + R): -2 * sp.acosh(sp.sqrt(b))})

print(sp.simplify(sp.together(Xi_rewritten - Xi_target)))  # -> 0
print(sp.simplify(sp.together(Nu_rewritten - Nu_target)))  # -> 0
