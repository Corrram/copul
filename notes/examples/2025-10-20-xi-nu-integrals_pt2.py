# -*- coding: utf-8 -*-
import sympy as sp

# ------------------------------------------------------------
# Symbols and assumptions
# ------------------------------------------------------------
b  = sp.symbols('b', positive=True)      # parameter b > 0
R  = sp.symbols('R', positive=True)      # R in regimes (i),(ii),(iii)
r  = sp.symbols('r', positive=True)      # r in regime (iv)

# Regime relations
r_of_R = sp.sqrt(R**2 - 1/b)             # r = sqrt(R^2 - 1/b) in (iii)
Rmid   = 1/sp.sqrt(b)

# ------------------------------------------------------------
# Primitives from Lemma 1D
#   F(x;q) = x^5/5 - (2q/3) x^3 + q^2 x
#   S(x;q) = x^5/5 - (q/3) x^3
# ------------------------------------------------------------
x, qq = sp.symbols('x qq')
F = sp.Lambda((x, qq),  sp.Rational(1,5)*x**5 - sp.Rational(2,3)*qq*x**3 + qq**2*x)
S = sp.Lambda((x, qq),  sp.Rational(1,5)*x**5 - sp.Rational(1,3)*qq*x**3)

# ------------------------------------------------------------
# Kernels G, H:
#   G = a + b^2 ( F(Xa;q) - F(Xs;q) )
#   H = (1-(1-a)^3)/3 + b ( S(Xa;q) - S(Xs;q) )
# ------------------------------------------------------------
def G_expr(a, Xa, Xs, qsym):
    return sp.simplify(a + b**2*(F(Xa, qsym) - F(Xs, qsym)))

def H_expr(a, Xa, Xs, qsym):
    return sp.simplify(sp.Rational(1,3)*(1 - (1-a)**3) + b*(S(Xa, qsym) - S(Xs, qsym)))

# ------------------------------------------------------------
# Regime integrands (generic)
# ------------------------------------------------------------
# (i) Upper–clamped: a=1-R, Xa=R, Xs=0, q=R^2-1/b, weight: (-v')dq = 2 b R^2 dR
q_i   = R**2 - 1/b
G_i   = G_expr(1 - R, R, 0, q_i)
H_i   = H_expr(1 - R, R, 0, q_i)
w_i   = 2*b*R**2

# (ii) Unclamped: a=0, Xa=1, Xs=0, q=R^2-1/b, weight: (-v')dq = 2 b R dR
q_ii  = R**2 - 1/b
G_ii  = G_expr(0, 1, 0, q_ii)
H_ii  = H_expr(0, 1, 0, q_ii)
w_ii  = 2*b*R

# (iii) Double–clamped: a=1-R, Xa=R, Xs=r, q=R^2-1/b, r=sqrt(R^2-1/b),
#       weight: (-v')dq = (1 + b(R-r)^2) dR
q_iii = R**2 - 1/b
G_iii = G_expr(1 - R, R, r_of_R, q_iii)
H_iii = H_expr(1 - R, R, r_of_R, q_iii)
w_iii = sp.simplify(1 + b*(R - r_of_R)**2)

# (iv) Lower–clamped: a=0, Xa=1, Xs=r, q=r^2, weight: (-v')dq = 2 b r(1-r) dr
q_iv  = r**2
G_iv  = G_expr(0, 1, r, q_iv)
H_iv  = H_expr(0, 1, r, q_iv)
w_iv  = 2*b*r*(1 - r)

# ------------------------------------------------------------
# Helper: pretty print with a title
# ------------------------------------------------------------
def show(title, expr):
    print(f"\n{title}")
    sp.pprint(sp.simplify(sp.together(expr)))

# ------------------------------------------------------------
# A helpful identity: the “acosh” piece in (iii)
# ∫_{1/√b}^1 dR / [ R sqrt(1 - 1/(bR^2)) ] = acosh(sqrt(b))
# SymPy may produce logs; we rewrite to acosh afterwards.
# ------------------------------------------------------------
J = sp.integrate( 1/(R*sp.sqrt(1 - 1/(b*R**2))), (R, Rmid, 1) )
J_simplified = sp.simplify(J)

print("Nonalgebraic piece J(b) =")
sp.pprint(J_simplified)  # expect acosh(sqrt(b)) or equivalent logs

# ------------------------------------------------------------
# CASE A: 0 < b <= 1   (regimes (i), (ii), (iv))
#   Xi_A = 6*( I_i_xi_A + I_ii_xi_A + I_iv_xi_A ) - 2
#   Nu_A = 12*( I_i_nu_A + I_ii_nu_A + I_iv_nu_A ) - 2
# ------------------------------------------------------------
I_i_xi_A   = sp.integrate(G_i*w_i,   (R, 0, 1))
I_ii_xi_A  = sp.integrate(G_ii*w_ii, (R, 1, 1/sp.sqrt(b)))
I_iv_xi_A  = sp.integrate(G_iv*w_iv, (r, 0, 1))
Xi_A       = sp.simplify( 6*(I_i_xi_A + I_ii_xi_A + I_iv_xi_A) - 2 )

I_i_nu_A   = sp.integrate(H_i*w_i,   (R, 0, 1))
I_ii_nu_A  = sp.integrate(H_ii*w_ii, (R, 1, 1/sp.sqrt(b)))
I_iv_nu_A  = sp.integrate(H_iv*w_iv, (r, 0, 1))
Nu_A       = sp.simplify( 12*(I_i_nu_A + I_ii_nu_A + I_iv_nu_A) - 2 )

print("\n=== Case A: 0 < b <= 1 ===")
show("I_i_xi_A  (regime i, xi):",  I_i_xi_A)
show("I_ii_xi_A (regime ii, xi):", I_ii_xi_A)
show("I_iv_xi_A (regime iv, xi):", I_iv_xi_A)
show("Xi_A (sum):", Xi_A)

show("I_i_nu_A  (regime i, nu):",  I_i_nu_A)
show("I_ii_nu_A (regime ii, nu):", I_ii_nu_A)
show("I_iv_nu_A (regime iv, nu):", I_iv_nu_A)
show("Nu_A (sum):", Nu_A)

# ------------------------------------------------------------
# CASE B: b > 1   (regimes (i), (iii), (iv))
#   Xi_B = 6*( I_i_xi_B + I_iii_xi_B + I_iv_xi_B ) - 2
#   Nu_B = 12*( I_i_nu_B + I_iii_nu_B + I_iv_nu_B ) - 2
# ------------------------------------------------------------
I_i_xi_B    = sp.integrate(G_i*w_i,    (R, 0, 1/sp.sqrt(b)))
I_iii_xi_B  = sp.integrate(G_iii*w_iii,(R, 1/sp.sqrt(b), 1))
I_iv_xi_B   = sp.integrate(G_iv*w_iv,  (r, sp.sqrt(1-1/b), 1))
Xi_B        = sp.simplify( 6*(I_i_xi_B + I_iii_xi_B + I_iv_xi_B) - 2 )

I_i_nu_B    = sp.integrate(H_i*w_i,    (R, 0, 1/sp.sqrt(b)))
I_iii_nu_B  = sp.integrate(H_iii*w_iii,(R, 1/sp.sqrt(b), 1))
I_iv_nu_B   = sp.integrate(H_iv*w_iv,  (r, sp.sqrt(1-1/b), 1))
Nu_B        = sp.simplify( 12*(I_i_nu_B + I_iii_nu_B + I_iv_nu_B) - 2 )

print("\n=== Case B: b > 1 ===")
show("I_i_xi_B    (regime i, xi):",     I_i_xi_B)
show("I_iii_xi_B  (regime iii, xi):",   I_iii_xi_B)
show("I_iv_xi_B   (regime iv, xi):",    I_iv_xi_B)
show("Xi_B (sum):", Xi_B)

show("I_i_nu_B    (regime i, nu):",     I_i_nu_B)
show("I_iii_nu_B  (regime iii, nu):",   I_iii_nu_B)
show("I_iv_nu_B   (regime iv, nu):",    I_iv_nu_B)
show("Nu_B (sum):", Nu_B)

# ------------------------------------------------------------
# Optional: rewrite logs -> acosh for b>1 closed forms
# acosh(sqrt(b)) = log( sqrt(b) + sqrt(b - 1) ) = (1/2)*log(b) - log(b + sqrt{b(b-1)}) with algebra
# We use a convenient replacement pattern SymPy often yields.
# ------------------------------------------------------------
gamma = sp.sqrt((b-1)/b)
A     = sp.acosh(sp.sqrt(b))

Xi_target = (183*gamma - 38*b*gamma - 88*b**2*gamma + 112*b**2
             + 48*b**3*gamma - 48*b**3 - 105*A/b) / 210
Nu_target = ((87*gamma)/b + 250*gamma - 376*b*gamma + 448*b
             + 144*b**2*gamma - 144*b**2 - 105*A/b**2) / 420

Rlog = sp.sqrt(b*(b-1))
Xi_B_rew = sp.simplify( (Xi_B).xreplace({ sp.log(b) - 2*sp.log(b + Rlog): -2*sp.acosh(sp.sqrt(b)) }) )
Nu_B_rew = sp.simplify( (Nu_B).xreplace({ sp.log(b) - 2*sp.log(b + Rlog): -2*sp.acosh(sp.sqrt(b)) }) )

print("\n=== b>1: rewrite to acosh(sqrt(b)) and compare to targets ===")
show("Xi_B - Xi_target:", sp.simplify(sp.together(Xi_B_rew - Xi_target)))
show("Nu_B - Nu_target:", sp.simplify(sp.together(Nu_B_rew - Nu_target)))
