# ================================================================
#  optimal_affine_jump_debug.py  —  analytic vs numeric checklist
# ================================================================
import numpy as np
from math import sqrt
from functools import lru_cache
from itertools import product
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import roots_legendre  # fast quad for J_num


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------
def clip01(x):  # projection Π[0,1]
    return 0.0 if x < 0 else 1.0 if x > 1 else x


# ----------------------------------------------------------------
# 1)  LOCAL MULTIPLIER  s_v
# ----------------------------------------------------------------
def sv_an(v, q, d):
    """closed-form s_v (Section 3)"""
    d_crit = 1.0 / q - 1.0
    s_star = q * (1 + d)
    v1, v2 = 0.5 * s_star, 1.0 - 0.5 * s_star
    if d <= d_crit:
        if v <= v1:
            return sqrt(2 * q * (1 + d) * v)
        if v <= v2:
            return v + 0.5 * s_star
        return 1.0 + s_star - sqrt(2 * q * (1 + d) * (1 - v))
    else:
        v0 = 1.0 - 1.0 / (2.0 * s_star)
        if v <= v0:
            return 0.5 + q * (1 + d) * v
        return 1.0 + s_star - sqrt(2 * q * (1 + d) * (1 - v))


def _mass_defect(s, v, q, d):
    """∫_0^1 h*  – v    (used by sv_num root-finder)"""
    b = 1.0 / q
    t1 = max(0.0, s - q * (1 + d))  # plateau length
    t0i = s - q * d
    # areas  (triangles / rectangles capped at 1)
    inner_plat = min(t1, v)  # height 1
    inner_ramp = 0.0
    if v > t1:
        lo, hi = max(t1, 0.0), min(v, t0i)
        if hi > lo:
            inner_ramp = b * (s * (hi - lo) - 0.5 * (hi**2 - lo**2)) - d * (hi - lo)
    outer = 0.0
    if v < 1:
        lo, hi = max(v, s - q), min(1.0, s)
        if hi > lo:
            outer = b * (s * (hi - lo) - 0.5 * (hi**2 - lo**2))
    return inner_plat + inner_ramp + outer - v


@lru_cache(None)
def sv_num(v, q, d):
    """numeric s_v  (solve mass constraint directly)"""
    # search bracket:   analytic formula is an excellent initial guess
    s_guess = sv_an(v, q, d)
    left = max(1e-12, s_guess * 0.2)
    right = s_guess * 5 + 5
    f_left, f_right = _mass_defect(left, v, q, d), _mass_defect(right, v, q, d)
    if f_left * f_right > 0:
        raise RuntimeError("s_v root not bracketed")
    return brentq(_mass_defect, left, right, args=(v, q, d), xtol=1e-12)


# ----------------------------------------------------------------
# 2)  OPTIMAL DENSITY  h*_v(t)   (uses chosen s_v)
# ----------------------------------------------------------------
def h_star(t, v, q, d, *, sv_func):
    s = sv_func(v, q, d)
    core = (1.0 / q) * (s - t) - (d if t <= v else 0.0)
    return clip01(core)


# ----------------------------------------------------------------
# 3)  WEDGE INTEGRAL  W(d)
# ----------------------------------------------------------------
def W_an(q, d):
    S = 1.0 + d
    return 0.5 - (q / 8.0) * S**2 + (q * q / 30.0) * S**3  # cubic


def W_num(q, d, *, sv_func):
    """
    Numerical wedge integral W(d) based on an externally supplied
    s_v-function  sv_func(v,q,d).  No analytic shortcuts, but robust.
    """
    b = 1.0 / q  # slope

    # --- local optimiser profile ------------------------------
    def h_star(t, v, q, d):
        """raw optimiser value (already clipped to [0,1])"""
        sv = sv_func(v, q, d)
        core = b * (sv - t) - (d if t <= v else 0.0)
        # clamp to [0,1] (projection operator)
        if core <= 0.0:
            return 0.0
        if core >= 1.0:
            return 1.0
        return core

    # --- inner integral  I(v,d) = ∫₀ᵛ h_v(t) dt ---------------
    def inner_I(vv):
        return quad(h_star, 0.0, vv, args=(vv, q, d), epsabs=1e-10)[0]

    # --- outer integral  W(d) = ∫₀¹ I(v,d) dv -----------------
    return quad(lambda vv: inner_I(vv), 0.0, 1.0, epsabs=1e-10)[0]


# ----------------------------------------------------------------
# 4)  GLOBAL MULTIPLIER  d⋆
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# 4)  GLOBAL MULTIPLIER  d⋆
# ----------------------------------------------------------------
def d_an(q, c):
    """smallest positive analytic root of 4 q²S³−15 qS²+20(1−c)=0 minus 1"""
    coeffs = [4 * q * q, -15 * q, 0.0, 20 * (1 - c)]
    roots = np.roots(coeffs)
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    return roots.min() - 1.0


# ----------------------------------------------------------------
# 4)  GLOBAL MULTIPLIER  d⋆
# ----------------------------------------------------------------
def d_num(q, c, *, W_func):
    """
    Smallest positive root of  6·W(d) − 2 = c
    using Brent’s method.  W_func must be a unary function of d.
    """
    rhs = (c + 2.0) / 6.0

    def f(d):
        return W_func(d) - rhs  # <-- only d !

    return brentq(f, -0.999, 20.0, xtol=1e-12)


# helper that binds q so W_func(d) has one arg -------------------
def make_W_fun(q, use_analytic, sv):
    if use_analytic:
        return lambda d: W_an(q, d)
    return lambda d: W_num(q, d, sv_func=sv)


# ----------------------------------------------------------------
# 5)  COPULA C(u,v)
#     (plain numerical integration of h*, analytic version uses wedge trick)
# ----------------------------------------------------------------
def C_num(u, v, q, d, *, sv_func):
    if u == 0:
        return 0.0
    return quad(
        h_star, 0, u, args=(v, q, d), kwargs={"sv_func": sv_func}, epsabs=1e-10
    )[0]


def C_an(u, v, q, d):
    # wedge/pyramid trick:   C(u,v)=I(u,d)+I(v,d)−I(min(u,v),d)
    # where I(x,d)=∫_0^x h*_x(t)dt  (mass below diagonal)
    def II(x):  # we still need sv formula but closed-form
        return quad(h_star, 0, x, args=(x, q, d), kwargs={"sv_func": sv_an})[0]

    return II(u) + II(v) - II(min(u, v))


# ----------------------------------------------------------------
# 6)  OBJECTIVE   J_max
# ----------------------------------------------------------------
def J_an(q, d):
    # quartic (Section 6, eq. (63))
    return (
        (q**3 / 20) * d**4
        + (2 * q**3 / 15 - q**2 / 6) * d**3
        + (q**3 / 10 - q**2 / 4) * d**2
        - q**3 / 60
        + q**2 / 12
        - q / 2
        + 2 / 3
    )


def J_num(q, d, *, sv_func, nodes=41):
    # Gauss–Legendre tensor grid  (fast & robust)
    xs, ws = roots_legendre(nodes)
    xs = 0.5 * (xs + 1)  # map to [0,1]
    ws = 0.5 * ws
    t, w = np.meshgrid(xs, ws)
    integrand = (
        2 * (1 - t) * h_star(t, xs[:, None], q, d, sv_func=sv_func)
        + (-q) * h_star(t, xs[:, None], q, d, sv_func=sv_func) ** 2
    )
    return np.sum(w * w.T * integrand)


# ----------------------------------------------------------------
# 7)  SPEARMAN FOOT-RULE  ψ
# ----------------------------------------------------------------
def psi_num(q, d, *, sv_func, grid=2001):
    u = np.linspace(0, 1, grid)
    cuu = np.array([C_num(ui, ui, q, d, sv_func=sv_func) for ui in u])
    return 6 * np.trapz(cuu, u) - 2


def psi_an(q, c):  # by definition, equals c
    return c


# ----------------------------------------------------------------
# MASTER DRIVER --------------------------------------------------
# ----------------------------------------------------------------
def compute_all(
    a,
    c,
    use_sv_an=True,
    use_W_an=True,
    use_d_an=True,
    use_C_an=False,
    use_J_an=False,
    verbose=True,
):
    q = -a
    sv = sv_an if use_sv_an else sv_num
    Wf = make_W_fun(q, use_W_an, sv)
    d_star = d_an(q, c) if use_d_an else d_num(q, c, W_func=Wf)

    if verbose:
        print(f"  d* = {d_star:.6f}")
    (
        (lambda u, v: C_an(u, v, q, d_star))
        if use_C_an
        else (lambda u, v: C_num(u, v, q, d_star, sv_func=sv))
    )
    psi_val = (
        psi_an(q, c)
        if (use_C_an and use_d_an and use_W_an)
        else psi_num(q, d_star, sv_func=sv)
    )
    if use_J_an:
        Jv = J_an(q, d_star)
    else:
        Jv = J_num(q, d_star, sv_func=sv)
    return psi_val, Jv


# ----------------------------------------------------------------
if __name__ == "__main__":
    a_vals = [-0.05, -0.1]
    c_vals = [0.8, 0.2, -0.4]

    TESTS = [
        dict(sv=False, W=False, d=False, C=False, J=False, label="all numeric"),
        dict(sv=True, W=False, d=False, C=False, J=False, label="sv analytic"),
        dict(sv=True, W=True, d=True, C=False, J=False, label="up to W,d"),
        dict(sv=True, W=True, d=True, C=True, J=True, label="all analytic"),
    ]

    for a, c in product(a_vals, c_vals):
        print(f"\n>>>>>>>> a={a:+.3f}, c={c:+.3f}")
        for cfg in TESTS:
            psi, Jv = compute_all(
                a,
                c,
                use_sv_an=cfg["sv"],
                use_W_an=cfg["W"],
                use_d_an=cfg["d"],
                use_C_an=cfg["C"],
                use_J_an=cfg["J"],
                verbose=False,
            )
            print(f"{cfg['label']:<15s}:   ψ = {psi:+.8f}   J = {Jv:+.8f}")
