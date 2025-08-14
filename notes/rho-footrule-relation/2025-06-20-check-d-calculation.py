import numpy as np


# --------------------------------------------------------------------
# 1. The closed-form wedge integral
# --------------------------------------------------------------------
def W(q, d):
    """
    Closed-form wedge integral

        W(d) = ½ − (q/8)(1+d)² + (q²/30)(1+d)³.

    Parameters
    ----------
    q : float   # q = −a,    0 < q < ½
    d : float   # jump size, 0 < d < 1
    """
    one_plus_d = 1.0 + d
    return 0.5 - (q * one_plus_d**2) / 8.0 + (q**2) * (one_plus_d**3) / 30.0


# --------------------------------------------------------------------
# 2. “Closed-form” solution for d: real root of the cubic in (0,1)
# --------------------------------------------------------------------
def d_closed_form(q, c):
    """
    Returns the unique admissible root of

        4 q² (1+d)³ − 15 q (1+d)² + 20(1−c) = 0

    that lies in the open interval (0,1).  If no such root exists
    (which may happen when the target value (c+2)/6 is outside the
    attainable range of W on [0,1]) the function returns None.
    """
    # polynomial coefficients in S = 1+d
    coeffs = [4 * q**2, -15 * q, 0.0, 20 * (1 - c)]
    roots = np.roots(coeffs)

    # admissible real roots for S must satisfy 1 < S < 2
    S_candidates = [r.real for r in roots if abs(r.imag) < 1e-10 and 1.0 < r.real < 2.0]

    if not S_candidates:  # infeasible (q, c) pair
        return None

    S = S_candidates[0]  # there is exactly one
    return S - 1.0  # convert back to d


# --------------------------------------------------------------------
# 3. High-precision numerical root (bisection, monotone)
# --------------------------------------------------------------------
def d_numeric(q, c, tol=1e-12, max_iter=200):
    """
    Solves W(d) = (c+2)/6 for d in (0,1) by bisection.
    Returns None if the continuous solution lies outside (0,1).
    """
    target = (c + 2.0) / 6.0

    def f(d):
        return W(q, d) - target

    lo, hi = 0.0, 1.0
    f_lo, f_hi = f(lo), f(hi)

    # If f has the same sign at both ends, no root in (0,1)
    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol:
            return mid
        if f_mid * f_lo > 0:  # root in upper half
            lo, f_lo = mid, f_mid
        else:  # root in lower half
            hi, f_hi = mid, f_mid
    return mid  # final approximation


# --------------------------------------------------------------------
# 4. Demonstration / sanity check
# --------------------------------------------------------------------
def run_checks(n_tests=15, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_tests):
        q = rng.uniform(0.05, 0.49)  #  0 < q < ½
        c = rng.uniform(-0.5, 1.0)  # −½ ≤ c ≤ 1

        d_cf = d_closed_form(q, c)
        d_num = d_numeric(q, c)

        if d_cf is None or d_num is None:
            print(f"q={q:5.3f}, c={c:6.3f}:   ✗  no admissible root in (0,1)")
        else:
            print(
                f"q={q:5.3f}, c={c:6.3f}:   "
                f"d_closed={d_cf:.10f},  "
                f"d_numeric={d_num:.10f},  "
                f"|Δ|={abs(d_cf - d_num):.2e}"
            )


if __name__ == "__main__":
    run_checks()
