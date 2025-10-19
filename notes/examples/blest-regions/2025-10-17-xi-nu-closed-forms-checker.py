#!/usr/bin/env python3
"""
Check that the closed-form xi and nu match their defining integrals
for the copula family C_b defined via the clamped ramp construction.

Requirements: numpy (and optionally matplotlib if you turn on plotting).
"""

import numpy as np

# ----------------------------
# Utilities
# ----------------------------


def clamp(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def trapz2(f, x, y):
    """
    2D composite trapezoidal rule over a product grid.
    f has shape (len(x), len(y)), where x corresponds to axis 0 and y to axis 1.
    """
    # integrate over x (axis=0), then over y
    ix = np.trapz(f, x, axis=0)
    return float(np.trapz(ix, y))


# ----------------------------
# Phi(q): integral of clamped ramp in t, for fixed b and q
# ----------------------------


def phi_of_q(q, b, t_grid):
    """
    Phi(q) = \int_0^1 clamp(b((1-t)^2 - q), 0, 1) dt
    Numerically via trapezoidal rule on t_grid.
    """
    s = b * ((1.0 - t_grid) ** 2 - q)
    h = clamp(s, 0.0, 1.0)
    return float(np.trapz(h, t_grid))


def q_from_v(v, b, t_grid, tol=1e-10, max_iter=80):
    """
    Solve Phi(q) = v for q in [-1/b, 1] by monotone bisection.
    Phi is continuous, strictly decreasing on this interval.
    """
    lo = -1.0 / b + 0.0
    hi = 1.0
    f_lo = phi_of_q(lo, b, t_grid) - v  # should be >= 0 since Phi(lo)=1
    f_hi = phi_of_q(hi, b, t_grid) - v  # should be <= 0 since Phi(hi)=0
    # Safety checks (allow small numerical slack)
    if f_lo < -1e-8 or f_hi > 1e-8:
        # In rare edge cases, nudge bounds a bit
        lo = min(lo, -1.0 / b - 1e-12)
        hi = max(hi, 1.0 + 1e-12)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = phi_of_q(mid, b, t_grid) - v
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return mid
        # Phi is decreasing in q
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------
# Build h_b(t, v) and compute xi, nu from definitions
# ----------------------------


def build_h_grid(b, Nt=400, Nv=400, t_grid=None, v_grid=None):
    """
    Construct a grid for h_b(t, v) = clamp(b((1-t)^2 - q(v)), 0, 1).
    Returns t_grid, v_grid, and h of shape (Nt, Nv).
    """
    if t_grid is None:
        t_grid = np.linspace(0.0, 1.0, Nt)
    if v_grid is None:
        v_grid = np.linspace(0.0, 1.0, Nv)

    # Precompute q(v) on the v-grid via bisection using the same t_grid for Phi
    q_vals = np.empty_like(v_grid)
    for j, v in enumerate(v_grid):
        q_vals[j] = q_from_v(v, b, t_grid)

    # Build h(t, v) by broadcasting
    T = t_grid[:, None]  # (Nt, 1)
    Q = q_vals[None, :]  # (1, Nv)
    S = b * ((1.0 - T) ** 2 - Q)  # (Nt, Nv)
    H = clamp(S, 0.0, 1.0)
    return t_grid, v_grid, H


def xi_numeric(b, Nt=400, Nv=400):
    """
    xi(C_b) = 6 * \int_0^1 \int_0^1 h(u,v)^2 du dv - 2
    (We can reuse 't' as the u-variable in notation.)
    """
    t, v, H = build_h_grid(b, Nt=Nt, Nv=Nv)
    integrand = H**2
    val = 6.0 * trapz2(integrand, t, v) - 2.0
    return val


def nu_numeric(b, Nt=400, Nv=400):
    """
    nu(C_b) = 12 * \int_0^1 \int_0^1 (1 - t)^2 h(t, v) dt dv - 2
    (Using the derivative form from your equation (intro-nu-forms).)
    """
    t, v, H = build_h_grid(b, Nt=Nt, Nv=Nv)
    W = (1.0 - t)[:, None] ** 2  # weights in t
    integrand = W * H
    val = 12.0 * trapz2(integrand, t, v) - 2.0
    return val


# ----------------------------
# Closed-form xi and nu from the paper, piecewise in b
# ----------------------------


# ----------------------------
# Closed-form xi and nu from the theorem (Exact (xi,nu)-region)
# ----------------------------


def xi_closed(b: float) -> float:
    """
    Xi(b) from Theorem:
      - 0 < b <= 1:  8 b^2 (7 - 3 b) / 105
      - b > 1: [ 183 γ - 38 b γ - 88 b^2 γ + 111 b^2 + 48 b^3 γ - 48 b^3
                 - (105 acosh(sqrt(b)))/b ] / 210,
        where γ = sqrt((b-1)/b).
    """
    assert b > 0.0, "Closed-form given here assumes b>0."
    if b <= 1.0:
        return (8.0 * b**2 * (7.0 - 3.0 * b)) / 105.0
    # b > 1
    gamma = np.sqrt((b - 1.0) / b)
    A = np.arccosh(np.sqrt(b))
    num = (
        183.0 * gamma
        - 38.0 * b * gamma
        - 88.0 * b**2 * gamma
        + 112.0 * b**2
        + 48.0 * b**3 * gamma
        - 48.0 * b**3
        - (105.0 * A) / b
    )
    den = 210.0
    return num / den


def nu_closed(b: float) -> float:
    """
    N(b) from Theorem:
      - 0 < b <= 1:  4 b (28 - 9 b) / 105
      - b > 1: [ (87 γ)/b + 250 γ - 376 b γ + 447 b + 144 b^2 γ - 144 b^2
                 - (105 acosh(sqrt(b)))/b^2 ] / 420,
        where γ = sqrt((b-1)/b).
    """
    assert b > 0.0, "Closed-form given here assumes b>0."
    if b <= 1.0:
        return (4.0 * b * (28.0 - 9.0 * b)) / 105.0
    # b > 1
    gamma = np.sqrt((b - 1.0) / b)
    A = np.arccosh(np.sqrt(b))
    num = (
        (87.0 * gamma) / b
        + 250.0 * gamma
        - 376.0 * b * gamma
        + 448.0 * b
        + 144.0 * b**2 * gamma
        - 144.0 * b**2
        - (105.0 * A) / (b**2)
    )
    den = 420.0
    return num / den


# ----------------------------
# Demo & comparison
# ----------------------------


def compare(b_list=(0.5, 1.0, 5.0, 10.0), Nt=500, Nv=500):
    print("Comparing numeric integrals vs closed forms for C_b (b>0).")
    print(f"Grids: Nt={Nt}, Nv={Nv}\n")
    for b in b_list:
        print(f"b = {b}")
        xi_num = xi_numeric(b, Nt=Nt, Nv=Nv)
        nu_num = nu_numeric(b, Nt=Nt, Nv=Nv)
        xi_cl = xi_closed(b)
        nu_cl = nu_closed(b)

        def rel_err(a, b):
            denom = max(1.0, abs(b))
            return abs(a - b) / denom

        print(f"  xi_numeric = {xi_num:.12f}")
        print(f"  xi_closed  = {xi_cl:.12f}")
        print(
            f"  abs err    = {abs(xi_num - xi_cl):.3e},  rel err = {rel_err(xi_num, xi_cl):.3e}"
        )

        print(f"  nu_numeric = {nu_num:.12f}")
        print(f"  nu_closed  = {nu_cl:.12f}")
        print(
            f"  abs err    = {abs(nu_num - nu_cl):.3e},  rel err = {rel_err(nu_num, nu_cl):.3e}"
        )
        print()


if __name__ == "__main__":
    # Example run; increase Nt, Nv for higher accuracy (at higher compute cost).
    compare(b_list=(0.25, 0.5, 1.0, 2.0, 5.0), Nt=400, Nv=400)
