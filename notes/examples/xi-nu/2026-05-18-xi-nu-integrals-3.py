#!/usr/bin/env python3
"""
Numerical checker for the appendix lemmata in the xi/nu boundary-family paper.

Checks:
  1. Lemma 1D:
       ∫ h dt,
       ∫ h^2 dt,
       ∫ (1-t)^2 h dt
     against numerical quadrature.

  2. Lemma vq:
       substitution weights -dq/dR * v'(q) or -dq/dr * v'(q)
     against finite differences.

  3. Theorem closed:
       regime-integral reconstruction of xi(C_b), nu(C_b)
     against the stated closed forms Xi(b), N(b).

Requires only numpy.
"""

import math
import numpy as np


# ---------------------------------------------------------------------
# Basic definitions
# ---------------------------------------------------------------------


def h_section(t, b, q):
    """h_b^(q)(t) = clamp(b((1-t)^2 - q), 0, 1)."""
    return np.clip(b * ((1.0 - t) ** 2 - q), 0.0, 1.0)


def F(x, q):
    return x**5 / 5.0 - 2.0 * q * x**3 / 3.0 + q**2 * x


def T(x, q):
    return x**3 / 3.0 - q * x


def S(x, q):
    return x**5 / 5.0 - q * x**3 / 3.0


def switches(b, q):
    """
    Works for both scalar q and numpy array q.
    """
    R = np.sqrt(np.maximum(0.0, q + 1.0 / b))
    r = np.where(q >= 0.0, np.sqrt(np.maximum(0.0, q)), 0.0)
    X_a = np.minimum(1.0, R)
    X_s = np.where(q >= 0.0, r, 0.0)
    a = np.maximum(0.0, 1.0 - R)
    s = 1.0 - X_s
    return R, r, X_a, X_s, a, s


def lemma_1d_formulas(b, q):
    """
    Lemma 1D formulas — scalar or array q.
    """
    R, r, X_a, X_s, a, s = switches(b, q)

    marginal = a + b * (T(X_a, q) - T(X_s, q))
    square = a + b**2 * (F(X_a, q) - F(X_s, q))
    weighted = (1.0 - (1.0 - a) ** 3) / 3.0 + b * (S(X_a, q) - S(X_s, q))

    return marginal, square, weighted


# ---------------------------------------------------------------------
# Numerical quadrature
# ---------------------------------------------------------------------


class GaussLegendre:
    def __init__(self, n=800):
        x, w = np.polynomial.legendre.leggauss(n)
        self.x = x
        self.w = w

    def integrate(self, f, a, b):
        if b <= a:
            return 0.0
        z = 0.5 * (b - a) * self.x + 0.5 * (a + b)
        return 0.5 * (b - a) * np.sum(self.w * f(z))


quad = GaussLegendre(n=800)


def lemma_1d_numeric(b, q):
    """Direct numerical integration of the three section integrals."""
    I0 = quad.integrate(lambda t: h_section(t, b, q), 0.0, 1.0)
    I2 = quad.integrate(lambda t: h_section(t, b, q) ** 2, 0.0, 1.0)
    Iw = quad.integrate(lambda t: (1.0 - t) ** 2 * h_section(t, b, q), 0.0, 1.0)
    return I0, I2, Iw


# ---------------------------------------------------------------------
# Closed forms Xi and N
# ---------------------------------------------------------------------


def Xi_closed(b):
    if b <= 1.0:
        return 8.0 * b**2 * (7.0 - 3.0 * b) / 105.0

    eta = math.sqrt((b - 1.0) / b)
    return (
        183.0 * eta
        - 38.0 * b * eta
        - 88.0 * b**2 * eta
        + 112.0 * b**2
        + 48.0 * b**3 * eta
        - 48.0 * b**3
        - 105.0 * math.acosh(math.sqrt(b)) / b
    ) / 210.0


def N_closed(b):
    if b <= 1.0:
        return 4.0 * b * (28.0 - 9.0 * b) / 105.0

    eta = math.sqrt((b - 1.0) / b)
    return (
        87.0 * eta / b
        + 250.0 * eta
        - 376.0 * b * eta
        + 448.0 * b
        + 144.0 * b**2 * eta
        - 144.0 * b**2
        - 105.0 * math.acosh(math.sqrt(b)) / b**2
    ) / 420.0


# ---------------------------------------------------------------------
# Regime integrands from Lemma 1D
# ---------------------------------------------------------------------


def G_xi(b, q):
    """Inner integrand for xi before multiplying by substitution weight."""
    _, sq, _ = lemma_1d_formulas(b, q)
    return sq


def H_nu(b, q):
    """Inner integrand for nu before multiplying by substitution weight."""
    _, _, weighted = lemma_1d_formulas(b, q)
    return weighted


def integrate_closed_from_regimes(b):
    """
    Reconstruct xi and nu by applying Lemma 1D + Lemma vq regime substitutions.

    xi = 6 * ∫ G(q) (-v'(q)) dq - 2
    nu = 12 * ∫ H(q) (-v'(q)) dq - 2
    """

    xi_int = 0.0
    nu_int = 0.0

    if b <= 1.0:
        # (i) upper-clamped:
        # q = R^2 - 1/b, R in [0,1], weight = 2b R^2 dR
        def add_upper(R):
            q = R**2 - 1.0 / b
            w = 2.0 * b * R**2
            return G_xi(b, q) * w

        def add_upper_nu(R):
            q = R**2 - 1.0 / b
            w = 2.0 * b * R**2
            return H_nu(b, q) * w

        xi_int += quad.integrate(add_upper, 0.0, 1.0)
        nu_int += quad.integrate(add_upper_nu, 0.0, 1.0)

        # (ii) unclamped:
        # q = R^2 - 1/b, R in [1,1/sqrt(b)], weight = 2b R dR
        Rmax = 1.0 / math.sqrt(b)

        def add_unclamped(R):
            q = R**2 - 1.0 / b
            w = 2.0 * b * R
            return G_xi(b, q) * w

        def add_unclamped_nu(R):
            q = R**2 - 1.0 / b
            w = 2.0 * b * R
            return H_nu(b, q) * w

        xi_int += quad.integrate(add_unclamped, 1.0, Rmax)
        nu_int += quad.integrate(add_unclamped_nu, 1.0, Rmax)

        # (iv) lower-clamped:
        # q = r^2, r in [0,1], weight = 2b r(1-r) dr
        def add_lower(r):
            q = r**2
            w = 2.0 * b * r * (1.0 - r)
            return G_xi(b, q) * w

        def add_lower_nu(r):
            q = r**2
            w = 2.0 * b * r * (1.0 - r)
            return H_nu(b, q) * w

        xi_int += quad.integrate(add_lower, 0.0, 1.0)
        nu_int += quad.integrate(add_lower_nu, 0.0, 1.0)

    else:
        # (i) upper-clamped:
        # q = R^2 - 1/b, R in [0,1/sqrt(b)], weight = 2b R^2 dR
        R0 = 1.0 / math.sqrt(b)

        def add_upper(R):
            q = R**2 - 1.0 / b
            w = 2.0 * b * R**2
            return G_xi(b, q) * w

        def add_upper_nu(R):
            q = R**2 - 1.0 / b
            w = 2.0 * b * R**2
            return H_nu(b, q) * w

        xi_int += quad.integrate(add_upper, 0.0, R0)
        nu_int += quad.integrate(add_upper_nu, 0.0, R0)

        # (iii) double-clamped:
        # q = R^2 - 1/b, R in [1/sqrt(b),1],
        # r = sqrt(R^2 - 1/b), weight = 1 + b(R-r)^2
        def add_double(R):
            q = R**2 - 1.0 / b
            r = math.sqrt(max(0.0, q))
            w = 1.0 + b * (R - r) ** 2
            return G_xi(b, q) * w

        def add_double_nu(R):
            q = R**2 - 1.0 / b
            r = math.sqrt(max(0.0, q))
            w = 1.0 + b * (R - r) ** 2
            return H_nu(b, q) * w

        xi_int += quad.integrate(add_double, R0, 1.0)
        nu_int += quad.integrate(add_double_nu, R0, 1.0)

        # (iv) lower-clamped:
        # q = r^2, r in [sqrt(1-1/b),1], weight = 2b r(1-r) dr
        r0 = math.sqrt(1.0 - 1.0 / b)

        def add_lower(r):
            q = r**2
            w = 2.0 * b * r * (1.0 - r)
            return G_xi(b, q) * w

        def add_lower_nu(r):
            q = r**2
            w = 2.0 * b * r * (1.0 - r)
            return H_nu(b, q) * w

        xi_int += quad.integrate(add_lower, r0, 1.0)
        nu_int += quad.integrate(add_lower_nu, r0, 1.0)

    xi = 6.0 * xi_int - 2.0
    nu = 12.0 * nu_int - 2.0
    return xi, nu


# ---------------------------------------------------------------------
# Lemma vq finite-difference checks
# ---------------------------------------------------------------------


def v_of_q(b, q):
    """v(q) = Phi(q), using Lemma 1D marginal formula."""
    return lemma_1d_formulas(b, q)[0]


def central_diff(f, x, eps):
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


def check_vq_one(b, regime, z):
    """
    Check Lemma vq substitution weights.

    regime:
      'upper'      z = R
      'unclamped'  z = R
      'double'     z = R
      'lower'      z = r
    """
    eps = 1e-6

    if regime in ("upper", "unclamped", "double"):
        R = z
        q = R**2 - 1.0 / b

        def q_of_R(x):
            return x**2 - 1.0 / b

        def v_as_R(x):
            return v_of_q(b, q_of_R(x))

        numeric_weight = -central_diff(v_as_R, R, eps)

        if regime == "upper":
            formula_weight = 2.0 * b * R**2
        elif regime == "unclamped":
            formula_weight = 2.0 * b * R
        else:
            r = math.sqrt(max(0.0, q))
            formula_weight = 1.0 + b * (R - r) ** 2

    elif regime == "lower":
        r = z

        def q_of_r(x):
            return x**2

        def v_as_r(x):
            return v_of_q(b, q_of_r(x))

        numeric_weight = -central_diff(v_as_r, r, eps)
        formula_weight = 2.0 * b * r * (1.0 - r)

    else:
        raise ValueError(regime)

    return numeric_weight, formula_weight, abs(numeric_weight - formula_weight)


# ---------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------


def check_lemma_1d():
    print("\nChecking Lemma 1D...")
    max_err = 0.0
    worst = None

    b_values = [0.05, 0.1, 0.3, 0.8, 1.0, 1.2, 2.0, 5.0, 20.0]

    for b in b_values:
        # Avoid endpoints, where finite numerical quadrature is less informative.
        q_values = np.linspace(-1.0 / b + 1e-5, 1.0 - 1e-5, 41)

        for q in q_values:
            exact = lemma_1d_formulas(b, q)
            numeric = lemma_1d_numeric(b, q)
            err = max(abs(exact[i] - numeric[i]) for i in range(3))

            if err > max_err:
                max_err = err
                worst = (b, q, exact, numeric)

    print(f"  max error: {max_err:.3e}")
    if worst:
        b, q, exact, numeric = worst
        print(f"  worst case b={b}, q={q}")
        print(f"  formula = {exact}")
        print(f"  numeric = {numeric}")


def check_lemma_vq():
    print("\nChecking Lemma vq...")
    max_err = 0.0
    worst = None

    # Upper-clamped: all b; choose R in valid open intervals.
    for b in [0.2, 0.8, 1.0, 1.5, 5.0, 20.0]:
        Rmax = min(1.0, 1.0 / math.sqrt(b))
        for R in np.linspace(0.05 * Rmax, 0.95 * Rmax, 10):
            num, form, err = check_vq_one(b, "upper", R)
            if err > max_err:
                max_err, worst = err, (b, "upper", R, num, form)

    # Unclamped: only b <= 1, R in [1, 1/sqrt(b)].
    for b in [0.1, 0.3, 0.8]:
        Rmin, Rmax = 1.0, 1.0 / math.sqrt(b)
        for R in np.linspace(Rmin + 1e-4, Rmax - 1e-4, 10):
            num, form, err = check_vq_one(b, "unclamped", R)
            if err > max_err:
                max_err, worst = err, (b, "unclamped", R, num, form)

    # Double-clamped: only b > 1, R in [1/sqrt(b), 1].
    for b in [1.2, 2.0, 5.0, 20.0]:
        Rmin, Rmax = 1.0 / math.sqrt(b), 1.0
        for R in np.linspace(Rmin + 1e-4, Rmax - 1e-4, 10):
            num, form, err = check_vq_one(b, "double", R)
            if err > max_err:
                max_err, worst = err, (b, "double", R, num, form)

    # Lower-clamped.
    for b in [0.1, 0.8, 1.0, 1.2, 2.0, 5.0, 20.0]:
        rmin = math.sqrt(max(0.0, 1.0 - 1.0 / b))
        for r in np.linspace(rmin + 1e-4, 1.0 - 1e-4, 10):
            num, form, err = check_vq_one(b, "lower", r)
            if err > max_err:
                max_err, worst = err, (b, "lower", r, num, form)

    print(f"  max error: {max_err:.3e}")
    if worst:
        b, reg, z, num, form = worst
        print(f"  worst case b={b}, regime={reg}, variable={z}")
        print(f"  numeric weight = {num}")
        print(f"  formula weight = {form}")


def check_final_closed_forms():
    print("\nChecking final Xi(b), N(b) closed forms via regime integration...")
    max_err = 0.0
    worst = None

    b_values = [
        0.02,
        0.05,
        0.1,
        0.3,
        0.8,
        1.0,
        1.0001,
        1.01,
        1.2,
        2.0,
        5.0,
        20.0,
        100.0,
    ]

    for b in b_values:
        xi_num, nu_num = integrate_closed_from_regimes(b)
        xi_ex = Xi_closed(b)
        nu_ex = N_closed(b)

        err = max(abs(xi_num - xi_ex), abs(nu_num - nu_ex))

        print(
            f"  b={b:8g}  "
            f"xi_num={xi_num:.15f}  xi_closed={xi_ex:.15f}  "
            f"nu_num={nu_num:.15f}  nu_closed={nu_ex:.15f}  "
            f"err={err:.3e}"
        )

        if err > max_err:
            max_err = err
            worst = (b, xi_num, xi_ex, nu_num, nu_ex)

    print(f"\n  max error: {max_err:.3e}")
    if worst:
        b, xi_num, xi_ex, nu_num, nu_ex = worst
        print(f"  worst case b={b}")
        print(f"  xi_num={xi_num}, xi_closed={xi_ex}")
        print(f"  nu_num={nu_num}, nu_closed={nu_ex}")


def main():
    check_lemma_1d()
    check_lemma_vq()
    check_final_closed_forms()


if __name__ == "__main__":
    main()
