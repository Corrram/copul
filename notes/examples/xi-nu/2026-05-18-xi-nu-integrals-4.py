#!/usr/bin/env python3
"""
Numerically check the individual appendix contributions

  I^{(i)}_{xi,A}, I^{(ii)}_{xi,A}, I^{(iv)}_{xi,A},
  I^{(i)}_{nu,A}, I^{(ii)}_{nu,A}, I^{(iv)}_{nu,A},

and

  I^{(i)}_{xi,B}, I^{(iii)}_{xi,B}, I^{(iv)}_{xi,B},
  I^{(i)}_{nu,B}, I^{(iii)}_{nu,B}, I^{(iv)}_{nu,B}.

For b > 1 there is no regime (ii); the relevant middle regime is (iii).

This version includes the two corrections:
  1. G_iv has -8 r^5 / 15.
  2. I^{(i)}_{nu,B} has -1/(30 b^2), not -1/(30 b).
"""

import math


# ---------------------------------------------------------------------
# Numerical integration
# ---------------------------------------------------------------------


def simpson(f, a, b, n=20000):
    """
    Composite Simpson rule.
    n must be even.
    """
    if b == a:
        return 0.0

    if n % 2:
        n += 1

    h = (b - a) / n
    s = f(a) + f(b)

    for k in range(1, n):
        x = a + k * h
        s += (4 if k % 2 else 2) * f(x)

    return s * h / 3.0


def safe_sqrt(x, tol=1e-14):
    """
    sqrt with protection against tiny negative floating-point roundoff.

    Example:
        at R = 1/sqrt(b), mathematically R**2 - 1/b = 0,
        but floating point may produce -1e-17.
    """
    if x < 0.0 and abs(x) < tol:
        x = 0.0
    if x < 0.0:
        raise ValueError(f"sqrt argument is negative beyond tolerance: {x}")
    return math.sqrt(x)


# ---------------------------------------------------------------------
# Regime integrands G_i, H_i, etc.
# ---------------------------------------------------------------------


def G_i(R, b):
    q = R**2 - 1.0 / b
    return (1.0 - R) + b**2 * (R**5 / 5.0 - 2.0 * q * R**3 / 3.0 + q**2 * R)


def H_i(R, b):
    q = R**2 - 1.0 / b
    return (1.0 - R**3) / 3.0 + b * (R**5 / 5.0 - q * R**3 / 3.0)


def G_ii(R, b):
    q = R**2 - 1.0 / b
    return b**2 * (1.0 / 5.0 - 2.0 * q / 3.0 + q**2)


def H_ii(R, b):
    q = R**2 - 1.0 / b
    return b * (1.0 / 5.0 - q / 3.0)


def G_iii(R, b):
    q = R**2 - 1.0 / b
    r = safe_sqrt(q)
    return (1.0 - R) + b**2 * (
        R**5 / 5.0
        - 2.0 * q * R**3 / 3.0
        + q**2 * R
        - r**5 / 5.0
        + 2.0 * q * r**3 / 3.0
        - q**2 * r
    )


def H_iii(R, b):
    q = R**2 - 1.0 / b
    r = safe_sqrt(q)
    return (1.0 - R**3) / 3.0 + b * (
        R**5 / 5.0 - q * R**3 / 3.0 - r**5 / 5.0 + q * r**3 / 3.0
    )


def G_iv(r, b):
    return b**2 * (1.0 / 5.0 - 2.0 * r**2 / 3.0 + r**4 - 8.0 * r**5 / 15.0)


def H_iv(r, b):
    return b * (1.0 / 5.0 - r**2 / 3.0 - r**5 / 5.0 + r**5 / 3.0)


# ---------------------------------------------------------------------
# Numerical definitions of the I-terms
# ---------------------------------------------------------------------


def I_numeric(name, b, n=20000):
    """
    Numerically evaluate the named I-term using the regime integral
    definitions from the appendix.
    """

    # A-case, 0 < b <= 1
    if name == "I_xi_A_i":
        return simpson(
            lambda R: G_i(R, b) * (2.0 * b * R**2),
            0.0,
            1.0,
            n,
        )

    if name == "I_xi_A_ii":
        return simpson(
            lambda R: G_ii(R, b) * (2.0 * b * R),
            1.0,
            1.0 / math.sqrt(b),
            n,
        )

    if name == "I_xi_A_iv":
        return simpson(
            lambda r: G_iv(r, b) * (2.0 * b * r * (1.0 - r)),
            0.0,
            1.0,
            n,
        )

    if name == "I_nu_A_i":
        return simpson(
            lambda R: H_i(R, b) * (2.0 * b * R**2),
            0.0,
            1.0,
            n,
        )

    if name == "I_nu_A_ii":
        return simpson(
            lambda R: H_ii(R, b) * (2.0 * b * R),
            1.0,
            1.0 / math.sqrt(b),
            n,
        )

    if name == "I_nu_A_iv":
        return simpson(
            lambda r: H_iv(r, b) * (2.0 * b * r * (1.0 - r)),
            0.0,
            1.0,
            n,
        )

    # B-case, b > 1
    if name == "I_xi_B_i":
        return simpson(
            lambda R: G_i(R, b) * (2.0 * b * R**2),
            0.0,
            1.0 / math.sqrt(b),
            n,
        )

    if name == "I_xi_B_iii":
        return simpson(
            lambda R: G_iii(R, b) * (1.0 + b * (R - safe_sqrt(R**2 - 1.0 / b)) ** 2),
            1.0 / math.sqrt(b),
            1.0,
            n,
        )

    if name == "I_xi_B_iv":
        return simpson(
            lambda r: G_iv(r, b) * (2.0 * b * r * (1.0 - r)),
            math.sqrt(1.0 - 1.0 / b),
            1.0,
            n,
        )

    if name == "I_nu_B_i":
        return simpson(
            lambda R: H_i(R, b) * (2.0 * b * R**2),
            0.0,
            1.0 / math.sqrt(b),
            n,
        )

    if name == "I_nu_B_iii":
        return simpson(
            lambda R: H_iii(R, b) * (1.0 + b * (R - safe_sqrt(R**2 - 1.0 / b)) ** 2),
            1.0 / math.sqrt(b),
            1.0,
            n,
        )

    if name == "I_nu_B_iv":
        return simpson(
            lambda r: H_iv(r, b) * (2.0 * b * r * (1.0 - r)),
            math.sqrt(1.0 - 1.0 / b),
            1.0,
            n,
        )

    raise ValueError(f"Unknown integral name: {name}")


# ---------------------------------------------------------------------
# Claimed closed forms for the I-terms
# ---------------------------------------------------------------------


def I_formula(name, b):
    """
    Closed forms from the appendix for the individual I-terms.
    """

    sqrt_b = math.sqrt(b)

    # A-case, 0 < b <= 1
    if name == "I_xi_A_i":
        return 2.0 * b * (3.0 * b**2 - 10.0 * b + 15.0) / 45.0

    if name == "I_xi_A_ii":
        return -(b**3) / 5.0 + 8.0 * b**2 / 15.0 - 2.0 * b / 3.0 + 1.0 / 3.0

    if name == "I_xi_A_iv":
        return b**3 / 35.0

    if name == "I_nu_A_i":
        return b * (20.0 - 3.0 * b) / 90.0

    if name == "I_nu_A_ii":
        return -(b**2) / 30.0 - 2.0 * b / 15.0 + 1.0 / 6.0

    if name == "I_nu_A_iv":
        return 4.0 * b**2 / 105.0

    # B-case, b > 1
    sqrt_b_minus_1 = math.sqrt(b - 1.0)
    log_b = math.log(b)

    # log(sqrt(b)*sqrt(b-1)+b) = acosh(sqrt(b)) + 0.5 log(b)
    log_term = math.log(math.sqrt(b) * math.sqrt(b - 1.0) + b)

    if name == "I_xi_B_i":
        return -14.0 / (45.0 * b) + 2.0 / (3.0 * sqrt_b)

    if name == "I_xi_B_iii":
        return (
            96.0 * b ** (9.0 / 2.0)
            - 352.0 * b ** (7.0 / 2.0)
            + 528.0 * b ** (5.0 / 2.0)
            - 192.0 * b ** (3.0 / 2.0)
            + 15.0 * sqrt_b * log_b
            - 30.0 * sqrt_b * log_term
            + 160.0 * sqrt_b
            - 96.0 * b**4 * sqrt_b_minus_1
            + 304.0 * b**3 * sqrt_b_minus_1
            - 388.0 * b**2 * sqrt_b_minus_1
            + 210.0 * b * sqrt_b_minus_1
            - 240.0 * b
        ) / (360.0 * b ** (3.0 / 2.0))

    if name == "I_xi_B_iv":
        return (
            -32.0 * b ** (9.0 / 2.0)
            + 112.0 * b ** (7.0 / 2.0)
            - 154.0 * b ** (5.0 / 2.0)
            + 91.0 * b ** (3.0 / 2.0)
            - 14.0 * sqrt_b
            + 32.0 * b**4 * sqrt_b_minus_1
            - 96.0 * b**3 * sqrt_b_minus_1
            + 110.0 * b**2 * sqrt_b_minus_1
            - 46.0 * b * sqrt_b_minus_1
        ) / (105.0 * b ** (3.0 / 2.0))

    if name == "I_nu_B_i":
        return -1.0 / (30.0 * b**2) + 2.0 / (9.0 * sqrt_b)

    if name == "I_nu_B_iii":
        return (
            b ** (3.0 / 2.0) * sqrt_b_minus_1 / 15.0
            - 29.0 * sqrt_b * sqrt_b_minus_1 / 90.0
            - b**2 / 15.0
            + 16.0 * b / 45.0
            - 1.0 / 5.0
            + 2.0 / (15.0 * b)
            + log_b / (96.0 * b**2)
            - log_term / (48.0 * b**2)
            + 107.0 * sqrt_b_minus_1 / (360.0 * sqrt_b)
            - 2.0 / (9.0 * sqrt_b)
            - sqrt_b_minus_1 / (48.0 * b ** (3.0 / 2.0))
        )

    if name == "I_nu_B_iv":
        return (
            2.0 * sqrt_b * (b - 1.0) ** (3.0 / 2.0) / 15.0
            - 17.0 * b**2 / 105.0
            + b / 5.0
            + (b - 1.0) ** 2 / 6.0
            + (b - 1.0) ** 4 / (30.0 * b**2)
            - 2.0 * (b - 1.0) ** (5.0 / 2.0) / (15.0 * sqrt_b)
            - 4.0 * (b - 1.0) ** (7.0 / 2.0) / (105.0 * b ** (3.0 / 2.0))
        )

    raise ValueError(f"Unknown integral name: {name}")


# ---------------------------------------------------------------------
# Optional aggregate checks
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


def aggregate_from_I_terms_numeric(b, n=20000):
    """
    Reconstruct xi and nu from the numerical I-terms:
      xi = 6 * sum I_xi - 2,
      nu = 12 * sum I_nu - 2.
    """
    if b <= 1.0:
        xi_sum = (
            I_numeric("I_xi_A_i", b, n)
            + I_numeric("I_xi_A_ii", b, n)
            + I_numeric("I_xi_A_iv", b, n)
        )
        nu_sum = (
            I_numeric("I_nu_A_i", b, n)
            + I_numeric("I_nu_A_ii", b, n)
            + I_numeric("I_nu_A_iv", b, n)
        )
    else:
        xi_sum = (
            I_numeric("I_xi_B_i", b, n)
            + I_numeric("I_xi_B_iii", b, n)
            + I_numeric("I_xi_B_iv", b, n)
        )
        nu_sum = (
            I_numeric("I_nu_B_i", b, n)
            + I_numeric("I_nu_B_iii", b, n)
            + I_numeric("I_nu_B_iv", b, n)
        )

    return 6.0 * xi_sum - 2.0, 12.0 * nu_sum - 2.0


def aggregate_from_I_terms_formula(b):
    """
    Reconstruct xi and nu from the closed-form I-terms.
    """
    if b <= 1.0:
        xi_sum = (
            I_formula("I_xi_A_i", b)
            + I_formula("I_xi_A_ii", b)
            + I_formula("I_xi_A_iv", b)
        )
        nu_sum = (
            I_formula("I_nu_A_i", b)
            + I_formula("I_nu_A_ii", b)
            + I_formula("I_nu_A_iv", b)
        )
    else:
        xi_sum = (
            I_formula("I_xi_B_i", b)
            + I_formula("I_xi_B_iii", b)
            + I_formula("I_xi_B_iv", b)
        )
        nu_sum = (
            I_formula("I_nu_B_i", b)
            + I_formula("I_nu_B_iii", b)
            + I_formula("I_nu_B_iv", b)
        )

    return 6.0 * xi_sum - 2.0, 12.0 * nu_sum - 2.0


# ---------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------


def check_family(
    b_values_A=(0.05, 0.2, 0.5, 0.9, 1.0),
    b_values_B=(1.01, 1.1, 2.0, 5.0, 20.0),
    n=20000,
):
    names_A = [
        "I_xi_A_i",
        "I_xi_A_ii",
        "I_xi_A_iv",
        "I_nu_A_i",
        "I_nu_A_ii",
        "I_nu_A_iv",
    ]

    names_B = [
        "I_xi_B_i",
        "I_xi_B_iii",
        "I_xi_B_iv",
        "I_nu_B_i",
        "I_nu_B_iii",
        "I_nu_B_iv",
    ]

    max_err = 0.0
    worst = None

    print("\nA-case: 0 < b <= 1")
    for b in b_values_A:
        print(f"\nb = {b}")
        for name in names_A:
            num = I_numeric(name, b, n=n)
            exact = I_formula(name, b)
            err = num - exact

            if abs(err) > max_err:
                max_err = abs(err)
                worst = (name, b, num, exact, err)

            print(
                f"  {name:14s}  "
                f"numeric={num:+.15e}  "
                f"formula={exact:+.15e}  "
                f"err={err:+.3e}"
            )

        xi_num, nu_num = aggregate_from_I_terms_numeric(b, n=n)
        xi_I, nu_I = aggregate_from_I_terms_formula(b)
        xi_closed, nu_closed = Xi_closed(b), N_closed(b)

        print("  aggregate:")
        print(
            f"    xi numeric I={xi_num:+.15e}, "
            f"xi formula I={xi_I:+.15e}, "
            f"Xi closed={xi_closed:+.15e}"
        )
        print(
            f"    nu numeric I={nu_num:+.15e}, "
            f"nu formula I={nu_I:+.15e}, "
            f"N closed={nu_closed:+.15e}"
        )

    print("\nB-case: b > 1")
    for b in b_values_B:
        print(f"\nb = {b}")
        for name in names_B:
            num = I_numeric(name, b, n=n)
            exact = I_formula(name, b)
            err = num - exact

            if abs(err) > max_err:
                max_err = abs(err)
                worst = (name, b, num, exact, err)

            print(
                f"  {name:14s}  "
                f"numeric={num:+.15e}  "
                f"formula={exact:+.15e}  "
                f"err={err:+.3e}"
            )

        xi_num, nu_num = aggregate_from_I_terms_numeric(b, n=n)
        xi_I, nu_I = aggregate_from_I_terms_formula(b)
        xi_closed, nu_closed = Xi_closed(b), N_closed(b)

        print("  aggregate:")
        print(
            f"    xi numeric I={xi_num:+.15e}, "
            f"xi formula I={xi_I:+.15e}, "
            f"Xi closed={xi_closed:+.15e}"
        )
        print(
            f"    nu numeric I={nu_num:+.15e}, "
            f"nu formula I={nu_I:+.15e}, "
            f"N closed={nu_closed:+.15e}"
        )

    print("\nWorst absolute individual I-term error:")
    if worst is None:
        print("  no checks run")
    else:
        name, b, num, exact, err = worst
        print(f"  {name} at b={b}:")
        print(f"    numeric = {num:+.17e}")
        print(f"    formula = {exact:+.17e}")
        print(f"    err     = {err:+.3e}")


if __name__ == "__main__":
    check_family(n=20000)
