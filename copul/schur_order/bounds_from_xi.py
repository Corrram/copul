# xi_extrema.py
# ---------------------------------------------------------
# Extrema of (rho, tau, psi) as functions of xi.
# Implements exact formulas for rho and tau, and exact/rigorous
# bounds for psi (depending on the copula class).
#
# Usage:
#   from xi_extrema import bounds_from_xi
#   bounds_from_xi(0.3, measure="rho")
#
# ---------------------------------------------------------

from math import sqrt, acos, cos, isfinite, log
import numpy as np
from typing import Tuple, Literal, Optional


def _validate_xi(x: float) -> None:
    if not (isinstance(x, (int, float)) and isfinite(x)):
        raise ValueError("xi must be a finite real number.")
    if x < 0.0 or x > 1.0:
        raise ValueError("xi must be in [0, 1].")


# ---------------------- RHO (Spearman) ----------------------
def _b_x_rho(x: float) -> float:
    """
    Parameter b(x) from the (xi, rho) boundary.
    Piecewise definition matching Eq. (b_x) in the manuscript.
    """
    if x == 0.0:
        return 0.0  # limiting
    if x <= 0.3:  # 3/10
        val = -3.0 * sqrt(6.0 * x) / 5.0
        val = max(-1.0, min(1.0, val))  # clamp into [-1,1]
        return (sqrt(6.0 * x)) / (2.0 * cos((1.0 / 3.0) * acos(val)))
    else:
        return (5.0 + sqrt(5.0 * (6.0 * x - 1.0))) / (10.0 * (1.0 - x))


def rho_max_given_xi(x: float) -> float:
    """
    Maximal rho given xi, over all copulas (exact).
    """
    _validate_xi(x)
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    b = _b_x_rho(x)
    if x <= 0.3:
        return b - 3.0 * b * b / 10.0
    else:
        return 1.0 - 1.0 / (2.0 * b * b) + 1.0 / (5.0 * b**3)


def rho_bounds_from_xi(x: float) -> Tuple[float, float]:
    """
    Returns (rho_min, rho_max). Exact symmetry: rho_min = -rho_max.
    """
    m = rho_max_given_xi(x)
    return (-m, m)


# ---------------------- TAU (Kendall) ----------------------
def _b_x_tau(x: float) -> float:
    """Same parameter b(x) as for rho boundary."""
    return _b_x_rho(x)


def tau_max_given_xi(x: float) -> float:
    """
    Maximal Kendall's tau given xi, over all copulas (exact).
    """
    _validate_xi(x)
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    if x <= 0.3:
        b = _b_x_tau(x)
        return (4.0 * b - b * b) / 6.0
    else:
        return (7.0 + 15.0 * x) / 27.0 + (5.0 / 27.0) * sqrt((6.0 * x - 1.0) / 5.0)


def tau_bounds_from_xi(x: float) -> Tuple[float, float]:
    """
    Returns (tau_min, tau_max). Exact symmetry: tau_min = -tau_max.
    """
    m = tau_max_given_xi(x)
    return (-m, m)


# ---------------------- PSI (Spearman's footrule) ----------------------
def psi_max_given_xi(x: float, cls: Literal["all", "SI"] = "all") -> float:
    """
    Maximal psi given xi.
    - For all copulas: exact psi_max = sqrt(xi).
    - For SI copulas: same upper bound.
    """
    _validate_xi(x)
    return sqrt(x)


def psi_min_given_xi_si(x: float) -> float:
    """Minimal psi given xi for SI copulas (exact)."""
    _validate_xi(x)
    return x


# ---- Lower bound for psi_min over ALL copulas ----
def _mu_from_y(y: float) -> float:
    """Solve cubic for mu in [0,2] given y in [-0.5, 0]."""
    coefs = [1.0, -(4.0 + 2.0 * y), -(4.0 + 8.0 * y), -8.0 * y]
    roots = np.roots(coefs)
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-10]
    for r in real_roots:
        if 0.0 <= r <= 2.0:
            return r
    if real_roots:
        r = min(real_roots, key=lambda z: min(abs(z), abs(z - 2.0)))
        return min(2.0, max(0.0, r))
    return 1.0


def _xi_bound_from_y(y: float) -> float:
    """Compute xi_lower_bound as function of y in [-0.5, 0]."""
    mu = _mu_from_y(y)
    v1 = 2.0 / (2.0 + mu)
    xi = -4.0 * v1 * v1 + 20.0 * v1 - 17.0 + 2.0 / v1 - 1.0 / (v1 * v1) - 12.0 * log(v1)
    return xi


def psi_min_lower_bound_given_xi_all(x: float) -> float:
    """
    Rigorous LOWER BOUND for the minimal psi over ALL copulas at fixed xi=x.
    """
    _validate_xi(x)
    if x == 0.0:
        return 0.0
    if x >= 0.5:
        return -0.5

    lo, hi = -0.5, 0.0
    xi_lo = _xi_bound_from_y(lo)
    if x <= xi_lo:
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        xi_mid = _xi_bound_from_y(mid)
        if xi_mid <= x:
            hi = mid
        else:
            lo = mid
    return hi


def psi_bounds_from_xi(
    x: float,
    cls: Literal["all", "SI"] = "all",
    return_lower_bound: bool = True,
) -> Tuple[float, float]:
    """
    Returns (psi_min, psi_max) for a given xi.
    """
    _validate_xi(x)
    psi_max = psi_max_given_xi(x, cls="all")
    if cls == "SI":
        return (psi_min_given_xi_si(x), psi_max)
    else:
        if return_lower_bound:
            psi_min_lb = psi_min_lower_bound_given_xi_all(x)
            return (psi_min_lb, psi_max)
        else:
            return (None, psi_max)


# ---------------------- NU (Blest's measure) ----------------------
def _Xi_of_b(b: float) -> float:
    """
    ξ = Xi(b) from Theorem (piecewise in b).
    s = 1/sqrt(b), t = sqrt((b-1)/b), A = asinh(t/s) = acosh(sqrt(b)).
    """
    if b <= 0.0 or not isfinite(b):
        raise ValueError("b must be a positive finite number.")
    s = 1.0 / sqrt(b)
    if b <= 1.0:
        # Xi(b) = 8(7 s^2 - 3) / (105 s^6)
        return 8.0 * (7.0 * s * s - 3.0) / (105.0 * s**6)
    else:
        t = sqrt((b - 1.0) / b)
        # asinh(t/s) = asinh(sqrt(b-1))  (since t/s = sqrt(b-1))
        # numerically stable form via log for large b
        A = np.arcsinh(sqrt(max(b - 1.0, 0.0)))
        num = (
            -105.0 * s**8 * A
            + 183.0 * s**6 * t
            - 38.0 * s**4 * t
            - 88.0 * s**2 * t
            + 112.0 * s**2
            + 48.0 * t
            - 48.0
        )
        den = 210.0 * s**6
        return num / den


def _N_of_b(b: float) -> float:
    """
    ν = N(b) from Theorem (piecewise in b).
    """
    if b <= 0.0 or not isfinite(b):
        raise ValueError("b must be a positive finite number.")
    s = 1.0 / sqrt(b)
    if b <= 1.0:
        # N(b) = 4(28 s^2 - 9) / (105 s^4)
        return 4.0 * (28.0 * s * s - 9.0) / (105.0 * s**4)
    else:
        t = sqrt((b - 1.0) / b)
        A = np.arcsinh(sqrt(max(b - 1.0, 0.0)))
        num = (
            -105.0 * s**8 * A
            + 87.0 * s**6 * t
            + 250.0 * s**4 * t
            - 376.0 * s**2 * t
            + 448.0 * s**2
            + 144.0 * t
            - 144.0
        )
        den = 420.0 * s**4
        return num / den


def _b_from_xi(x: float, *, tol: float = 1e-12, max_iter: int = 200) -> float:
    """
    Invert Xi(b)=x for b>0 by monotone bisection.
    Xi(b) is strictly increasing from 0 (as b->0+) to 1 (as b->∞).
    """
    _validate_xi(x)
    if x == 0.0:
        return 0.0 + 1e-12  # “near-zero” b
    if x == 1.0:
        return 1e12  # “very large” b

    # Bracket: start near b=0 and expand hi until Xi(hi) >= x
    lo, hi = 1e-12, 1.0
    Xi_hi = _Xi_of_b(hi)
    while Xi_hi < x and hi < 1e16:
        hi *= 2.0
        Xi_hi = _Xi_of_b(hi)
    Xi_lo = _Xi_of_b(lo)

    # Safety: if numerical pathologies, fall back to a big hi
    if Xi_lo > x:
        return lo

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        Xi_mid = _Xi_of_b(mid)
        if abs(Xi_mid - x) <= tol:
            return mid
        if Xi_mid < x:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def nu_bounds_from_xi(x: float) -> Tuple[float, float]:
    """
    Exact bounds (nu_min, nu_max) given xi=x,
    via the parametric boundary y = ±N(b) with Xi(b)=x.
    """
    _validate_xi(x)
    if x == 0.0:
        return (0.0, 0.0)
    if x == 1.0:
        return (-1.0, 1.0)  # limiting behavior as b→∞

    b = _b_from_xi(x)
    N = _N_of_b(b)
    # Region is symmetric in ν
    return (-N, N)


# ---------------------- Friendly façade ----------------------
def bounds_from_xi(
    x: float,
    measure: Literal["rho", "tau", "psi", "nu"],
    return_lower_bound: bool = False,
    cls: Literal["all", "SI"] = "all",
) -> Tuple[Optional[float], float]:
    """
    Unified entry point.
    Returns (min_value, max_value) for the chosen measure given xi=x.
    """
    if measure == "rho":
        return rho_bounds_from_xi(x)
    elif measure == "tau":
        return tau_bounds_from_xi(x)
    elif measure == "psi":
        return psi_bounds_from_xi(x, cls=cls, return_lower_bound=return_lower_bound)
    elif measure == "nu":
        return nu_bounds_from_xi(x)
    else:
        raise ValueError("measure must be one of: 'rho', 'tau', 'psi', 'nu'")


if __name__ == "__main__":
    for xi in [0.0, 0.1, 0.3, 0.7, 1.0]:
        print(
            f"xi={xi:0.3f}  rho_bounds={rho_bounds_from_xi(xi)}  tau_bounds={tau_bounds_from_xi(xi)}"
        )
    for xi in [0.0, 0.25, 0.49, 0.5, 0.8]:
        print(
            f"xi={xi:0.3f}  psi_bounds_all={psi_bounds_from_xi(xi)}  psi_bounds_SI={psi_bounds_from_xi(xi, cls='SI')}"
        )
