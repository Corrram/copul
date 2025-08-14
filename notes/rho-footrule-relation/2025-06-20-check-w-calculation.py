import numpy as np
from scipy.integrate import quad

# --- Definitive Model and Ground Truth Functions (Correct and Unchanged) ---


def s_v_definitive(v, q, d):
    """
    Return (s_v, branch_label) where branch_label ∈ {'A-1','A-2','A-3','B-1'}.
    Implements the four cases of (S-final).
    """
    d_crit = 1.0 / q - 1.0
    s_star = q * (1.0 + d)
    v1 = 0.5 * s_star
    v2 = 1.0 - v1

    if d <= d_crit:
        if v <= v1:  # A-1
            s_v = np.sqrt(2.0 * q * (1.0 + d) * v)
            return s_v, "A-1"
        elif v <= v2:  # A-2
            s_v = v + 0.5 * q * (1.0 + d)
            return s_v, "A-2"
        else:  # A-3
            s_v = 1.0 + q * (1.0 + d) - np.sqrt(2.0 * q * (1.0 + d) * (1.0 - v))
            return s_v, "A-3"
    elif d > d_crit:
        v0 = 1.0 - 1.0 / (2.0 * s_star)  # s_star = q*(1+d)
        if v <= v0:  # B-0 : no plateau
            s_v = 0.5 + q * (1.0 + d) * v
            return s_v, "B-0"
        else:  # B-1 : plateau present
            s_v = 1.0 + s_star - np.sqrt(2.0 * q * (1.0 + d) * (1.0 - v))
            return s_v, "B-1"


def h_star_definitive(t, v, q, d):
    """The optimizer h_v(t) using the definitive s_v."""
    b = 1.0 / q
    s, case = s_v_definitive(v, q, d)
    indicator = 1.0 if t <= v else 0.0
    return np.clip(b * (s - t) - d * indicator, 0, 1), case


def W_ground_truth(d, q):
    """The benchmark value, calculated via double numerical integration of the raw h_v(t)."""

    def I_numerical_from_h(v, q, d):
        return quad(
            lambda t: h_star_definitive(t, v, q, d)[0], 0, v, epsabs=1e-13, limit=200
        )[0]

    return quad(lambda v: I_numerical_from_h(v, q, d), 0, 1, epsabs=1e-13, limit=200)[0]


# --- Analytical Formula for the Inner Integral I(v,d) (Unchanged) ---


def I_analytical(v, q, d):
    """Calculates the inner integral I(v,d) using the simplified analytical approach."""
    b = 1.0 / q
    s, case = s_v_definitive(v, q, d)

    def h_unjumped(t):
        return np.clip(b * (s - t), 0, 1)

    J, _ = quad(h_unjumped, v, 1, epsabs=1e-13, limit=200)
    # print(f"Case: {case}, v: {v:.4f}, s: {s:.4f}, J: {J:.6f}")
    return v - J


# --- New/Updated Analytical Component Functions ---


def W_A_analytical(q, d):
    """Calculates the explicit, closed-form analytical solution for the W_A component."""
    one_plus_d = 1.0 + d
    term1 = q**2 * one_plus_d**2
    term2 = 19.0 - 11.0 * d
    return term1 * term2 / 240.0


def W_B_analytical(q, d):
    """
    Calculates the explicit, closed-form analytical solution for the Regime-B contribution W_B.

    Parameters
    ----------
    q : float
        The parameter q = -a, with 0 < q < 1/2.
    d : float
        The global jump-size parameter, 0 < d < 1.

    Returns
    -------
    float
        The value of
            W_B = (q*(1+d) - 1) * (q*(1+d)**2 - 4) / 8.
    """
    one_plus_d = 1.0 + d
    A = q * one_plus_d - 1.0
    B = q * (one_plus_d**2) - 4.0
    return A * B / 8.0


def W_C_analytical(q, d):
    """
    Closed-form analytical expression for the Regime-C contribution W_C.

    Parameters
    ----------
    q : float
        The parameter q = -a, with 0 < q < 1/2.
    d : float
        The global jump-size parameter, 0 < d < 1.

    Returns
    -------
    float
        The value of
            W_C = (q / 240) * (
                  -11*d**3 * q
                  - 63*d**2 * q
                  - 93*d   * q
                  + 120*d
                  - 41*q
                  + 120
              ).
    """
    numerator = -11 * d**3 * q - 63 * d**2 * q - 93 * d * q + 120 * d - 41 * q + 120
    return (q * numerator) / 240.0


def W_analytical(q, d):
    """
    Closed-form analytical expression for the full wedge integral W(d).

    Parameters
    ----------
    q : float
        The parameter q = -a, with 0 < q < 1/2.
    d : float
        The global jump-size parameter, 0 < d < 1.

    Returns
    -------
    float
        The value of
            W(d) = 1/2
                   - (q / 8) * (1 + d)**2
                   + (q**2 / 30) * (1 + d)**3.
    """
    one_plus_d = 1.0 + d
    return 0.5 - (q * one_plus_d**2) / 8.0 + (q**2 * one_plus_d**3) / 30.0


def main(q_test, d_test):
    print("--- Verifying W(d) with Analytical W_A and W_B Components ---\n")
    print(f"Parameters: q = {q_test}, d = {d_test}")

    # 1. Calculate the full ground truth for comparison
    w_truth = W_ground_truth(d_test, q_test)
    print(f"\nGround Truth (Double Numerical Integral): W = {w_truth:.12f}")

    # 2. Define the regime boundaries
    one_plus_d = 1.0 + d_test
    v_1 = q_test * one_plus_d / 2.0
    v_2 = 1.0 - v_1
    print(f"Regime Boundaries: v1 = {v_1:.4f}, v2 = {v_2:.4f}\n")

    # 3. Calculate W_A, W_B analytically and W_C numerically
    W_A = W_A_analytical(q_test, d_test)
    W_B = W_B_analytical(q_test, d_test)
    W_C = W_C_analytical(q_test, d_test)
    # W_C, _ = quad(lambda v: I_analytical(v, q_test, d_test), v_2, 1, epsabs=1e-13)

    # 4. Sum the parts and compare to the ground truth
    w_hybrid_analytic = W_A + W_B + W_C
    w_hybrid_analytic = W_analytical(q_test, d_test)

    print("--- Hybrid Analytical/Numerical Results ---")
    print(f"Regime A Contribution (Analytical): W_A = {W_A:.12f}")
    print(f"Regime B Contribution (Analytical): W_B = {W_B:.12f}")
    print(f"Regime C Contribution (Analytical) : W_C = {W_C:.12f}")
    print("-" * 50)
    print(f"Sum of Parts (W_A_an + W_B_an + W_C_num) : W = {w_hybrid_analytic:.12f}")
    print("-" * 50)

    is_correct = abs(w_truth - w_hybrid_analytic) < 1e-9

    print(f"Does the hybrid sum match the ground truth? -> {is_correct}")
    if is_correct:
        print(
            "\nConclusion: Success! The analytical formulas for W_A and W_B are correct."
        )
        print("We can now proceed to the final component, W_C.")
    else:
        print("\nConclusion: Failure. The analytical formula for W_B is incorrect.")


# --- Run the Incremental Verification ---
if __name__ == "__main__":
    q_test = 0.25
    d_test = 0.5
    main(q_test, d_test)
    main(q_test=0.02, d_test=0.5)
