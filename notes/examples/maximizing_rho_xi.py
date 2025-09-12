import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import brentq  # For robust root finding

# Assuming 'copul' is a custom module available in the environment.
# If not, the part using 'copul' (section 4 in main) might not run
# without further context or access to this module.
try:
    import copul as cp
except ImportError:
    print("Warning: 'copul' module not found. Section 4 of the demo might not work.")
    cp = None


# ----------------------------------------------------------------------
# 1.  Helper:  b as a function of x (Updated Logic)
# ----------------------------------------------------------------------
def b_from_x(x: float) -> float:
    """
    Return b for a given x, based on the corrected proof logic.
    - If x is in (0, 3/10], b is the root in (0,1] of 2b³ - 5b² + 10x = 0.
    - If x is in (3/10, 1), b is (5 + sqrt(30x-5)) / (10(1-x)).
    x must lie in (0, 1).
    """
    if not (0 < x < 1):
        raise ValueError(f"x must lie in (0, 1), but got x = {x}")

    if np.isclose(x, 0.3):  # x = 3/10 numerically
        return 1.0

    if 0 < x < 0.3:  # (0, 3/10)
        # Solve 2b³ - 5b² + 10x = 0 for b in (0, 1).
        # Let p(b) = 2b³ - 5b² + 10x.
        # p(0) = 10x > 0.
        # p(1) = 2 - 5 + 10x = 10x - 3. Since x < 0.3, 10x < 3, so 10x - 3 < 0.
        # A unique root exists in (0, 1).
        try:
            # Using brentq for robust root finding in the interval (eps, 1.0)
            # A very small positive lower bound like 1e-10 or 1e-12 avoids issues if root is exactly 0 for x->0
            # but since x > 0, b should be > 0.
            b_val = brentq(
                lambda b_var: 2 * b_var**3 - 5 * b_var**2 + 10 * x, 1e-10, 1.0
            )
            return b_val
        except ValueError as e:
            # This should ideally not happen given the sign change argument.
            raise RuntimeError(f"Root finding failed for x={x} in (0, 3/10) range: {e}")

    elif 0.3 < x < 1:  # (3/10, 1)
        # Use the formula b = (5 + sqrt(30x - 5)) / (10(1-x))
        # This corresponds to x = 1 - 1/b + 3/(10b^2), valid for b > 1.
        # For x > 0.3, 30x - 5 > 30(0.3) - 5 = 9 - 5 = 4 > 0, so discriminant_term is positive.
        discriminant_term = 30.0 * x - 5.0

        numerator = 5.0 + np.sqrt(discriminant_term)
        denominator = 10.0 * (1.0 - x)

        return numerator / denominator
    else:  # Should not be reached due to initial checks if logic is sound.
        raise ValueError(f"Internal error: Unhandled x = {x} in b_from_x function.")


# ----------------------------------------------------------------------
# 2.  Extremal copula C(u,v) for a given x
# ----------------------------------------------------------------------
def extremal_copula(u, v, x=0.5):
    """
    C(u,v) = ∫₀ᵘ h_v(t) dt  with
        h_v(t) = clamp(b(s_v - t), 0, 1)

    s_v is determined by a 4-part formula depending on b and v (from Proof Part 2).
    b is determined from x by the updated b_from_x function.
    """
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)

    b = b_from_x(float(x))  # Call the updated function

    # --- s_v calculation based on Proof Part 2 (LaTeX Step (2)) ---
    s = np.empty_like(v_arr, dtype=float)

    # Conditions for b >= 1.0
    if b >= 1.0:  # This includes b=1.0 where x=0.3
        one_over_2b = 1.0 / (2.0 * b)

        # Mask for v values
        mask_v_le_one_over_2b = (
            v_arr <= one_over_2b + 1e-12
        )  # Add tolerance for floating point
        mask_v_gt_one_minus_one_over_2b = v_arr > (1.0 - one_over_2b) - 1e-12

        # Case 1: (b >= 1 and 0 <= v <= 1/(2b)) -> s_v = sqrt(2v/b)
        current_mask = mask_v_le_one_over_2b
        s[current_mask] = np.sqrt(
            np.maximum(0, 2.0 * v_arr[current_mask] / b)
        )  # ensure sqrt non-negative

        # Case 4: (b >= 1 and 1-1/(2b) < v <= 1) -> s_v = 1 + 1/b - sqrt(2(1-v)/b)
        current_mask = mask_v_gt_one_minus_one_over_2b
        term_under_sqrt_c4 = np.maximum(0, 2.0 * (1.0 - v_arr[current_mask]) / b)
        s[current_mask] = 1.0 + 1.0 / b - np.sqrt(term_under_sqrt_c4)

        # Case 2: (b > 1 and 1/(2b) < v <= 1-1/(2b)) -> s_v = v + 1/(2b)
        # This condition only applies if b is strictly > 1 because if b=1, one_over_2b = 0.5,
        # and the previous two masks cover v<=0.5 and v>0.5.
        if b > 1.0 + 1e-12:  # Strictly b > 1
            # Intermediate v values not covered by Case 1 or Case 4
            mask_c2 = np.logical_not(
                np.logical_or(mask_v_le_one_over_2b, mask_v_gt_one_minus_one_over_2b)
            )
            s[mask_c2] = v_arr[mask_c2] + one_over_2b

    # Conditions for 0 < b < 1.0
    else:
        b_over_2 = b / 2.0

        mask_v_le_b_over_2 = v_arr <= b_over_2 + 1e-12
        mask_v_gt_one_minus_b_over_2 = v_arr > (1.0 - b_over_2) - 1e-12

        # Case 1 (variant for 0 < b < 1): (0 < b < 1 and 0 <= v <= b/2) -> s_v = sqrt(2v/b)
        current_mask = mask_v_le_b_over_2
        s[current_mask] = np.sqrt(np.maximum(0, 2.0 * v_arr[current_mask] / b))

        # Case 4 (variant for 0 < b < 1): (0 < b < 1 and 1-b/2 < v <= 1) -> s_v = 1 + 1/b - sqrt(2(1-v)/b)
        current_mask = mask_v_gt_one_minus_b_over_2
        term_under_sqrt_c4_b_lt_1 = np.maximum(0, 2.0 * (1.0 - v_arr[current_mask]) / b)
        s[current_mask] = 1.0 + 1.0 / b - np.sqrt(term_under_sqrt_c4_b_lt_1)

        # Case 3 (for 0 < b < 1): (0 < b < 1 and b/2 < v <= 1-b/2) -> s_v = v/b + 1/2
        # Intermediate v values not covered by the two masks above for 0 < b < 1
        mask_c3_b_lt_1 = np.logical_not(
            np.logical_or(mask_v_le_b_over_2, mask_v_gt_one_minus_b_over_2)
        )
        s[mask_c3_b_lt_1] = v_arr[mask_c3_b_lt_1] / b + 0.5

    # Clamp s_v values if necessary, though formulas should yield valid s_v for v in [0,1]
    # For v=0, s_v should be 0. For v=1, s_v can be > 1 (e.g., 1+1/b - 0).
    # The clamp in h_v(t) handles t effectively.

    # ------------------------------------------------------------------
    # Triangle (b s_v <= 1)  versus  plateau-plus-triangle (b s_v > 1)
    # Integration logic for C(u,v) = integral_0^u clamp(b(s_v-t),0,1) dt
    # This part remains structurally the same.
    # ------------------------------------------------------------------
    bs = b * s
    tri_mask = bs <= 1.0 + 1e-12
    plat_mask = ~tri_mask

    C = np.zeros_like(u_arr, dtype=float)

    if np.any(tri_mask):
        s_tri = s[tri_mask]
        # v_arr_tri = v_arr[tri_mask] # This is the v that generated s_tri
        u_tri = u_arr[tri_mask]
        C_tri_vals = np.empty_like(u_tri, dtype=float)

        # Sub-branch (i) u <= s_v : integral b(s_v-t) from 0 to u
        # h_v(t) = b(s_v-t) for t in [0, u], and u <= s_v.
        # So h_v(t) >= 0. It's also <= 1 because bs_v <= 1 means max height is at most 1.
        sub1 = u_tri <= s_tri + 1e-12  # u is less than or equal to where line hits zero
        if np.any(sub1):
            C_tri_vals[sub1] = b * (s_tri[sub1] * u_tri[sub1] - 0.5 * u_tri[sub1] ** 2)

        # Sub-branch (ii) u > s_v : integral b(s_v-t) from 0 to s_v. Total mass is v.
        # This happens if h_v(t) becomes 0 before t=u.
        sub2 = u_tri > s_tri + 1e-12
        if np.any(sub2):
            # The integral up to s_v (where h_v(t) becomes 0) is v.
            # C_tri_vals[sub2] = v_arr[tri_mask][sub2] # This was in original snippet logic
            # The integral from 0 to s_v of b(s_v-t) is b * s_v^2 / 2.
            # This should be equal to v_arr[tri_mask][sub2] due to how s_v was defined for this case.
            C_tri_vals[sub2] = b * s_tri[sub2] ** 2 / 2.0

        # Ensure C_tri_vals does not exceed v_arr[tri_mask] or u_arr[tri_mask] (copula bounds)
        # Also, C(u,v) must be <= v because h_v(t) <= 1. C(u,v) = int_0^u h_v(t) dt. If u=1, C(1,v)=v.
        # Max value of integral_0^u b(s_v-t)dt is v (when u >= s_v).
        # Clamp to v value, effectively using v_arr[tri_mask] for sub2 as intended by original logic.
        if np.any(sub2):
            C_tri_vals[sub2] = v_arr[tri_mask][sub2]

        C[tri_mask] = np.maximum(0, C_tri_vals)  # Ensure non-negative

    if np.any(plat_mask):
        s_plat = s[plat_mask]
        v_plat_for_C = v_arr[plat_mask]  # v that generated s_plat
        u_plat = u_arr[plat_mask]

        a_plat = s_plat - 1.0 / b
        C_plat_vals = np.empty_like(u_plat, dtype=float)

        sub1_plat = u_plat <= a_plat + 1e-12
        if np.any(sub1_plat):
            C_plat_vals[sub1_plat] = u_plat[sub1_plat]  # Integral of 1 is u

        # For u_plat > a_plat, ensure a_plat is not negative (can happen if s_plat < 1/b due to precision with v near 0)
        # However, plat_mask (bs > 1) implies s > 1/b, so a_plat should be > 0.
        a_plat_safe = np.maximum(0, a_plat)  # Ensure plateau length isn't negative

        # Consider u_plat relative to 1.0 as well, as integral is over [0, u_plat]
        # h_v(t) is 1 on [0, a_plat_safe], then b(s_plat-t) on (a_plat_safe, min(1, s_plat)]

        # If a_plat < u_plat <= s_plat (and u_plat <= 1)
        mask_u_gt_a_le_s = np.logical_and(
            u_plat > a_plat_safe - 1e-12, u_plat <= s_plat + 1e-12
        )
        if np.any(mask_u_gt_a_le_s):
            u2 = u_plat[mask_u_gt_a_le_s]
            a2 = a_plat_safe[mask_u_gt_a_le_s]
            s2 = s_plat[mask_u_gt_a_le_s]
            C_plat_vals[mask_u_gt_a_le_s] = (
                a2  # Integral of 1 over [0, a2]
                + b
                * (
                    s2 * (u2 - a2) - 0.5 * (u2**2 - a2**2)
                )  # Integral of ramp b(s2-t) from a2 to u2
            )

        # If u_plat > s_plat (meaning h_v(t) potentially becomes 0 before u_plat, if s_plat <= 1)
        # Or if ramp continues to t=1 because s_plat > 1
        # The total integral up to t=1 should be v. For u_plat > s_plat (and s_plat <=1), result is v.
        # For s_plat > 1, the integral up to t=1 defines v.
        # If u_plat is simply > s_plat, it means we've integrated past where the density might have changed.
        # The total mass up to min(1, s_plat) (if density is zero beyond s_plat) is v.
        mask_u_gt_s = u_plat > s_plat + 1e-12
        if np.any(mask_u_gt_s):
            C_plat_vals[mask_u_gt_s] = v_plat_for_C[mask_u_gt_s]

        C[plat_mask] = np.maximum(0, C_plat_vals)  # Ensure non-negative

    # Final check on copula properties C(u,v) <= u and C(u,v) <= v
    C = np.minimum(C, u_arr)
    C = np.minimum(
        C, np.broadcast_to(v_arr, C.shape)
    )  # v_arr might need broadcasting if u/v shapes differ

    if np.isscalar(u) and np.isscalar(v):
        return C.item()
    return C


# ----------------------------------------------------------------------
# 3.  Demonstration  – surface plot for a given x
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # choose moment-constraint parameter
    # Test with a value like x = 0.1 (should use cubic solver for b)
    # Test with x = 1/6 (approx 0.1667, should use cubic, b will NOT be 0.6)
    # Test with x = 0.3 (b=1)
    # Test with x = 0.5 (should use quadratic solver for b, b > 1)

    x_values_to_test = [0.1, 1 / 6, 0.25, 0.3, 0.5, 0.9]

    for x_val in x_values_to_test:
        print("-" * 50)
        try:
            b_val = b_from_x(x_val)
            print(f"For x = {x_val:.6f}, b(x) = {b_val:.6f}")

            n = 30  # Reduced n for quicker testing, can be increased for smoother plots
            u_plot = np.linspace(0.0, 1.0, n)
            v_plot = np.linspace(0.0, 1.0, n)
            U, V = np.meshgrid(u_plot, v_plot)

            C_vals = extremal_copula(U, V, x=x_val)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(U, V, C_vals, cmap="viridis", edgecolor="none", alpha=0.9)
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.set_zlabel("C(u,v)")
            ax.set_title(f"Extremal Copula Surface (x = {x_val:.4f}, b = {b_val:.4f})")
            ax.view_init(elev=30, azim=-120)
            plt.tight_layout()
            # plt.show()

            # Optional: Run checkerboard approximation if module is available
            k = 30
            grid = np.linspace(0.0, 1.0, k + 1)
            M_density = np.zeros((k, k), dtype=float)

            for i in range(k):
                for j in range(k):
                    u0, u1 = grid[i], grid[i + 1]
                    v0, v1 = grid[j], grid[j + 1]

                    mass = (
                        extremal_copula(u1, v1, x_val)
                        - extremal_copula(u0, v1, x_val)
                        - extremal_copula(u1, v0, x_val)
                        + extremal_copula(u0, v0, x_val)
                    )
                    M_density[i, j] = mass * k**2

            # Ensure M_density sums to 1 (or close to it)
            # print(f"Sum of checkerboard densities: {np.sum(M_density / k**2):.6f}")

            ccop = cp.BivCheckPi(M_density)
            xi = ccop.chatterjees_xi()
            tau = ccop.kendalls_tau()
            rho = ccop.spearmans_rho()
            print(f"Approximated from {k}x{k} checkerboard density for x={x_val:.4f}:")
            print(
                f"τ (Kendall) = {tau:.6f}   ξ (xi) = {xi:.6f}   ρ (Spearman) = {rho:.6f}"
            )

        except ValueError as e:
            print(f"Error for x = {x_val:.6f}: {e}")
        except RuntimeError as e:
            print(f"Runtime Error for x = {x_val:.6f}: {e}")

    if not cp:
        print(
            "\nSkipped Section 4 (checkerboard approximation examples) as 'copul' module is not available."
        )
