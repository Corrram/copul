import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import dblquad

# Attempt to import the 'copul' module
try:
    import copul as cp
except ImportError:
    print("Warning: 'copul' module not found. Spearman's rho calculation will use numerical fallback (slower).")
    cp = None

# ----------------------------------------------------------------------
# Copied and refined functions from the previous script
# ----------------------------------------------------------------------
def b_from_x(x_param: float) -> float:
    """
    Return b for a given x_param (parameter 'x' from the proof), based on the corrected proof logic.
    - If x_param is in (0, 3/10], b is the root in (0,1] of 2b³ - 5b² + 10*x_param = 0.
    - If x_param is in (3/10, 1), b is (5 + sqrt(30*x_param-5)) / (10*(1-x_param)).
    x_param must lie in (0, 1).
    """
    if not (0 < x_param < 1):
        raise ValueError(f"x_param must lie in (0, 1), but got x_param = {x_param}")

    if np.isclose(x_param, 0.3): # x_param = 3/10 numerically
        return 1.0

    if 0 < x_param < 0.3:  # (0, 3/10)
        try:
            # p(b) = 2b³ - 5b² + 10*x_param. p(0) > 0, p(1) < 0 for this x_param range.
            b_val = brentq(lambda b_var: 2 * b_var**3 - 5 * b_var**2 + 10 * x_param, 1e-10, 1.0)
            return b_val
        except ValueError as e:
            raise RuntimeError(f"Root finding failed for x_param={x_param} in (0, 3/10) range: {e}")

    elif 0.3 < x_param < 1:  # (3/10, 1)
        discriminant_term = 30.0 * x_param - 5.0
        if discriminant_term < 0: # Should not happen for x_param > 0.3
             discriminant_term = 0 # Safety for precision issues near 1/6 if this branch was misused
        
        numerator = 5.0 + np.sqrt(discriminant_term)
        denominator = 10.0 * (1.0 - x_param)
        if np.isclose(denominator, 0): # Should not happen for x_param < 1
            raise ValueError(f"Denominator is zero for x_param={x_param}, cannot calculate b.")
        return numerator / denominator
    else: 
        # This case should ideally not be reached if x_param is strictly between 0 and 1,
        # and not 0.3. Handles any missed floating point edge cases.
        raise ValueError(f"Internal error: Unhandled x_param = {x_param} in b_from_x function.")

def extremal_copula(u, v, x_param_for_copula: float):
    """
    Calculates C(u,v) for the extremal copula defined by x_param_for_copula.
    """
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float) 
    
    # Obtain b using the corrected b_from_x logic
    b = b_from_x(float(x_param_for_copula))

    s = np.empty_like(v_arr, dtype=float)
    # --- s_v calculation based on Proof Part 2 (LaTeX Step (2)) ---
    # This logic matches the four-part formula for s_v based on b and v.
    # Floating point comparisons use a small epsilon.
    eps = 1e-12 

    if b >= 1.0 - eps: # b >= 1
        one_over_2b = 1.0 / (2.0 * b) if b > eps else np.inf
        
        mask1 = (v_arr <= one_over_2b + eps)
        s[mask1] = np.sqrt(np.maximum(0, 2.0 * v_arr[mask1] / b))
        
        mask4 = (v_arr > (1.0 - one_over_2b) - eps)
        term_sqrt_m4 = np.maximum(0, 2.0 * (1.0 - v_arr[mask4]) / b)
        s[mask4] = 1.0 + (1.0/b if b > eps else np.inf) - np.sqrt(term_sqrt_m4)
        
        if b > 1.0 + eps: # Strictly b > 1 for Case 2 of s_v
            mask2 = np.logical_not(np.logical_or(mask1, mask4))
            s[mask2] = v_arr[mask2] + one_over_2b
        elif not (b > 1.0 + eps) and np.any(np.logical_not(np.logical_or(mask1, mask4))): 
            # If b is very close to 1, mask1 and mask4 should cover all, if not, assign to avoid uninit s
            mask_middle_b_eq_1 = np.logical_not(np.logical_or(mask1, mask4))
            s[mask_middle_b_eq_1] = v_arr[mask_middle_b_eq_1] + 0.5 # if b=1, one_over_2b = 0.5
            
    else: # 0 < b < 1
        b_over_2 = b / 2.0
        
        mask1_lt1 = (v_arr <= b_over_2 + eps)
        s[mask1_lt1] = np.sqrt(np.maximum(0, 2.0 * v_arr[mask1_lt1] / b))
        
        mask4_lt1 = (v_arr > (1.0 - b_over_2) - eps)
        term_sqrt_m4_lt1 = np.maximum(0, 2.0 * (1.0 - v_arr[mask4_lt1]) / b)
        s[mask4_lt1] = 1.0 + (1.0/b if b > eps else np.inf) - np.sqrt(term_sqrt_m4_lt1)
        
        mask3_lt1 = np.logical_not(np.logical_or(mask1_lt1, mask4_lt1))
        s[mask3_lt1] = v_arr[mask3_lt1] / b + 0.5

    # Integration logic for C(u,v)
    bs = b * s
    tri_mask = bs <= 1.0 + eps 
    plat_mask = ~tri_mask

    C = np.zeros_like(u_arr, dtype=float)

    # Triangle case
    if np.any(tri_mask):
        s_m = s[tri_mask]; u_m = u_arr[tri_mask]; v_m = v_arr[tri_mask]
        C_vals = np.empty_like(u_m)
        
        sub1 = u_m <= s_m + eps
        C_vals[sub1] = b * (s_m[sub1] * u_m[sub1] - 0.5 * u_m[sub1]**2)
        
        sub2 = u_m > s_m + eps
        C_vals[sub2] = v_m[sub2] # Integral saturates to v
        C[tri_mask] = C_vals

    # Plateau + Triangle case
    if np.any(plat_mask):
        s_m = s[plat_mask]; u_m = u_arr[plat_mask]; v_m = v_arr[plat_mask]
        C_vals = np.empty_like(u_m)
        a_m = np.maximum(0, s_m - (1.0/b if b > eps else np.inf)) # Plateau length, ensure a_m >= 0

        sub1 = u_m <= a_m + eps
        C_vals[sub1] = u_m[sub1]

        sub2_cond = np.logical_and(u_m > a_m - eps, u_m <= s_m + eps)
        u2 = u_m[sub2_cond]; a2 = a_m[sub2_cond]; s2 = s_m[sub2_cond]
        C_vals[sub2_cond] = a2 + b * (s2 * (u2 - a2) - 0.5 * (u2**2 - a2**2))
        
        sub3 = u_m > s_m + eps
        C_vals[sub3] = v_m[sub3] # Integral saturates to v
        C[plat_mask] = C_vals
        
    # Ensure copula properties
    C = np.maximum(0, C)
    C = np.minimum(C, u_arr)
    C = np.minimum(C, np.broadcast_to(v_arr, C.shape)) # Ensure C(u,v) <= v

    return C.item() if np.isscalar(u) and np.isscalar(v) else C

def calculate_spearmans_rho_numerical(xi_param_for_rho, k_grid_integral=50):
    """
    Numerically estimates Spearman's rho for C_xi_param using formula:
    rho_S = 12 * E[C(U,V)] - 3, where U,V are independent Unif(0,1).
    E[C(U,V)] is approximated by dblquad.
    """
    copula_to_integrate = lambda v, u: extremal_copula(u, v, x_param_for_copula=xi_param_for_rho)
    
    # dblquad can be sensitive and slow; epsabs and epsrel control precision.
    # For plotting, high precision might not be strictly needed.
    integral_C_uv, _ = dblquad(copula_to_integrate, 0, 1, 
                               lambda u_outer: 0, lambda u_outer: 1, 
                               epsabs=1e-3, epsrel=1e-3) # Looser tolerance for speed
    rho_s = 12 * integral_C_uv - 3
    return rho_s

def get_rho_for_xi(xi_param, k_checkerboard=40, use_copul_module_if_available=True):
    """
    Calculates Spearman's Rho for a given xi_param.
    Tries to use 'copul' module first if available and requested.
    Falls back to numerical integration of C(u,v) otherwise.
    """
    if cp and use_copul_module_if_available:
        try:
            grid = np.linspace(0.0, 1.0, k_checkerboard + 1)
            M_density_mass = np.zeros((k_checkerboard, k_checkerboard), dtype=float) # Stores mass in cell

            for i in range(k_checkerboard):
                for j in range(k_checkerboard):
                    u0, u1 = grid[i], grid[i+1]
                    v0, v1 = grid[j], grid[j+1]
                    
                    mass = (
                        extremal_copula(u1, v1, xi_param)
                        - extremal_copula(u0, v1, xi_param)
                        - extremal_copula(u1, v0, xi_param)
                        + extremal_copula(u0, v0, xi_param)
                    )
                    # M_density_mass[i, j] = mass  # This is mass, BivCheckPi usually wants density
                    M_density_mass[i, j] = mass * (k_checkerboard**2) # Convert mass to density

            ccop = cp.BivCheckPi(M_density_mass) # Pass density matrix
            return ccop.rho()
        except Exception as e:
            print(f"Error using copul module for xi={xi_param:.4f}: {e}. Will attempt fallback.")
            # Fall through to numerical integration

    # Fallback or if copul module is not chosen/available
    # Announce fallback only if it wasn't due to 'cp' being None initially
    if use_copul_module_if_available and cp: # Means copul was available but failed
      print(f"Fallback to numerical integration for Spearman's rho for xi={xi_param:.4f} (slower).")
    elif not cp and use_copul_module_if_available: # cp was None from the start
      # This warning is now printed once at the start of the script.
      pass 
      
    return calculate_spearmans_rho_numerical(xi_param_for_rho)

# ----------------------------------------------------------------------
# Main script for plotting the xi-rho region
# ----------------------------------------------------------------------
def plot_xi_rho_region():
    num_xi_points = 30  # Number of xi values to sample (can increase for smoother plot)
                        # Lowered for faster execution during testing.
    xi_values = np.linspace(0.01, 0.99, num_xi_points) # x_param range (0,1)
    rho_s_values = np.zeros_like(xi_values)
    
    print(f"Calculating Spearman's rho for {num_xi_points} xi values...")
    
    use_copul = cp is not None # Decide once whether to attempt using copul
    if not use_copul:
        print("Numerical integration (dblquad) will be used for Rho_S as 'copul' module is unavailable.")

    for i, current_xi in enumerate(xi_values):
        print(f"Processing xi = {current_xi:.4f} ({i+1}/{num_xi_points})... ", end="")
        try:
            rho_s_values[i] = get_rho_for_xi(current_xi, k_checkerboard=15, use_copul_module_if_available=use_copul)
            print(f"rho_S = {rho_s_values[i]:.4f}")
        except Exception as e:
            rho_s_values[i] = np.nan # Mark as NaN if calculation fails
            print(f"Failed: {e}")

    # Filter out NaN values for plotting
    valid_indices = ~np.isnan(rho_s_values)
    xi_plot = xi_values[valid_indices]
    rho_s_plot = rho_s_values[valid_indices]

    # add (0,0) point for completeness
    if len(xi_plot) > 0 and xi_plot[0] > 0:
        xi_plot = np.insert(xi_plot, 0, 0.0)
        rho_s_plot = np.insert(rho_s_plot, 0, 0.0)

    if len(xi_plot) == 0:
        print("No Rho_S values could be calculated. Cannot generate plot.")
        return

    plt.figure(figsize=(10, 7))
    plt.plot(xi_plot, rho_s_plot, label=r"$\rho_S(\xi)$", color="blue", marker='o', markersize=4, linestyle='-')
    plt.plot(xi_plot, -rho_s_plot, label=r"$-\rho_S(\xi)$ (Implied by symmetry)", color="blue", marker='o', markersize=4, linestyle='--') # Assuming symmetry
    plt.fill_between(xi_plot, rho_s_plot, -rho_s_plot, color="skyblue", alpha=0.3, label=r"Attainable $(\xi, \rho_S)$ Region")
    
    # Plot known boundaries if available (e.g., Fréchet-Hoeffding bounds for rho_S)
    # Rho_S for M(u,v) is 1, for W(u,v) is -1. Pi(u,v) is 0.
    # Our xi parameter is not directly these bounds.
    
    plt.xlabel(r"$\xi$ (Parameter $x$ from constraint defining the copula family)")
    plt.ylabel(r"Spearman's $\rho_S$")
    plt.title(r"Attainable Region for $(\xi, \rho_S)$")
    plt.legend()
    plt.grid(True)
    plt.ylim([-1.05, 1.05]) # Spearman's rho is between -1 and 1
    plt.xlim([0, 1])
    plt.axhline(0, color='black', lw=0.5)
    plt.show()

if __name__ == "__main__":
    # To make this script runnable, ensure that extremal_copula and b_from_x are defined
    # (they are defined above in this combined snippet)
    plot_xi_rho_region()