import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from tqdm import tqdm

# =============================================================================
#   SCRIPT PARAMETERS
# =============================================================================
# Set the number of points to evaluate for D in its valid range [0, 1]
# More points will result in a smoother curve but will take longer to compute.
NUM_POINTS = 30
# =============================================================================


def t_minus_scalar(v, D):
    """
    Calculates the lower breakpoint t_-(v) for a scalar v, based on the
    corrected derivation.
    """
    v_minus = D / 2.0
    v_plus = 1.0 - D / 2.0
    
    if 0 <= v <= v_minus:
        return 0.0
    elif v_minus < v < v_plus:
        return v - D / 2.0
    elif v_plus <= v <= 1:
        return 2.0 * v - 1.0
    return 0.0 # Fallback for out-of-range v

def t_plus_scalar(v, D):
    """
    Calculates the upper breakpoint t_+(v) for a scalar v, based on the
    corrected derivation.
    """
    v_minus = D / 2.0
    v_plus = 1.0 - D / 2.0

    if 0 <= v <= v_minus:
        return 2.0 * v
    elif v_minus < v < v_plus:
        return v + D / 2.0
    elif v_plus <= v <= 1:
        return 1.0
    return 0.0 # Fallback for out-of-range v

def copula_C_scalar(u, v, D):
    """
    Calculates the copula value C(u,v) for scalar inputs, based on the
    corrected formula (C_0), ensuring t_minus is non-negative.
    """
    # Original calculation for the breakpoints
    t_minus_raw = t_minus_scalar(v, D)
    t_plus = t_plus_scalar(v, D)
    
    # --- BUG FIX ---
    # The support of the optimizer h_v^0(t) is on [0,1]. If the theoretical
    # breakpoint t_-(v) is negative, the actual support interval [0, t_-(v)]
    # is empty. We must use max(0, t_-(v)) for the copula calculation.
    t_minus = max(0.0, t_minus_raw)
    
    if u <= t_minus:
        return u
    elif t_minus < u <= v:
        return t_minus
    elif v < u <= t_plus:
        return t_minus + u - v
    else:  # u > t_plus
        return v

def analytical_psi(D):
    """
    Calculates Spearman's footrule using the derived analytical formula for C_0:
    psi(C_0) = 1 - 3*D + 1.5*D^2
    """
    return 1.0 - 3.0 * D + 1.5 * D**2

def numerical_rho(D):
    """
    Numerically computes Spearman's rho by integrating the copula C_0(u,v).
    """
    # dblquad integrates f(x,y) dy dx. Here, x=u, y=v.
    # We need to integrate C(u,v) du dv. The argument order for our
    # copula function is already (u, v).
    integrand = lambda u, v: copula_C_scalar(u, v, D)
    
    # Perform the double integration of C(u,v) over the unit square.
    # The integration order is d-u first, then d-v.
    integral_val, _ = dblquad(integrand, 0, 1, lambda v: 0, lambda v: 1)
    
    # Compute rho using the integral value
    rho = 12 * integral_val - 3
    return rho


# --- Main script for evaluation and plotting ---
if __name__ == '__main__':
    # 1. Set up the range of D values to evaluate from 0 to 1
    D_values = np.linspace(0, 1, NUM_POINTS)
    
    psi_results = []
    rho_results = []

    print(f"Numerically evaluating Rho and Psi for {NUM_POINTS} values of D in [0, 1]...")
    # 2. Loop through each D, calculate Psi and Rho, and store the results
    for D in tqdm(D_values, desc="Calculating (psi, rho) pairs"):
        psi_val = analytical_psi(D)
        rho_val = numerical_rho(D)
        
        psi_results.append(psi_val)
        rho_results.append(rho_val)
    print("Calculation complete.")

    # 3. Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 9))

    # Plot the results
    ax.plot(psi_results, rho_results, '-o', color='royalblue', lw=2.5, markersize=5, label='Maximal $\\rho$ for given $\psi$')

    # 4. Annotate and customize the plot
    ax.set_title("Feasible Region for (Spearman's Footrule, Spearman's Rho)", fontsize=16)
    ax.set_xlabel(r"Spearman's Footrule $\psi(C)$", fontsize=14)
    ax.set_ylabel(r"Spearman's Rho $\rho(C)$", fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    
    # Annotate the start and end points
    # D=0 corresponds to the comonotonicity copula M(u,v)=min(u,v)
    ax.annotate(f'$D=0$\n(Comonotonicity)\n($\psi=1, \\rho=1$)', 
                xy=(1, 1), xytext=(0.4, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                ha='center', fontsize=12)

    # D=1 does not correspond to the countermonotonicity copula because rho=-0.5 not -1.
    ax.annotate(f'$D=1$\n($\psi=-0.5, \\rho=-0.5$)', 
                xy=(-0.5, -0.5), xytext=(-0.1, -0.7),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                ha='center', fontsize=12)

    # Set axis limits and grid
    ax.set_xlim(-0.6, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=12)
    
    # 5. Show the plot
    plt.show()