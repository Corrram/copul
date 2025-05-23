import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import copul as cp                               # “checkerboard” helper

# ----------------------------------------------------------------------
# 1.  Helper:  b as a function of x  (3/8 < x < 1)
# ----------------------------------------------------------------------
def b_from_x(x: float) -> float:
    """Return the positive solution of 5(1-x)b² – 5b + 2 = 0."""
    if not (0.375 < x < 1):
        raise ValueError("x must lie in (3/8, 1)")
    disc = 40.0 * x - 15.0                       # discriminant  Δ = 40x-15
    return (5.0 + np.sqrt(disc)) / (10.0 * (1.0 - x))


# ----------------------------------------------------------------------
# 2.  Extremal copula C(u,v) for a given x
# ----------------------------------------------------------------------
def extremal_copula(u, v, x=0.5):
    """
    C(u,v) = ∫₀ᵘ h_v(t) dt  with
        h_v(t) = clamp(b(s_v - t), 0, 1)

    s_v  =
          √(2v / b)                       , 0 ≤ v ≤ 1/(2b)
          v / b + 1/2                     , 1/(2b) ≤ v ≤ 1 - 1/(2b)
          1 + 1/b − √(2(1-v) / b)         , 1 - 1/(2b) ≤ v ≤ 1 .
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    b = b_from_x(float(x))

    # --- s_v -----------------------------------------------------------
    alpha = 1.0 / (2.0 * b)
    s = np.empty_like(v)

    mask1 = v <= alpha
    s[mask1] = np.sqrt(2.0 * v[mask1] / b)

    mask2 = (v > alpha) & (v <= 1.0 - alpha)
    s[mask2] = v[mask2] / b + 0.5

    mask3 = v > 1.0 - alpha
    s[mask3] = 1.0 + 1.0 / b - np.sqrt(2.0 * (1.0 - v[mask3]) / b)

    # ------------------------------------------------------------------
    # Triangle (b s_v ≤ 1)  versus  plateau-plus-triangle (b s_v > 1)
    # ------------------------------------------------------------------
    bs = b * s
    tri_mask = bs <= 1.0 + 1e-12           # numerical tolerance
    plat_mask = ~tri_mask

    C = np.zeros_like(u)

    # ---------- TRIANGLE  ---------------------------------------------
    if tri_mask.any():
        s_tri = s[tri_mask]
        v_tri = v[tri_mask]
        u_tri = u[tri_mask]

        # sub-branch (i)  u ≤ s_v
        sub1 = u_tri <= s_tri
        C_tri = np.empty_like(u_tri)
        C_tri[sub1] = b * (s_tri[sub1] * u_tri[sub1] - 0.5 * u_tri[sub1] ** 2)

        # sub-branch (ii) u > s_v
        C_tri[~sub1] = v_tri[~sub1]        # mass saturates to v
        C[tri_mask] = C_tri

    # ---------- PLATEAU + TRIANGLE  -----------------------------------
    if plat_mask.any():
        s_plat = s[plat_mask]
        v_plat = v[plat_mask]
        u_plat = u[plat_mask]
        a_plat = s_plat - 1.0 / b          # plateau length

        C_plat = np.empty_like(u_plat)

        # (i)  u ≤ a_v : only plateau contributes
        sub1 = u_plat <= a_plat
        C_plat[sub1] = u_plat[sub1]

        # (ii) a_v < u ≤ s_v : plateau + partial triangle
        sub2 = (u_plat > a_plat) & (u_plat <= s_plat)
        if sub2.any():
            u2 = u_plat[sub2]
            a2 = a_plat[sub2]
            s2 = s_plat[sub2]
            C_plat[sub2] = (
                a2
                + b * (s2 * (u2 - a2) - 0.5 * (u2 ** 2 - a2 ** 2))
            )

        # (iii) u > s_v : full mass equals v
        C_plat[u_plat > s_plat] = v_plat[u_plat > s_plat]

        C[plat_mask] = C_plat

    return C


# ----------------------------------------------------------------------
# 3.  Demonstration  – surface plot for a given x
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # choose moment-constraint parameter
    x_val = 0.50                                 # any value in (0.375,1)
    b_val = b_from_x(x_val)
    print(f"b(x) = {b_val:.6f}  for  x = {x_val}")

    # grid for surface plot
    n = 200
    u = np.linspace(0.0, 1.0, n)
    v = np.linspace(0.0, 1.0, n)
    U, V = np.meshgrid(u, v)
    C_vals = extremal_copula(U, V, x=x_val)

    # 3-D surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U, V, C_vals,
                    cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('C(u,v)')
    ax.set_title(f'Extremal Copula Surface  (x = {x_val},  b = {b_val:.3f})')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 4.  k × k checkerboard approximation and summary indices
    # ------------------------------------------------------------------
    k = 10
    grid = np.linspace(0.0, 1.0, k + 1)
    M = np.zeros((k, k), dtype=float)

    for i in range(1, k + 1):
        for j in range(1, k + 1):
            u0, u1 = grid[i - 1], grid[i]
            v0, v1 = grid[j - 1], grid[j]
            mass = (
                extremal_copula(u1, v1, x_val)
                - extremal_copula(u0, v1, x_val)
                - extremal_copula(u1, v0, x_val)
                + extremal_copula(u0, v0, x_val)
            )
            M[i - 1, j - 1] = mass * k ** 2   # density on the block

    ccop = cp.BivCheckPi(M)
    xi  = ccop.xi()
    tau = ccop.tau()
    rho = ccop.rho()
    print(f"τ (Kendall) = {tau:.6f}   ξ = {xi:.6f}   ρ (Spearman) = {rho:.6f}")
