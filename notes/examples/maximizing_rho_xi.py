import copul as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extremal_copula(u, v):
    """
    Compute C(u,v) for the extremal copula given by:
    
    C(u,v) = ∫_0^u h_v^*(t) dt
    with closed form:
    
    For 0 <= v <= 1/2:
        if u <= sqrt(2v):  C = sqrt(2v)*u - u^2/2
        else:              C = v
    For 1/2 < v <= 1:
        a = 1 - sqrt(2*(1-v))
        if u <= a:         C = u
        else:              C = (1+a)*u - u^2/2 - a^2/2
    """
    u = np.asarray(u)
    v = np.asarray(v)
    C = np.zeros_like(u)

    # branch 1: v <= 1/2
    mask1 = (v <= 0.5)
    v1 = v[mask1]
    u1 = u[mask1]
    sqrt2v = np.sqrt(2*v1)
    # sub‐branch: u <= sqrt(2v)
    m1a = u1 <= sqrt2v
    C1 = np.empty_like(u1)
    C1[m1a] = sqrt2v[m1a]*u1[m1a] - 0.5*u1[m1a]**2
    C1[~m1a] = v1[~m1a]
    C[mask1] = C1

    # branch 2: v > 1/2
    mask2 = ~mask1
    v2 = v[mask2]
    u2 = u[mask2]
    a = 1 - np.sqrt(2*(1-v2))
    # sub‐branch: u <= a
    m2a = u2 <= a
    C2 = np.empty_like(u2)
    C2[m2a] = u2[m2a]
    C2[~m2a] = (1 + a[~m2a])*u2[~m2a] - 0.5*u2[~m2a]**2 - 0.5*a[~m2a]**2
    C[mask2] = C2

    return C

# create grid
n = 200
u = np.linspace(0, 1, n)
v = np.linspace(0, 1, n)
U, V = np.meshgrid(u, v)

# evaluate copula
C_vals = extremal_copula(U, V)

# plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(U, V, C_vals, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('C(u,v)')
ax.set_title('Extremal Copula Surface')
plt.tight_layout()
plt.show()

# Parameters
k = 10
grid = np.linspace(0, 1, k+1)

# Initialize matrix
M = np.zeros((k, k), dtype=float)

# Compute checkerboard densities
for i in range(1, k+1):
    for j in range(1, k+1):
        u0, u1 = grid[i-1], grid[i]
        v0, v1 = grid[j-1], grid[j]
        # block mass via inclusion–exclusion
        mass = (extremal_copula(u1, v1)
                - extremal_copula(u0, v1)
                - extremal_copula(u1, v0)
                + extremal_copula(u0, v0))
        # scale to density on unit square
        M[i-1, j-1] = mass * k**2

# Print result
ccop = cp.BivCheckPi(M)
xi = ccop.xi()
tau = ccop.tau()
rho = ccop.rho()
print(f"tau = {tau}, xi = {xi}, rho = {rho}")