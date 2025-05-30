import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# 1) Parameter
x0     = 0.1
v_line = 2/3

# 2) explicit b(x)
def b_of_x(x):
    if x <= 0.3:
        return (2/np.sqrt(6*x)) * np.cos((1/3)*np.arccos(-3*np.sqrt(6*x)/5))
    else:
        return (5 - np.sqrt(5*(6*x - 1))) / 3

b0 = b_of_x(x0)

# 3) piecewise s(v) and derivative
def s_small(v):
    if   v <= b0/2:
        return np.sqrt(2*v*b0)
    elif v <= 1 - b0/2:
        return v + b0/2
    else:
        return 1 + b0 - np.sqrt(2*b0*(1-v))

def ds_small(v):
    if   v <= b0/2:
        return np.sqrt(b0/(2*v))
    elif v <= 1 - b0/2:
        return 1.0
    else:
        return b0/np.sqrt(2*b0*(1-v))

def s_large(v):
    if   v <= 1/(2*b0):
        return np.sqrt(2*v*b0)
    elif v <= 1 - 1/(2*b0):
        return v*b0 + 1/2
    else:
        return 1 + b0 - np.sqrt(2*b0*(1-v))

def ds_large(v):
    if   v <= 1/(2*b0):
        return np.sqrt(b0/(2*v))
    elif v <= 1 - 1/(2*b0):
        return b0
    else:
        return b0/np.sqrt(2*b0*(1-v))

# pick regime
if b0 <= 1:
    s_func, ds_func = s_small, ds_small
else:
    s_func, ds_func = s_large, ds_large

# 4) support endpoints
def a_of_v(v):
    return s_func(v) - b0

# 5) build grid
n = 400
u = np.linspace(0,1,n)
v = np.linspace(0,1,n)
U, V = np.meshgrid(u, v)

# 6) compute density
C = np.zeros_like(U)
for i, vi in enumerate(v):
    sv  = s_func(vi)
    av  = a_of_v(vi)
    dsv = ds_func(vi)
    mask = (U[i,:] >= av) & (U[i,:] <= sv)
    C[i,mask] = dsv / b0

#largest finite C value
C_max = np.max(C[np.isfinite(C)])

# 7) compute support at v_line
a_v = a_of_v(v_line)
s_v = s_func(v_line)


# 8) plot
fig, ax = plt.subplots(figsize=(6,5))
norm = PowerNorm(gamma=0.5, vmin=0, vmax=C_max)  # try gamma=0.3, 0.5, 0.7, etc.
cs = ax.contourf(U, V, C, levels=300, cmap='viridis', norm=norm)
plt.colorbar(cs, label='copula density $c(u,v)$')

# 9) plot boundary curves a(v), s(v) across all v
v_curve = np.linspace(0,1,400)
a_curve = np.array([a_of_v(vi) for vi in v_curve])
s_curve = np.array([s_func(vi) for vi in v_curve])
ax.plot(a_curve, v_curve, 'k-', lw=1.2)
ax.plot(s_curve, v_curve, 'k-', lw=1.2)

# 10) restrict x‐axis to [a_v, s_v]
lower_bound = min(0, a_v - 0.1)
upper_bound = max(1, s_v + 0.1)
ax.set_xlim(lower_bound, upper_bound)

# 11) horizontal line at v_line (now only shows between a_v and s_v)
if a_v < 0:
    ax.hlines(v_line, a_v, 0, colors='black', linestyles='--', linewidth=1.5)
# middle segment (inside support)
ax.hlines(v_line, 0, 1, colors='white', linestyles='--', linewidth=1.5)
# right segment (outside support)
if s_v > 1:
    ax.hlines(v_line, 1, s_v, colors='black', linestyles='--', linewidth=1.5)

# 12) markers & labels
ax.scatter([a_v, s_v], [v_line, v_line],
           c='white', edgecolor='black', zorder=3)
for x_pt, label in [(a_v, r'$a_v$'), (s_v, r'$s_v$')]:
    col = 'white' if (0 <= x_pt <= 1) else 'black'
    ax.text(x_pt, v_line + 0.02, label,
            ha='center', color=col, fontsize=14, zorder=3)

# 13) finalize
ax.set_xlabel('$u$')
ax.set_ylabel('$v$')
ax.set_title(
    f'Contour of $c(u,v)$ for $x={x0}$\n'
    f'line $v={v_line:.3f}$, support [{a_v:.3f}, {s_v:.3f}]'
)
plt.tight_layout()
plt.show()