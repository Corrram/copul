import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

# 2) piecewise s(v) and derivative, defined in terms of b0
def s_small(v, b):
    if v <= b/2:
        return np.sqrt(2 * v * b)
    elif v <= 1 - b/2:
        return v + b/2
    else:
        return 1 + b - np.sqrt(2 * b * (1 - v))

def ds_small(v, b):
    if v <= b/2:
        return np.sqrt(b / (2 * v))
    elif v <= 1 - b/2:
        return 1.0
    else:
        return b / np.sqrt(2 * b * (1 - v))

def s_large(v, b):
    if v <= 1 / (2 * b):
        return np.sqrt(2 * v * b)
    elif v <= 1 - 1 / (2 * b):
        return v * b + 1/2
    else:
        return 1 + b - np.sqrt(2 * b * (1 - v))

def ds_large(v, b):
    if v <= 1 / (2 * b):
        return np.sqrt(b / (2 * v))
    elif v <= 1 - 1 / (2 * b):
        return b
    else:
        return b / np.sqrt(2 * b * (1 - v))

def main(b0, v_line):
    # 3) select regime based on b0
    if b0 <= 1:
        s_func = lambda v: s_small(v, b0)
        ds_func = lambda v: ds_small(v, b0)
    else:
        s_func = lambda v: s_large(v, b0)
        ds_func = lambda v: ds_large(v, b0)

    # 4) support endpoint a(v)
    def a_of_v(v):
        return s_func(v) - b0

    # 5) build grid
    n = 400
    u = np.linspace(-1, 2, n)
    v = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, v)

    # 6) compute density
    C = np.zeros_like(U)
    for i, vi in enumerate(v):
        sv = s_func(vi)
        av = a_of_v(vi)
        dsv = ds_func(vi)
        mask = (U[i, :] >= av) & (U[i, :] <= sv) & (U[i, :] >= 0) & (U[i, :] <= 1)
        C[i, mask] = dsv / b0

    # largest finite C value
    C_max = np.max(C[np.isfinite(C)])

    # 7) compute support at v_line
    a_v = a_of_v(v_line)
    s_v = s_func(v_line)

    # 8) plot
    fig, ax = plt.subplots(figsize=(6, 5))
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=C_max)
    cs = ax.contourf(U, V, C, levels=300, cmap='viridis', norm=norm)
    plt.colorbar(cs)  # , label='copula density $c(u,v)$')

    # 9) plot boundary curves a(v), s(v)
    v_curve = np.linspace(0, 1, 400)
    a_curve = np.array([a_of_v(vi) for vi in v_curve])
    s_curve = np.array([s_func(vi) for vi in v_curve])
    inner_are_color = "white"  # color for the inner area
    outer_area_color = "white"
    ax.plot(a_curve, v_curve, 'k-', lw=1.2, color=inner_are_color)
    ax.plot(s_curve, v_curve, 'k-', lw=1.2, color=inner_are_color)

    # 10) set x-axis limits to show support around v_line
    lower_bound = min(0, a_v - 0.1)
    upper_bound = max(1, s_v + 0.1)
    ax.set_xlim(lower_bound, upper_bound)

    # 11) horizontal line at v_line
    if a_v < 0:
        ax.hlines(v_line, a_v, 0, colors=outer_area_color, linestyles='--', linewidth=1.5)
    # middle segment (inside support)
    ax.hlines(v_line, 0, 1, colors='white', linestyles='--', linewidth=1.5)
    # right segment (outside support)
    if s_v > 1:
        ax.hlines(v_line, 1, s_v, colors=outer_area_color, linestyles='--', linewidth=1.5)

    # 12) markers & labels
    ax.scatter([a_v, s_v], [v_line, v_line],
            c='white', edgecolor=outer_area_color, zorder=3)
    for x_pt, label in [(a_v, r'$a_v$'), (s_v, r'$s_v$')]:
        col = 'white' if (0 <= x_pt <= 1) else outer_area_color
        ax.text(x_pt, v_line + 0.02, label,
                ha='center', color=col, fontsize=14, zorder=3)

    # 13) dashed vertical bars at u = 0 and u = 1 (in white)
    ax.vlines(0, 0, 1, colors='white', linestyles='--', linewidth=1.5)
    ax.vlines(1, 0, 1, colors='white', linestyles='--', linewidth=1.5)

    # 14) finalize
    ax.set_xlabel(f'$u$')  #~~~(b={b_inv},~v={v_line}$)')
    ax.set_ylabel('$v$')
    # ax.set_title(
    #     # f'Contour of $c(u,v)$ for $b_{{inv}}={b_inv}$ (so $b=1/b_{{inv}}={b0:.3f}$)\n'
    #     # f'$b={b_inv},~v={v_line}$'
    # )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1) Parameter: now using b_inv so that b0 = 1 / b_inv
    b_inv = 0.5 # choose b_inv ≠ 0; then b0 = 1/b_inv
    b0 = 1.0 / b_inv
    v_line = 0.6
    main(b0, v_line)