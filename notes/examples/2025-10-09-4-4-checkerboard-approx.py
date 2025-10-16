# checkerboard_fixed_4panels_visibleblocks_4x4.py
# 4x4 version:
# - disjoint 2x2 blocks: B+ at (0,0), B- at (2,2)
# - shows only B+ in first transform, only B- in second, both in the paired case
# - labels ("B+","B-") are white for visibility
# - fixed color scale, thick grid, annotated cell values

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Base matrix (4x4, doubly-stochastic) ----------
# Integers sum to 4 in each row/col, then divide by 4 -> doubly-stochastic
m = 4
P_base = np.zeros((m, m), dtype=float)
for i in range(m):
    for j in range(m):
        if i + j != m - 1:  # not on anti-diagonal
            P_base[i, j] = 1.0 / 3.0
# B+ (off-diagonal positive): at (0,0), need P[0,1]>0 and P[1,0]>0  -> both 0.25 OK
# B- (diagonal positive): at (2,2), need P[2,2]>0 and P[3,3]>0      -> 0.25 and 0.5 OK


# ---------- Helper functions ----------
def row_cumulatives(P):
    m = P.shape[0]
    H = np.zeros((m, m + 1))
    for j in range(1, m + 1):
        H[:, j] = m * P[:, :j].sum(axis=1)
    return H


def xi_checkerboard(P):
    m = P.shape[0]
    H = row_cumulatives(P)
    s = 0.0
    for i in range(m):
        for j in range(1, m + 1):
            Hijm1 = H[i, j - 1]
            pij = P[i, j - 1]
            s += (Hijm1**2) / (m**2) + (Hijm1 / m) * pij + (1 / 3) * (pij**2)
    return 6 * s - 2


def K_matrix(P):
    m = P.shape[0]
    S = P.cumsum(axis=0).cumsum(axis=1)

    def rect_sum(i1, j1, i2, j2):
        if i1 > i2 or j1 > j2:
            return 0.0
        tot = S[i2, j2]
        if i1 > 0:
            tot -= S[i1 - 1, j2]
        if j1 > 0:
            tot -= S[i2, j1 - 1]
        if i1 > 0 and j1 > 0:
            tot += S[i1 - 1, j1 - 1]
        return tot

    K = np.zeros_like(P)
    for r in range(m):
        for s in range(m):
            gt_gt = rect_sum(r + 1, s + 1, m - 1, m - 1)
            lt_lt = rect_sum(0, 0, r - 1, s - 1)
            gt_lt = rect_sum(r + 1, 0, m - 1, s - 1)
            lt_gt = rect_sum(0, s + 1, r - 1, m - 1)
            K[r, s] = gt_gt + lt_lt - gt_lt - lt_gt
    return K


def tau_checkerboard(P):
    K = K_matrix(P)
    return 4 * np.sum(P * K) - 1


def S_T_for_block(P, i, j):
    m = P.shape[0]
    H = row_cumulatives(P)
    Hijm1 = H[i, j]
    Hip1jm1 = H[i + 1, j]
    pij, pi1j = P[i, j], P[i + 1, j]
    pij1, pi1j1 = P[i, j + 1], P[i + 1, j + 1]
    S = (6 / m) * (
        (2 / m) * (Hijm1 - Hip1jm1) + (5 / 3) * (pij - pi1j) + (1 / 3) * (pij1 - pi1j1)
    )
    K = K_matrix(P)
    dK = K[i, j] - K[i, j + 1] + K[i + 1, j + 1] - K[i + 1, j]
    T = 4 * dK
    return S, T


def t_transform(P, i, j, eps):
    """Apply a 2x2 T-transform with epsilon at block (i,j)."""
    P2 = P.copy()
    P2[i, j] += eps
    P2[i, j + 1] -= eps
    P2[i + 1, j] -= eps
    P2[i + 1, j + 1] += eps
    return P2


def max_eps_block(P, i, j, improving=True):
    """Max epsilon to keep entries nonnegative within the 2x2 block."""
    if improving:
        # move from off-diagonals to diagonals
        return min(P[i, j + 1], P[i + 1, j]) * 0.99
    else:
        # move from diagonals to off-diagonals
        return min(P[i, j], P[i + 1, j + 1]) * 0.99


# ---------- Plot helper ----------
def annotate(ax, M):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )


def draw_blocks(ax, blocks, label_color="white"):
    for i, j, lbl in blocks:
        rect = Rectangle(
            (j - 0.5, i - 0.5), 2, 2, fill=False, linewidth=2, edgecolor="black"
        )
        ax.add_patch(rect)
        ax.text(
            j + 0.5,
            i + 0.5,
            lbl,
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color=label_color,
        )


def show_mat(M, title, blocks, fname):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(M, origin="upper", interpolation="nearest", vmin=0.0, vmax=0.6)
    ax.set_title(title, fontsize=13)
    n = M.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    annotate(ax, M)
    if blocks:
        draw_blocks(ax, blocks, label_color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.show()


# ---------- Block setup (disjoint) ----------
Bplus = (0, 0, "B+")  # top-left 2x2
Bminus = (2, 2, "B-")  # bottom-right 2x2 (disjoint from B+)

P = P_base.copy()
S1, T1 = S_T_for_block(P, Bplus[0], Bplus[1])
S2, T2 = S_T_for_block(P, Bminus[0], Bminus[1])

# Choose eps safely under nonnegativity
eps_plus_max = max_eps_block(P, Bplus[0], Bplus[1], improving=True)
eps_minus_max = max_eps_block(P, Bminus[0], Bminus[1], improving=False)
eps = min(0.12, eps_plus_max, eps_minus_max)  # conservative choice

# Paired-move slope to (approximately) cancel xi to first order
lam = (S1 / S2) if abs(S2) > 1e-12 else 0.0

# ---------- Apply transforms ----------
P_plus = t_transform(P, Bplus[0], Bplus[1], +eps)  # diagonal-improving on B+
P_minus = t_transform(P, Bminus[0], Bminus[1], -eps)  # diagonal-worsening on B-
P_both = t_transform(P, Bplus[0], Bplus[1], +eps)
P_both = t_transform(P_both, Bminus[0], Bminus[1], -lam * eps)

# ---------- Plot four panels ----------
show_mat(P, "4×4 (No transform)", [], "fixed4x4_none.png")
show_mat(P_plus, "4×4 (After first transform: B+)", [Bplus], "fixed4x4_first.png")
show_mat(P_minus, "4×4 (After second transform: B-)", [Bminus], "fixed4x4_second.png")
show_mat(P_both, "4×4 (After both transforms)", [Bplus, Bminus], "fixed4x4_both.png")

print(
    "Saved: fixed4x4_none.png, fixed4x4_first.png, fixed4x4_second.png, fixed4x4_both.png"
)

# (Optional) quick printouts
print("tau, xi (base):   ", tau_checkerboard(P), xi_checkerboard(P))
print("tau, xi (B+):     ", tau_checkerboard(P_plus), xi_checkerboard(P_plus))
print("tau, xi (B-):     ", tau_checkerboard(P_minus), xi_checkerboard(P_minus))
print("tau, xi (paired): ", tau_checkerboard(P_both), xi_checkerboard(P_both))
print("S1,T1 (B+):", S1, T1, "  S2,T2 (B-):", S2, T2, "  lambda:", lam)
