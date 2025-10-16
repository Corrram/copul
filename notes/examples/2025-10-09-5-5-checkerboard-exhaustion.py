# checkerboard_fixed_4panels_antidiag_zero_5x5_flipped.py
# 5x5, equal weights (1/4) off the anti-diagonal, 0 on the anti-diagonal.
# Doubly-stochastic by construction.
# Disjoint blocks: B+ at (0,0) (off-diagonal positive), B- at (3,3) (diagonal positive).
# Y-axis flipped for matrix-style display (row 0 at bottom).

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Build 5x5 matrix: zeros on anti-diagonal, 1/4 elsewhere ----------
m = 5
P = np.zeros((m, m), dtype=float)
for i in range(m):
    for j in range(m):
        if i + j != m - 1:  # not on anti-diagonal
            P[i, j] = 1.0 / 4.0

# Sanity checks: doubly-stochastic
assert np.allclose(P.sum(axis=1), 1.0)
assert np.allclose(P.sum(axis=0), 1.0)


# ---------- Helpers ----------
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
    P2 = P.copy()
    P2[i, j] += eps
    P2[i, j + 1] -= eps
    P2[i + 1, j] -= eps
    P2[i + 1, j + 1] += eps
    return P2


def max_eps_block(P, i, j, improving=True):
    if improving:
        return min(P[i, j + 1], P[i + 1, j]) * 0.99
    else:
        return min(P[i, j], P[i + 1, j + 1]) * 0.99


# ---------- Plot helpers ----------
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
    fig, ax = plt.subplots(figsize=(5.1, 5.1))
    im = ax.imshow(M, origin="upper", interpolation="nearest", vmin=0.0, vmax=0.26)
    ax.set_title(title, fontsize=13)
    n = M.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    annotate(ax, M)
    if blocks:
        draw_blocks(ax, blocks, label_color="white")
    ax.invert_yaxis()  # <<< flip y-axis so row 0 is at the bottom
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.show()


# ---------- Disjoint blocks ----------
Bplus = (0, 0, "B+")  # off-diagonal positive
Bminus = (3, 3, "B-")  # diagonal positive

eps_plus_max = max_eps_block(P, Bplus[0], Bplus[1], improving=True)
eps_minus_max = max_eps_block(P, Bminus[0], Bminus[1], improving=False)
eps = min(0.08, eps_plus_max, eps_minus_max)

S1, T1 = S_T_for_block(P, Bplus[0], Bplus[1])
S2, T2 = S_T_for_block(P, Bminus[0], Bminus[1])
lam = (S1 / S2) if abs(S2) > 1e-12 else 0.0
lam = 1

# ---------- Apply transforms ----------
P_base = P.copy()

# First-only transforms
P_plus = t_transform(P_base, Bplus[0], Bplus[1], +eps)
P_minus = t_transform(P_base, Bminus[0], Bminus[1], -eps)

# Paired transform with SAFE capping of the second step
P_both = t_transform(P_base, Bplus[0], Bplus[1], +eps)

# Desired second step
eps2_target = -lam * eps
# Max allowed (worsening) on the CURRENT matrix after B+ step
eps2_max = max_eps_block(P_both, Bminus[0], Bminus[1], improving=False)
# Cap by nonnegativity
eps2 = np.sign(eps2_target) * min(abs(eps2_target), eps2_max)

if abs(eps2) < abs(eps2_target) - 1e-12:
    print(
        f"[note] paired B- step capped: target={eps2_target:.6f}, used={eps2:.6f}, max={eps2_max:.6f}"
    )

P_both = t_transform(P_both, Bminus[0], Bminus[1], eps2)


# ---------- Plots ----------
show_mat(P_base, "5×5 (No transform)", [], "anti5x5_none_flipped.png")
show_mat(
    P_plus, "5×5 (After first transform: B+)", [Bplus], "anti5x5_first_flipped.png"
)
show_mat(
    P_minus, "5×5 (After second transform: B-)", [Bminus], "anti5x5_second_flipped.png"
)
show_mat(
    P_both, "5×5 (After both transforms)", [Bplus, Bminus], "anti5x5_both_flipped.png"
)


# ---------- Quick diagnostics ----------
def pr(t, M):
    print(f"{t:22s} tau={tau_checkerboard(M): .6f}   xi={xi_checkerboard(M): .6f}")


pr("base", P_base)
pr("B+ only", P_plus)
pr("B- only", P_minus)
pr("paired", P_both)
print("S1,T1 (B+):", S1, T1, "  S2,T2 (B-):", S2, T2, "  lambda:", lam)
print(
    "Saved: anti5x5_none_flipped.png, anti5x5_first_flipped.png, anti5x5_second_flipped.png, anti5x5_both_flipped.png"
)
