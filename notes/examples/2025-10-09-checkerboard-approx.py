# checkerboard_exhaustion_plot.py
# Illustrative plot for the checkerboard exhaustion with paired T-transforms.
# - Produces two figures (before/after) highlighting two disjoint 2x2 blocks.
# - Rectangles and labels ("B+" / "B-") are white as requested.
# - Also prints xi/tau before & after and saves figures to the current folder.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

np.random.seed(7)

# ---------- Utilities for checkerboard copulas ----------


def sinkhorn(A, iters=200, eps=1e-12):
    """Project a positive matrix onto the Birkhoff polytope (approximately doubly-stochastic)."""
    P = A.copy()
    for _ in range(iters):
        P = P / (P.sum(axis=1, keepdims=True) + eps)
        P = P / (P.sum(axis=0, keepdims=True) + eps)
    return P


def make_doubly_stochastic(m):
    A = np.random.rand(m, m) + 0.1  # strictly positive
    return sinkhorn(A)


def row_cumulatives(P):
    """H[i, j] = m * sum_{k<=j} p_{i,k}, for j = 0..m (with H[:,0]=0)."""
    m = P.shape[0]
    H = np.zeros((m, m + 1))
    for j in range(1, m + 1):
        H[:, j] = m * P[:, :j].sum(axis=1)
    return H


def xi_checkerboard(P):
    """Chatterjee's xi for checkerboard (from the paper's formula)."""
    m = P.shape[0]
    H = row_cumulatives(P)
    s = 0.0
    for i in range(m):
        for j in range(1, m + 1):
            Hijm1 = H[i, j - 1]  # m * sum_{k<j} p_{i,k}
            pij = P[i, j - 1]
            s += (Hijm1**2) / (m**2) + (Hijm1 / m) * pij + (1 / 3) * (pij**2)
    return 6 * s - 2


def K_matrix(P):
    """Compute K_{r,s} (0-based)."""
    m = P.shape[0]
    S = P.cumsum(axis=0).cumsum(axis=1)  # 2D prefix sums

    def rect_sum(i1, j1, i2, j2):
        if i1 > i2 or j1 > j2:
            return 0.0
        total = S[i2, j2]
        if i1 > 0:
            total -= S[i1 - 1, j2]
        if j1 > 0:
            total -= S[i2, j1 - 1]
        if i1 > 0 and j1 > 0:
            total += S[i1 - 1, j1 - 1]
        return total

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
    """Compute S_{i,j} and T_{i,j} for 2x2 block with top-left (i,j) (0-based)."""
    m = P.shape[0]
    H = row_cumulatives(P)
    Hijm1 = H[i, j]
    Hip1_jm1 = H[i + 1, j]
    pij, pi1j = P[i, j], P[i + 1, j]
    pij1, pi1j1 = P[i, j + 1], P[i + 1, j + 1]
    S = (6 / m) * (
        (2 / m) * (Hijm1 - Hip1_jm1) + (5 / 3) * (pij - pi1j) + (1 / 3) * (pij1 - pi1j1)
    )
    K = K_matrix(P)
    dK = K[i, j] - K[i, j + 1] + K[i + 1, j + 1] - K[i + 1, j]
    T = 4 * dK
    return S, T


def t_transform(P, i, j, eps):
    """T-transform at block (i,j): [[+eps,-eps],[-eps,+eps]]."""
    P2 = P.copy()
    P2[i, j] += eps
    P2[i, j + 1] -= eps
    P2[i + 1, j] -= eps
    P2[i + 1, j + 1] += eps
    return P2


def find_blocks(P):
    """Find disjoint off-diagonal-positive B+ and diagonal-positive B- blocks with a ratio gap."""
    m = P.shape[0]
    offdiag, diag = [], []
    for i in range(m - 1):
        for j in range(m - 1):
            if P[i, j + 1] > 1e-12 and P[i + 1, j] > 1e-12:
                S, T = S_T_for_block(P, i, j)
                if abs(S) > 1e-12:
                    offdiag.append((i, j, S, T))
            if P[i, j] > 1e-12 and P[i + 1, j + 1] > 1e-12:
                S, T = S_T_for_block(P, i, j)
                if abs(S) > 1e-12:
                    diag.append((i, j, S, T))
    offdiag.sort(key=lambda x: x[3] / x[2])
    diag.sort(key=lambda x: x[3] / x[2])

    for i1, j1, S1, T1 in offdiag[::-1]:  # largest ratio first
        for i2, j2, S2, T2 in diag:  # smallest ratio first
            disjoint = (abs(i1 - i2) > 1) or (abs(j1 - j2) > 1)
            if disjoint and (T1 / S1 > T2 / S2 + 1e-6):
                return (i1, j1, S1, T1), (i2, j2, S2, T2)
    # fallback: first disjoint pair
    for i1, j1, S1, T1 in offdiag:
        for i2, j2, S2, T2 in diag:
            disjoint = (abs(i1 - i2) > 1) or (abs(j1 - j2) > 1)
            if disjoint:
                return (i1, j1, S1, T1), (i2, j2, S2, T2)
    return None, None


# ---------- Build an example ----------

m = 10
P = make_doubly_stochastic(m)

Bplus, Bminus = find_blocks(P)
if Bplus is None or Bminus is None:
    # fallback to two disjoint positions
    Bplus = (2, 5, *S_T_for_block(P, 2, 5))
    Bminus = (6, 2, *S_T_for_block(P, 6, 2))

(i1, j1, S1, T1) = Bplus
(i2, j2, S2, T2) = Bminus

tau_before = tau_checkerboard(P)
xi_before = xi_checkerboard(P)

# Choose small eps and lambda so that dxi ≈ 0: eps2 = -lambda * eps1 with lambda = S1/S2
eps = 0.04 / m  # small to keep non-negativity
lam = S1 / S2 if abs(S2) > 1e-12 else 0.0

# Conservative feasibility guard for eps (avoid negatives in the two 2x2 blocks)
eps1_max = min(P[i1, j1 + 1], P[i1 + 1, j1], P[i1 + 1, j1 + 1]) * 0.9
eps2_max = min(P[i2, j2], P[i2, j2 + 1], P[i2 + 1, j2]) * 0.9
if abs(lam) > 1e-12:
    eps = min(eps, eps1_max, eps2_max / abs(lam))
else:
    eps = min(eps, eps1_max)

P1 = t_transform(P, i1, j1, +eps)  # improving at B+
P2 = t_transform(P1, i2, j2, -lam * eps)  # worsening at B-

tau_after = tau_checkerboard(P2)
xi_after = xi_checkerboard(P2)

# ---------- Plotting (white rectangles and labels) ----------


def plot_matrix_with_blocks(P, blocks, title, fname):
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    im = ax.imshow(P, interpolation="nearest", origin="upper")
    ax.set_title(title)
    ax.set_xticks(range(P.shape[1]))
    ax.set_yticks(range(P.shape[0]))

    # Draw white rectangles & labels
    for i, j, label in blocks:
        rect = Rectangle(
            (j - 0.5, i - 0.5), 2, 2, fill=False, linewidth=2, edgecolor="white"
        )
        ax.add_patch(rect)
        ax.text(
            j + 0.5,
            i + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="white",
        )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.show()


ratio1 = T1 / S1 if abs(S1) > 1e-12 else np.nan
ratio2 = T2 / S2 if abs(S2) > 1e-12 else np.nan

plot_matrix_with_blocks(
    P,
    blocks=[(i1, j1, f"B+ (T/S≈{ratio1:.2f})"), (i2, j2, f"B- (T/S≈{ratio2:.2f})")],
    title="Checkerboard P (Before)",
    fname="checkerboard_before.png",
)

plot_matrix_with_blocks(
    P2,
    blocks=[(i1, j1, "B+"), (i2, j2, "B-")],
    title="After paired T-transforms (P')",
    fname="checkerboard_after.png",
)

# ---------- Print summary ----------

summary = pd.DataFrame(
    {
        "metric": ["xi", "tau"],
        "before": [xi_before, tau_before],
        "after": [xi_after, tau_after],
        "delta": [xi_after - xi_before, tau_after - tau_before],
    }
)

print(summary.round(6))
print()
print("Chosen blocks:")
print(f"  B+ at (i={i1}, j={j1}), S={S1:.6f}, T={T1:.6f}, T/S={ratio1:.6f}")
print(f"  B- at (i={i2}, j={j2}), S={S2:.6f}, T={T2:.6f}, T/S={ratio2:.6f}")
print(f"lambda (for Δxi≈0) = {lam:.6f}, eps = {eps:.6f}")
print("Saved: checkerboard_before.png, checkerboard_after.png")
