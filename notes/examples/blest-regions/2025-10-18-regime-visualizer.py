#!/usr/bin/env python3
"""
Illustration of Lemma (Piecewise formulas for v(q) and v'(q)).

- Plots v(q) and v'(q) for two representative values:
  * b_hi > 1  (e.g., 2.0)
  * b_lo <= 1 (e.g., 0.7)

- Color codes the lemma's cases:
  (i)  Negative q with plateau            (q<0 and R<1)
  (ii) Negative q without plateau         (q<0 and R>=1, only if b<=1)
  (iii) Middle branch                     (0 <= q < 1 - 1/b, only if b>1)
  (iv) High q                             (1 - 1/b <= q <= 1)

- Saves figure as 'lemma_vq_cases.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Helper functions ----------


def R_of_q(q, b):
    # R = sqrt(q + 1/b); valid for q > -1/b
    return np.sqrt(q + 1.0 / b)


def r_of_q(q):
    # r = sqrt(q) for q >= 0; else 0
    out = np.zeros_like(q)
    mask = q >= 0
    out[mask] = np.sqrt(q[mask])
    return out


def v_and_vprime_piecewise(q, b):
    """
    Compute v(q) and v'(q) on a grid q ∈ (-1/b, 1], using the piecewise formulas.

    Returns:
        v, vp, case_idx
    where case_idx in {1,2,3,4} marks (i),(ii),(iii),(iv) respectively.
    """
    q = np.asarray(q)
    R = R_of_q(q, b)  # needs q > -1/b
    r = r_of_q(q)
    R - r

    v = np.full_like(q, np.nan, dtype=float)
    vp = np.full_like(q, np.nan, dtype=float)
    case = np.zeros_like(q, dtype=int)

    # Convenience thresholds
    q_min = -1.0 / b
    q_star = 1.0 - 1.0 / b  # boundary between (iii) and (iv)

    # Domain mask (open at q_min numerically)
    dom = (q > q_min) & (q <= 1.0)

    # (i) Negative q with plateau: q<0 and R<1
    mask_i = dom & (q < 0.0) & (R < 1.0)
    if np.any(mask_i):
        v[mask_i] = 1.0 - (2.0 * b / 3.0) * (R[mask_i] ** 3)
        vp[mask_i] = -b * R[mask_i]
        case[mask_i] = 1

    # (ii) Negative q without plateau: q<0 and R>=1 (only if b <= 1)
    mask_ii = dom & (q < 0.0) & (R >= 1.0) & (b <= 1.0 + 0.0)  # numerical guard
    if np.any(mask_ii):
        v[mask_ii] = b * (1.0 / 3.0 - q[mask_ii])
        vp[mask_ii] = -b
        case[mask_ii] = 2

    # (iii) Middle branch: 0 <= q < 1 - 1/b (only if b > 1)
    mask_iii = dom & (q >= 0.0) & (q < q_star) & (b > 1.0 + 0.0)
    if np.any(mask_iii):
        Rm = R[mask_iii]
        rm = r[mask_iii]
        Delta_m = Rm - rm
        v[mask_iii] = (1.0 - Rm) + b * ((Rm**3 - rm**3) / 3.0 - q[mask_iii] * (Rm - rm))
        # v'(q) = - (1/(2R)) * (1 + b * (R - r)^2)
        vp[mask_iii] = -(1.0 / (2.0 * Rm)) * (1.0 + b * (Delta_m**2))
        case[mask_iii] = 3

    # (iv) High q: 1 - 1/b <= q <= 1
    mask_iv = dom & (q >= q_star) & (q <= 1.0)
    if np.any(mask_iv):
        rm = r[mask_iv]
        v[mask_iv] = b * (1.0 / 3.0 - rm**2 + (2.0 / 3.0) * rm**3)
        vp[mask_iv] = b * (rm - 1.0)
        case[mask_iv] = 4

    return v, vp, case


def plot_for_b(ax_v, ax_vp, b, N=4000):
    """
    Make plots of v(q) and v'(q) for a given b on provided axes.
    Color by cases (i)-(iv).
    """
    # Grid in q
    q_min = -1.0 / b + 1e-8  # stay inside domain
    q = np.linspace(q_min, 1.0, N)

    v, vp, case = v_and_vprime_piecewise(q, b)

    # Colors per case
    colors = {
        1: "#1f77b4",  # blue    (i)
        2: "#ff7f0e",  # orange  (ii)
        3: "#2ca02c",  # green   (iii)
        4: "#d62728",  # red     (iv)
    }
    labels = {
        1: "(i) q<0, plateau (R<1)",
        2: "(ii) q<0, no plateau (R≥1, b≤1)",
        3: "(iii) 0≤q<1-1/b (b>1)",
        4: "(iv) 1-1/b ≤ q ≤ 1",
    }

    # Plot v(q) and v'(q) branch-wise
    for k in (1, 2, 3, 4):
        m = case == k
        if np.any(m):
            ax_v.plot(q[m], v[m], lw=2.2, color=colors[k], label=labels[k])
            ax_vp.plot(q[m], vp[m], lw=2.2, color=colors[k], label=labels[k])

    # Vertical reference lines at boundaries (if inside domain)
    q_star = 1.0 - 1.0 / b
    for ax in (ax_v, ax_vp):
        ax.axvline(0.0, color="#888", lw=1.0, ls="--", alpha=0.8)
        if q_star > q_min and q_star < 1.0:
            ax.axvline(q_star, color="#444", lw=1.0, ls=":", alpha=0.9)

    # Axis labels and titles
    ax_v.set_title(rf"$v(q)$ for $b={b}$")
    ax_v.set_xlabel(r"$q$")
    ax_v.set_ylabel(r"$v(q)$")
    ax_v.grid(True, linestyle=":", alpha=0.6)

    ax_vp.set_title(rf"$v'(q)$ for $b={b}$")
    ax_vp.set_xlabel(r"$q$")
    ax_vp.set_ylabel(r"$v'(q)$")
    ax_vp.grid(True, linestyle=":", alpha=0.6)

    # Legends (merge labels without duplicates)
    handles_v, labels_v = ax_v.get_legend_handles_labels()
    by_label_v = dict(zip(labels_v, handles_v))
    if by_label_v:
        ax_v.legend(by_label_v.values(), by_label_v.keys(), fontsize=9, frameon=True)

    handles_p, labels_p = ax_vp.get_legend_handles_labels()
    by_label_p = dict(zip(labels_p, handles_p))
    if by_label_p:
        ax_vp.legend(by_label_p.values(), by_label_p.keys(), fontsize=9, frameon=True)


# ---------- Main: two regimes side-by-side ----------

if __name__ == "__main__":
    b_hi = 2.0  # > 1 : all cases except (ii)
    b_lo = 0.7  # <= 1: enables case (ii), disables case (iii)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    # Top row: b > 1
    plot_for_b(axes[0, 0], axes[1, 0], b_hi)

    # Bottom row: b <= 1
    plot_for_b(axes[0, 1], axes[1, 1], b_lo)

    # Layout tweaks
    plt.tight_layout()
    plt.savefig("images/lemma_vq_cases.png", dpi=200)
    plt.show()
