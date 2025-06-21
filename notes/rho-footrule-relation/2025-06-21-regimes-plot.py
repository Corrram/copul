#!/usr/bin/env python3
"""
Six prototype profiles for h*_v(t)   ––   Regime A vs. Regime B
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

# ----------------- helper functions ---------------------------------
def proj(x):
    return np.minimum(1.0, np.maximum(0.0, x))


def s_v(q, d, v):
    """closed-form six-branch expression"""
    d_crit  = 1.0 / q - 1.0
    s_star  = q * (1.0 + d)

    if d <= d_crit:                            # Regime A
        v1, v2 = 0.5 * s_star, 1 - 0.5 * s_star
        if v <= v1:                            # A-1
            return np.sqrt(2 * q * (1 + d) * v)
        elif v <= v2:                          # A-2
            return v + 0.5 * s_star
        else:                                  # A-3
            return 1 + s_star - np.sqrt(2 * q * (1 + d) * (1 - v))
    else:                                      # Regime B
        v_star = 1.0 / (2 * q * (1 + d))
        v0     = 1.0 - v_star
        if v <= v_star:                        # B-0a
            return np.sqrt(2 * q * (1 + d) * v)
        elif v <= v0:                          # B-0b
            return 0.5 + q * (1 + d) * v
        else:                                  # B-1
            return 1 + s_star - np.sqrt(2 * q * (1 + d) * (1 - v))


def h_star(q, d, v, t):
    b  = 1.0 / q
    sv = s_v(q, d, v)
    raw = b * (sv - t) - d * (t <= v)
    return proj(raw)

# --------------------------------------------------------------------
q       = 0.25
d_crit  = 1 / q - 1                       # = 3.0
d_A     = 1.0                             # Regime A   (d < d_crit)
d_B     = 4.0                             # Regime B   (d > d_crit)

t   = np.linspace(0, 1, 2001)
dt  = t[1] - t[0]

# ---------------- choose v so that every feature is visible ----------
# Regime A (s_star = 0.50,  v1 = 0.25,  v2 = 0.75)
v_A = [0.05,       # A-1 : short, no plateau
       0.40,       # A-2 : inner plateau on [0,0.15]
       0.90]       # A-3 : outer ramp truncated

# Regime B (s_star = 1.25, v_star = 0.20, v0 = 0.80)
v_B = [0.15,       # B-0a : s_v ≤ 1, outer ramp truncated *at* 1
       0.25,       # B-0b : s_v very close to 1 ⇒ NO clipping to 1
       0.90]       # B-1  : clear inner plateau

titles_A = [r"A–1 (no inner plateau)",
            r"A–2 (inner plateau)",
            r"A–3 (outer ramp trnc.)"]

titles_B = [r"B–0a ($s_v\leq 1$)",
            r"B–0b ($s_v>1$, no inner plateau)",
            r"B–1 (inner plateau)"]

# --------------------------------------------------------------------
fig, axmat = plt.subplots(2, 3, figsize=(13, 6), sharey=True)

for ax, v, ttl in zip(axmat[0], v_A, titles_A):
    y = h_star(q, d_A, v, t).copy()
    y[np.abs(t - v) < dt/2] = np.nan      # hide the 1-pixel jump
    ax.plot(t, y, lw=2.4, color="tab:blue")
    ax.set_title(ttl, pad=6)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$h_v^{\!*}(t)$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(ls="--", alpha=.34)

for ax, v, ttl in zip(axmat[1], v_B, titles_B):
    y = h_star(q, d_B, v, t).copy()
    y[np.abs(t - v) < dt/2] = np.nan
    ax.plot(t, y, lw=2.4, color="tab:orange")
    ax.set_title(ttl, pad=6)
    ax.set_xlabel(r"$t$")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(ls="--", alpha=.34)

fig.subplots_adjust(top=0.88)
fig.suptitle(
    rf"Regime A: $d={d_A}$ < $d_{{\mathrm{{crit}}}}={d_crit:.1f}$"
    r"\qquad---\qquad"
    rf"Regime B: $d={d_B}$ > $d_{{\mathrm{{crit}}}}={d_crit:.1f}$",
    fontsize=13
)
fig.tight_layout()
plt.show()
