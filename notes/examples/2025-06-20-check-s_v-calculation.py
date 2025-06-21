#!/usr/bin/env python3
"""
Diagnostic checker for the updated local multiplier s_v(d).

  • identifies which analytical branch 
    (A-1, A-2, A-3, B-0a, B-0b, B-1) is used
  • reports pass / fail counts per branch
  • prints the parameters that give the worst mass-constraint error

Only NumPy is required.
"""
import numpy as np
from collections import defaultdict

# ----------------------------------------------------------------------
# Basic primitives
# ----------------------------------------------------------------------
def proj(x):
    """Project to [0,1] element-wise."""
    return np.minimum(1.0, np.maximum(0.0, x))


def classify_branch(q, d, v):
    """
    Return (s_v, branch_label) for the six analytical cases
    described in the four-case master formula (Regime A plus
    Regime B split into B-0a, B-0b, B-1).
    """
    d_crit = 1.0 / q - 1.0
    s_star = q * (1.0 + d)
    v1 = 0.5 * s_star
    v2 = 1.0 - v1

    if d <= d_crit:  # ----- Regime A : mixed ramps -------------------
        if v <= v1:        # A-1  (no plateau)
            s_v = np.sqrt(2.0 * q * (1.0 + d) * v)
            return s_v, "A-1"
        elif v <= v2:      # A-2  (central plateau)
            s_v = v + 0.5 * q * (1.0 + d)
            return s_v, "A-2"
        else:              # A-3  (truncated outer ramp)
            s_v = 1.0 + q * (1.0 + d) - np.sqrt(
                    2.0 * q * (1.0 + d) * (1.0 - v)
                  )
            return s_v, "A-3"

    else:          # ----- Regime B : fully truncated -----------------
        v_star = 1.0 / (2.0 * s_star)                #  v_{\!*}
        v0     = 1.0 - v_star                        #  v_0
        if v <= v_star:             # B-0a  (outer ramp ends at t=s_v≤1)
            s_v = np.sqrt(2.0 * q * (1.0 + d) * v)
            return s_v, "B-0a"
        elif v <= v0:               # B-0b  (no plateau, outer ramp hits t=1)
            s_v = 0.5 + q * (1.0 + d) * v
            return s_v, "B-0b"
        else:                       # B-1   (inner plateau present)
            s_v = 1.0 + s_star - np.sqrt(
                    2.0 * q * (1.0 + d) * (1.0 - v)
                  )
            return s_v, "B-1"


def h_profile(q, d, v, t_grid):
    """Evaluate h*_v(t) on t_grid using the closed-form s_v."""
    b = 1.0 / q
    s_v, _ = classify_branch(q, d, v)
    core = b * (s_v - t_grid) - d * (t_grid <= v)
    return proj(core)


# ----------------------------------------------------------------------
# Main diagnostic routine
# ----------------------------------------------------------------------
def diagnose(n_samples=10_000, n_grid=20_001, tol=1e-8, seed=42):
    rng = np.random.default_rng(seed)

    # integration grid on [0,1]
    t = np.linspace(0.0, 1.0, n_grid)
    dt = t[1] - t[0]

    branches = ["A-1", "A-2", "A-3", "B-0a", "B-0b", "B-1"]
    stats = defaultdict(lambda: {"pass": 0, "fail": 0, "worst_err": 0.0})
    worst_global = {"err": -1.0}

    for _ in range(n_samples):
        # sample parameters
        q = rng.uniform(5e-3, 0.495)       # 0 < q < 0.5
        d = rng.uniform(-0.999, 4.0)       # generous range
        v = rng.random()                   # 0 ≤ v ≤ 1

        s_v, branch = classify_branch(q, d, v)
        h_vals = h_profile(q, d, v, t)
        mass = np.trapz(h_vals, dx=dt)
        err = abs(mass - v)

        # update branch-wise statistics
        if err < tol:
            stats[branch]["pass"] += 1
        else:
            stats[branch]["fail"] += 1
        if err > stats[branch]["worst_err"]:
            stats[branch]["worst_err"] = err

        # track global worst offender
        if err > worst_global["err"]:
            worst_global = {
                "err": err,
                "q": q,
                "d": d,
                "v": v,
                "branch": branch,
                "s_v": s_v,
            }

    # ------------------  report  ------------------
    print(f"Tolerance for pass: {tol:.1e}\n")
    for br in branches:
        p = stats[br]["pass"]
        f = stats[br]["fail"]
        w = stats[br]["worst_err"]
        total = p + f
        if total == 0:
            continue
        print(f"Branch {br:>4}: {p:5d} pass   {f:5d} fail   "
              f"worst |∫h−v| = {w:.3e}")
    print("\nWorst overall sample:")
    print(f"  branch  : {worst_global['branch']}")
    print(f"  q       : {worst_global['q']:.6f}")
    print(f"  d       : {worst_global['d']:.6f}")
    print(f"  v       : {worst_global['v']:.6f}")
    print(f"  s_v     : {worst_global['s_v']:.6f}")
    print(f"  |∫h−v|  : {worst_global['err']:.3e}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    diagnose(n_samples=10000, tol=1e-5)
