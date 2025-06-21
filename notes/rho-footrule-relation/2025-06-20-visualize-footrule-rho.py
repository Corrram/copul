#!/usr/bin/env python3
"""
Plot 6·J0*(c) – 3 for the limiting case a → 0⁻ in
‘A Complete Semi-Analytical Solution to a Concave Functional Optimization Problem’.

The cubic that fixes the global multiplier d₀*(c) is taken from the
corrected derivation (Eq. 3.5), which in the a→0⁻ limit simplifies to:
    8(1-c)·d·(d+1)² = 0 .
For every c∈[−½,1), this equation has a unique non-negative root d=0.

NOTE: The formula for J₀*(d) must also be re-derived. The function used
here is from the old analysis and is incorrect. This script is for demonstration
of the new d₀*(c) only.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- core algebra -----------------------------------------------------------

def cubic_F(d: float, c: float) -> float:
    """CORRECTED cubic for a=0 whose root gives d₀*(c)."""
    # If c=1, the equation vanishes, F(d)=0 for all d.
    if c == 1.0:
        return 0.0
    # For c<1, this is the correct cubic from F(d)=0 with q=0.
    return 8 * (1 - c) * d * (d + 1)**2

def d_star(c: float) -> float:
    """
    Locate the unique non-negative root of the corrected cubic_F(·,c) for a→0⁻.
    """
    # For any c < 1, the unique non-negative root of 8(1-c)d(d+1)² = 0 is d=0.
    # For the singular case c=1, the limiting value of d as a→0⁻ is also 0.
    # The original bisection algorithm is no longer needed.
    return 0.0

def J0_star(d: float) -> float:
    """
    WARNING: This formula is from the original, INCORRECT derivation.
    A new formula consistent with the corrected d₀*(c) is needed.
    Using this function will NOT produce the correct final plot.
    """
    return (1/6 - d/4) * (1 - d) - (1/12) * (1 - d)**4

# --- sampling & plotting ----------------------------------------------------

cs = np.linspace(-0.5, 1.0, 400)
ys = np.empty_like(cs)

for i, c in enumerate(cs):
    # With the corrected framework, d₀*(c) is 0 for all c in the range.
    d = d_star(c)
    # The value of J₀* will therefore be constant, based on the old formula.
    # This demonstrates the inconsistency.
    ys[i] = 6.0 * J0_star(d) - 3.0

# We know the true value at c=1 is 1. We add it manually to the plot.
cs_true = np.array([1.0])
ys_true = np.array([1.0])


plt.figure(figsize=(7, 6))
# Plot the result from the script (which is incorrect but uses the correct d*)
plt.plot(cs, ys, linewidth=1.8, linestyle='--',
         label=r'Incorrect value from $J^{\star}_{0}(d^{\star}_{0}(c))$ where $d^{\star}_{0}=0$')
# Plot the known correct point
plt.plot(cs_true, ys_true, 'ro', markersize=8, label=r'Known correct value at $c=1$')

plt.xlabel(r'$c$', fontsize=12)
plt.ylabel(r'$6J^{\star}_{0}(c) - 3$', fontsize=12)
plt.title(r'Value of $6J^{\star}_{0}-3$ using correct $d^{\star}_{0}(c)$ but old $J^{\star}_{0}$ formula',
          fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()