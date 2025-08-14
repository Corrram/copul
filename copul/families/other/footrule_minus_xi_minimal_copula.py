import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

from copul.families.core.biv_copula import BivCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula


class RepairedLowerBoundaryCopula(BivCopula):
    r"""
    A valid, one-parameter family of copulas that are candidates for the
    lower boundary of the attainable region for (xi, psi).

    This copula is a repaired version of the analytical solution from the
    paper "The exact region between Chatterjee's rank correlation and
    Spearman's footrule". The repair enforces the necessary monotonicity
    condition by constructing a "non-decreasing envelope" for the function
    v -> h(t,v), creating a "plateau of constancy" in the middle v-region.

    -----------------
    Parameter mu
    -----------------
    mu ∈ [1/2, ∞). This parameter controls the thresholds of the piecewise
    regions. As mu -> ∞, the copula approaches the independence copula.
    """

    # symbolic parameter & admissible interval
    mu = sp.symbols("mu", real=True)
    params = [mu]
    intervals = {"mu": sp.Interval(sp.S(1) / 2, sp.oo)}
    special_cases = {sp.oo: BivIndependenceCopula}

    # convenience symbols
    u, v = sp.symbols("u v", positive=True)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["mu"] = args[0]
        super().__init__(**kwargs)

    def h_vectorized(self, t, v):
        """
        Vectorized implementation of the repaired h(t,v) function.
        """
        # Ensure inputs are numpy arrays for broadcasting
        t = np.asarray(t, dtype=float)
        v = np.asarray(v, dtype=float)

        # 1. Calculate the original, non-monotonic h matrix from the paper's theorem
        h_original = np.zeros_like(v)
        mu = float(self.mu)
        v0 = 1 / (2 * mu + 1)
        v1 = (2 * mu) / (2 * mu + 1)

        mask_zone1 = v <= v0
        mask_zone3 = v > v1
        mask_zone2 = ~mask_zone1 & ~mask_zone3

        # Zone 1
        idx1 = np.where(mask_zone1)
        t1, v1_vals = t[idx1], v[idx1]
        h_original[idx1] = np.where(t1 <= v1_vals, 0.0, v1_vals / (1 - v1_vals + 1e-12))

        # Zone 3
        idx3 = np.where(mask_zone3)
        t3, v3_vals = t[idx3], v[idx3]
        h_original[idx3] = np.where(t3 <= v3_vals, 2.0 - 1.0 / (v3_vals + 1e-12), 1.0)

        # Zone 2 (The source of the "dip")
        idx2 = np.where(mask_zone2)
        t2, v2_vals = t[idx2], v[idx2]
        h_lower = v2_vals - (1 - v2_vals) / (2 * mu)
        h_upper = v2_vals + v2_vals / (2 * mu)
        h_original[idx2] = np.where(t2 <= v2_vals, h_lower, h_upper)

        # 2. Repair the matrix by creating the non-decreasing envelope.
        # This enforces the plateau of constancy and guarantees validity.
        h_repaired = np.maximum.accumulate(h_original, axis=1)

        return h_repaired

    def cdf_vectorized(self, u, v, n_steps=1000):
        """
        Vectorized CDF via numerical integration of the repaired h(t,v).
        A symbolic CDF for this complex structure is not feasible.
        """
        u = np.asarray(u)
        v = np.asarray(v)
        # We need to compute C(u,v) = integral from 0 to u of h(t,v) dt
        # This is done for each (u,v) pair.

        # This implementation is for scalar u,v for simplicity.
        # A fully vectorized version would be more complex.
        if u.shape or v.shape:
            return super().cdf_vectorized(u, v)  # Fallback to base class

        t_steps = np.linspace(0, u, n_steps)
        h_vals = self.h_vectorized(t_steps, np.full_like(t_steps, v))

        # Use trapezoidal rule for integration
        return np.trapz(h_vals, t_steps)

    def plot_repaired_slice(self, t_val=0.4, n=500):
        """
        Plots the original vs. the repaired function v -> h(t,v)
        to clearly show the effect of the plateau.
        """
        v_coords = np.linspace(0.001, 0.999, n)
        t_coords = np.full_like(v_coords, t_val)

        # The repaired function from this class
        h_repaired_vals = self.h_vectorized(t_coords[np.newaxis, :], v_coords[np.newaxis, :])[0]

        # For comparison, get the original function with the dip
        original_func = LowerBoundaryHFunction(self.mu)
        h_original_vals = original_func.h_vectorized(t_coords, v_coords)

        plt.figure(figsize=(12, 6))
        plt.plot(v_coords, h_original_vals, 'r--', label='Original Invalid Function (with dip)')
        plt.plot(v_coords, h_repaired_vals, 'b-', lw=2,
                 label='Repaired Valid Function (with plateau)')

        plt.axvline(x=self.v0, color='k', linestyle='--', label=f'v0={self.v0:.2f}')
        plt.axvline(x=self.v1, color='k', linestyle=':', label=f'v1={self.v1:.2f}')
        plt.xlabel('v')
        plt.ylabel(f'h(t={t_val}, v)')
        plt.title(f'Repairing the Structure with a Plateau (μ = {self.mu:.2f})')
        plt.grid(True, linestyle=':')
        plt.ylim(bottom=-0.1)
        plt.legend()
        plt.show()


# We include the original class to show the comparison in the plot
class LowerBoundaryHFunction:
    def __init__(self, mu): self.mu = float(mu); self.v0 = 1 / (2 * mu + 1); self.v1 = (2 * mu) / (
                2 * mu + 1)

    def h_vectorized(self, t, v):
        t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float);
        res = np.zeros_like(v)
        mu, v0, v1 = self.mu, self.v0, self.v1;
        mask1, mask3 = v <= v0, v > v1;
        mask2 = ~mask1 & ~mask3
        idx1 = np.where(mask1);
        t1, v1_vals = t[idx1], v[idx1];
        res[idx1] = np.where(t1 <= v1_vals, 0.0, v1_vals / (1 - v1_vals + 1e-12))
        idx3 = np.where(mask3);
        t3, v3_vals = t[idx3], v[idx3];
        res[idx3] = np.where(t3 <= v3_vals, 2.0 - 1.0 / (v3_vals + 1e-12), 1.0)
        idx2 = np.where(mask2);
        t2, v2_vals = t[idx2], v[idx2];
        h_lower = v2_vals - (1 - v2_vals) / (2 * mu);
        h_upper = v2_vals + v2_vals / (2 * mu)
        res[idx2] = np.where(t2 <= v2_vals, h_lower, h_upper)
        return res


if __name__ == "__main__":
    # Instantiate the new, valid copula family
    copula = RepairedLowerBoundaryCopula(mu=1.0)
    copula.plot_cdf()
    copula.plot_pdf()
    copula.plot_cond_distr_1()
    copula.plot_cond_distr_2()