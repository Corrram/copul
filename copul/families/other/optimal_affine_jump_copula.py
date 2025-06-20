import sympy as sp
import numpy as np
from functools import lru_cache

from copul.families.core.biv_copula import BivCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class OptimalAffineJumpCopula(BivCopula):
    # Class-level definitions
    a, c = sp.symbols("a c", real=True)
    params = [a, c]
    intervals = {
        "a": sp.Interval.Lopen(-sp.Rational(1, 2), 0),
        "c": sp.Interval(-sp.Rational(1, 2), 1),
    }
    u, v = sp.symbols("u v", positive=True)

    # ------------------------------------------------------------------
    # Symbolic Expression for d* (used by _cdf_expr)
    # ------------------------------------------------------------------
    @staticmethod
    def _d_expr(a, c):
        """Symbolic closed-form solution for d* using Cardano's method."""
        q = -a
        k3 = 8 - 8 * c + 24 * q
        k2 = 16 - 16 * c + 24 * q + 6 * q**2
        k1 = 8 - 8 * c + 6 * q
        k0 = -3 * q**2
        
        A, B, C = k2 / k3, k1 / k3, k0 / k3
        p = B - A**2 / 3
        q_dep = C - (A * B / 3) + (2 * A**3 / 27)
        
        inner_term = (q_dep / 2)**2 + (p / 3)**3
        inner_sqrt = sp.sqrt(sp.Max(0, inner_term))
        
        term1 = sp.cbrt(-q_dep / 2 + inner_sqrt)
        term2 = sp.cbrt(-q_dep / 2 - inner_sqrt)
        
        return term1 + term2 - A / 3
    
    # ------------------------------------------------------------------
    # Symbolic CDF (for PDF derivation and other symbolic tasks)
    # ------------------------------------------------------------------
    @property
    def _cdf_expr(self):
        """
        Symbolic CDF expression. This is a direct translation of the correct,
        integrated optimizer h*(t). It is very complex and slow to evaluate.
        """
        u_s, v_s, a_s, c_s = self.u, self.v, self.a, self.c
        
        d_s = self._d_expr(a_s, c_s)
        q_s = -a_s
        b_s = 1/q_s
        
        v_b_s = q_s / (2 * d_s)
        s_v_s = sp.Piecewise(
            ((1 + d_s) * v_s + q_s / 2, v_s <= v_b_s),
            ((v_s - q_s / 2 + d_s * v_s) / (1 + d_s) + q_s, True)
        )
        
        # Symbolic breakpoints
        t1_le_s = s_v_s - q_s * (1 + d_s)
        t0_le_s = s_v_s - q_s * d_s
        t1_gt_s = s_v_s - q_s
        t0_gt_s = s_v_s

        def sp_integrate_ramp(start, end, s, offset):
            return b_s * (s * (end - start) - (end**2 - start**2) / 2) - offset * (end - start)

        # Part 1: Symbolic integral from 0 to min(u,v)
        u_eff = sp.Min(u_s, v_s)
        area1_pre = sp.Max(0, sp.Min(u_eff, t1_le_s))
        start_r_pre = sp.Min(u_eff, t1_le_s)
        end_r_pre = sp.Min(u_eff, t0_le_s)
        area_r_pre = sp.Piecewise((sp_integrate_ramp(start_r_pre, end_r_pre, s_v_s, d_s), end_r_pre > start_r_pre), (0, True))
        C_pre_v = area1_pre + area_r_pre

        # Part 2: Symbolic integral from v to u
        area_p2 = sp.Max(0, sp.Min(u_s, t1_gt_s) - v_s)
        start_r2 = sp.Max(v_s, t1_gt_s)
        end_r2 = sp.Min(u_s, t0_gt_s)
        area_r2 = sp.Piecewise((sp_integrate_ramp(start_r2, end_r2, s_v_s, 0), end_r2 > start_r2), (0, True))
        C_post_v = sp.Piecewise((area_p2 + area_r2, u_s > v_s), (0, True))
        
        return C_pre_v + C_post_v

    # ------------------------------------------------------------------
    # Vectorized Numerical Implementation (for speed)
    # ------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _d_numeric(a, c):
        """Numerical implementation of Cardano's method for d*. Cached for performance."""
        q = -a
        k3 = 8 - 8*c + 24*q
        k2 = 16 - 16*c + 24*q + 6*q**2
        k1 = 8 - 8*c + 6*q
        k0 = -3*q**2
        
        A, B, C = k2/k3, k1/k3, k0/k3
        p = B - A**2/3
        q_dep = C - (A*B/3) + (2*A**3/27)
        
        inner_term = (q_dep/2)**2 + (p/3)**3
        inner_sqrt = np.sqrt(max(0, inner_term))
        
        return np.cbrt(-q_dep/2 + inner_sqrt) + np.cbrt(-q_dep/2 - inner_sqrt) - A/3

    def cdf_vectorized(self, u, v):
        """Correct, vectorized CDF for fast numerical evaluation on a grid."""
        a, c = self.a, self.c
        u_in, v_in = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        
        d = self._d_numeric(a, c)
        q = -a
        b = 1.0 / q
        
        v_b = q / (2 * d) if d > 0 else np.inf
        s_v = np.where(v_in <= v_b, v_in*(1+d) + q/2, (v_in-q/2+d*v_in)/(1+d) + q)
        
        # Define breakpoints
        t1_le = s_v - q * (1 + d)
        t0_le = s_v - q * d
        t1_gt = s_v - q
        t0_gt = s_v

        def integrate_ramp(start, end, s, offset):
            return b * (s * (end - start) - (end**2 - start**2) / 2) - offset * (end - start)

        # Part 1: Integral from 0 to min(u,v)
        u_eff = np.minimum(u_in, v_in)
        area1_pre = np.maximum(0, np.minimum(u_eff, t1_le))
        start_r_pre = np.minimum(u_eff, t1_le)
        end_r_pre = np.minimum(u_eff, t0_le)
        area_r_pre = np.where(end_r_pre > start_r_pre, integrate_ramp(start_r_pre, end_r_pre, s_v, d), 0)
        C_pre_v = area1_pre + area_r_pre

        # Part 2: Integral from v to u
        C_post_v = np.zeros_like(u_in)
        mask_post = u_in > v_in
        if np.any(mask_post):
            u_p, v_p, s_p = u_in[mask_post], v_in[mask_post], s_v[mask_post]
            t1_p, t0_p = t1_gt[mask_post], t0_gt[mask_post]
            
            start_p2 = v_p
            end_p2 = np.minimum(u_p, t1_p)
            area_p2 = np.maximum(0, end_p2 - start_p2)
            
            start_r2 = np.maximum(v_p, t1_p)
            end_r2 = np.minimum(u_p, t0_p)
            area_r2 = np.where(end_r2 > start_r2, integrate_ramp(start_r2, end_r2, s_p, 0), 0)
            C_post_v[mask_post] = area_p2 + area_r2
            
        return C_pre_v + C_post_v

    # --- Other Methods ---
    
    @property
    def _pdf_expr(self):
        # Derives the PDF by differentiating the symbolic CDF. This will be very slow.
        return self._cdf_expr.diff(self.u).diff(self.v)

    def footrule(self, numeric=False, grid=501):
        if not numeric:
            raise NotImplementedError("Symbolic calculation for footrule is not supported due to extreme complexity.")
        
        # Numeric mode uses the correct vectorized CDF.
        u_lin = np.linspace(0.5 / grid, 1 - 0.5 / grid, grid)
        v_lin = u_lin.copy()
        uu, vv = np.meshgrid(u_lin, v_lin)
        Cvals = self.cdf_vectorized(uu, vv) 
        F_numeric = 12 * np.mean(np.abs(Cvals - uu * vv))
        return float(F_numeric)

    # Required properties
    @property
    def is_absolutely_continuous(self): return True
    @property
    def is_symmetric(self): return False
# ---------------- Tiny self‑check --------------------------------------
if __name__ == "__main__":
    # Example parameters
    a_param = -0.02
    c_param = 0.5

    print(f"Testing with a={a_param}, c={c_param}\n")
    
    # Calculate d* using the numerical method
    d_val = OptimalAffineJumpCopula._d_numeric(a=a_param, c=c_param)
    print(f"Correct d* calculated via Cardano's formula = {d_val:.8f}\n")

    # Instantiate the copula
    cop = OptimalAffineJumpCopula(a=a_param, c=c_param)
    cop.plot_cdf()

    # Test CDF at a point using the new vectorized method
    u_test, v_test = 0.5, 0.6
    cdf_val = cop.cdf_vectorized(u_test, v_test)
    print(f"CDF at C({u_test}, {v_test}) = {cdf_val:.8f}")

    # Calculate footrule using the new vectorized method
    print("\nCalculating Spearman's Footrule (numeric)...")
    footrule_val = cop.footrule(numeric=True)
    print(f"Numeric Footrule = {footrule_val:.6f}")
    
    # Run checkerboard approximation
    from copul.checkerboard.checkerboarder import Checkerboarder
    print("\nConverting to Checkerboard Copula...")
    ccop = cop.to_checkerboard()
    print("Successfully created checkerboard copula object.")
    rho_val = ccop.rho()
    footrule = ccop.footrule()
    print(f"Checkerboard Copula: rho = {rho_val}, footrule = {footrule}")