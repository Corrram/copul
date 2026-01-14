from typing import Tuple, TypeAlias

import numpy as np
import sympy as sp
from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


class NumericWrapper:
    """
    A simple wrapper for numeric functions to make them compatible
    with the plotting utilities that expect a callable object.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, u, v):
        # Delegate directly to the numeric function
        return self.func(u, v)


class XiPsiApproxLowerBoundaryCopula(BivCopula):
    r"""
    Two-parameter "diagonal strip" copula with a rectangular hole.

    The copula is defined by a transformation variable T and a hole geometry H.
    C(u,v) = (u*t - Area(Intersection)) / (1 - beta)
    where t = F_T^{-1}(v).

    Parameters
    ----------
    alpha : float
        Controls the horizontal start of the diagonal slope. alpha in [0, 0.5).
    beta : float
        Controls the vertical thickness of the hole. beta in (0, 1).
    """

    # Parameters and domains
    alpha, beta = sp.symbols("alpha beta", real=True)
    params = [alpha, beta]
    intervals = {
        # Update: include 1/2 in the domain (Lopen = Left-open, Right-closed ideally,
        # or simply use a closed bound if supported by the framework)
        "alpha": sp.Interval(0, sp.Rational(1, 2), left_open=True, right_open=False),
        "beta": sp.Interval.open(0, 1),
    }
    special_cases = {0: BivIndependenceCopula}

    # Convenience symbols
    u, v = sp.symbols("u v", nonnegative=True)

    def __init__(self, *args, **kwargs):
        if args:
            if len(args) == 1:
                kwargs["alpha"] = args[0]
            elif len(args) == 2:
                kwargs["alpha"], kwargs["beta"] = args
            else:
                raise ValueError("Provide at most two positional args: alpha, beta.")
        super().__init__(**kwargs)

    def _get_constants(self) -> Tuple[float, float, float, float, float, float, float]:
        """Computes structural constants A, K, C, u1, k."""
        a = float(self.alpha)
        b = float(self.beta)
        denom = 1.0 - b
        denom2 = denom * denom

        # Determine if we are in the limiting case alpha = 0.5
        is_limit = np.isclose(a, 0.5)

        A = (1.0 - a) / denom

        if is_limit:
            # Limits as alpha -> 0.5
            K = 0.0
            # C_val = 1 / (1-b)
            C_val = 1.0 / denom
            # k_slope is infinite (vertical jump), handled via logic
            k_slope = np.inf
        else:
            K = (1.0 - 2.0 * a) / denom2
            C_val = (1.0 - 2.0 * b + 2.0 * a * b) / denom2
            k_slope = denom / (1.0 - 2.0 * a)

        # Threshold CDF value at v=beta
        # If K=0, this simplifies to A*b
        u1 = A * b - 0.5 * K * (b**2)

        return a, b, A, K, C_val, u1, k_slope

    @staticmethod
    def _psi_vec(s, alpha, beta, k_slope):
        """Vectorized ψ(s). Handles step function at alpha=0.5."""
        s = np.asarray(s, dtype=float)
        res = np.zeros_like(s)

        # Case alpha = 0.5: Step function
        if np.isclose(alpha, 0.5):
            # psi(s) = 1 - beta if s > 0.5 else 0
            # We use 0.5 as the threshold
            mask_up = s > 0.5
            if np.any(mask_up):
                res[mask_up] = 1.0 - beta
            return res

        # Standard Case alpha < 0.5
        # Region 2 (middle)
        mask_mid = (s > alpha) & (s < 1.0 - alpha)
        if np.any(mask_mid):
            res[mask_mid] = k_slope * (s[mask_mid] - alpha)

        # Region 3 (upper)
        mask_up = s >= 1.0 - alpha
        if np.any(mask_up):
            res[mask_up] = 1.0 - beta

        return res

    def _quantile_t(self, v):
        """Computes t = F_T^{-1}(v). Handles K=0 (Linear segments)."""
        v = np.asarray(v, dtype=float)
        a, b, A, K, C_val, u1, _ = self._get_constants()
        u2 = 1.0 - u1
        t_vals = np.zeros_like(v)

        # Check for linear limit (alpha = 0.5 implies K = 0)
        is_linear = np.isclose(K, 0.0)

        # 1. Lower Region (v < u1)
        mask1 = v < u1
        if np.any(mask1):
            if is_linear:
                # F(t) = A*t => t = v/A
                t_vals[mask1] = v[mask1] / A
            else:
                disc = A**2 - 2.0 * K * v[mask1]
                t_vals[mask1] = (A - np.sqrt(np.maximum(0, disc))) / K

        # 2. Middle Region (u1 <= v <= u2)
        # This region is always linear with slope 1/C_val
        mask2 = (v >= u1) & (v <= u2)
        if np.any(mask2):
            t_vals[mask2] = b + (v[mask2] - u1) / C_val

        # 3. Upper Region (v > u2)
        mask3 = v > u2
        if np.any(mask3):
            if is_linear:
                # Symmetric to lower: t = 1 - (1-v)/A
                t_vals[mask3] = 1.0 - (1.0 - v[mask3]) / A
            else:
                disc = A**2 - 2.0 * K * (1.0 - v[mask3])
                t_vals[mask3] = 1.0 - (A - np.sqrt(np.maximum(0, disc))) / K

        return np.clip(t_vals, 0.0, 1.0)

    def _calc_H(self, u, t):
        """
        Calculates H(u, t).
        Handles the split cases for alpha=0.5 (two disjoint rectangles) vs alpha<0.5 (connected ramp).
        """
        a, b, _, _, _, _, k = self._get_constants()

        u = np.asarray(u, dtype=float)
        t = np.asarray(t, dtype=float)

        if u.shape != t.shape:
            u, t = np.broadcast_arrays(u, t)

        total_area = np.zeros_like(u)

        # --- Case: alpha = 0.5 (Disjoint Rectangles) ---
        if np.isclose(a, 0.5):
            # Hole 1: [0, 0.5] x [0, beta]
            len1 = np.clip(u, 0, 0.5)
            total_area += len1 * np.minimum(t, b)

            # Hole 2: (0.5, 1.0] x [1-beta, 1.0]
            # Contribution only if u > 0.5
            len2 = np.maximum(0, u - 0.5)
            height2 = np.maximum(0, t - (1.0 - b))
            total_area += len2 * height2

            return total_area

        # --- Case: alpha < 0.5 (Ramp Connection) ---

        # Region 1: [0, alpha] -> Rect height min(t, beta)
        len1 = np.clip(u, 0, a)
        total_area += len1 * np.minimum(t, b)

        # Region 2: [alpha, 1-alpha] -> Trapezoid/Triangle logic
        U_eff = np.clip(u - a, 0, 1.0 - 2.0 * a)
        mask_r2 = U_eff > 0
        if np.any(mask_r2):
            Ue = U_eff[mask_r2]
            tv = t[mask_r2]

            inv_k = 1.0 / k
            x0 = tv * inv_k
            xb = (tv - b) * inv_k
            area_r2 = np.zeros_like(Ue)

            # Sub-case A: t <= beta
            mask_no_sat = tv <= b
            if np.any(mask_no_sat):
                u_ns = Ue[mask_no_sat]
                t_ns = tv[mask_no_sat]
                x0_ns = x0[mask_no_sat]
                limit = np.minimum(u_ns, np.maximum(0, x0_ns))
                area_ns = t_ns * limit - 0.5 * k * (limit**2)
                area_r2[mask_no_sat] = area_ns

            # Sub-case B: t > beta
            mask_sat = tv > b
            if np.any(mask_sat):
                u_s = Ue[mask_sat]
                t_s = tv[mask_sat]
                xb_s = xb[mask_sat]
                x0_s = x0[mask_sat]

                # 1. Saturation part (beta * x)
                sat_lim = np.minimum(u_s, xb_s)
                area_s = b * sat_lim

                # 2. Ramp part
                ramp_start = xb_s
                ramp_end = np.minimum(u_s, x0_s)
                valid_ramp = ramp_end > ramp_start
                if np.any(valid_ramp):
                    rs = ramp_start[valid_ramp]
                    re = ramp_end[valid_ramp]
                    ts = t_s[valid_ramp]
                    val_end = ts * re - 0.5 * k * (re**2)
                    val_start = ts * rs - 0.5 * k * (rs**2)
                    area_s[valid_ramp] += val_end - val_start

                area_r2[mask_sat] = area_s

            total_area[mask_r2] += area_r2

        # Region 3: [1-alpha, 1] -> Rect height max(0, t - (1-beta))
        len3 = np.maximum(0, u - (1.0 - a))
        height3 = np.maximum(0, t - (1.0 - b))
        total_area += len3 * height3

        return total_area

    def _calc_hole_width_at_t(self, u, t):
        """
        Calculates partial derivative of H w.r.t t (dh/dt).
        This represents the horizontal width of the hole at height t,
        intersected with [0, u].
        """
        a, b, _, _, _, _, k = self._get_constants()
        width = np.zeros_like(u)

        # --- Case: alpha = 0.5 ---
        if np.isclose(a, 0.5):
            # Hole 1 width: t in (0, beta) => width is min(u, 0.5)
            in_reg1 = (t > 0) & (t < b)
            width += np.where(in_reg1, np.clip(u, 0, 0.5), 0.0)

            # Hole 2 width: t in (1-beta, 1) => width is max(0, u - 0.5)
            in_reg2 = (t > 1.0 - b) & (t < 1.0)
            width += np.where(in_reg2, np.maximum(0, u - 0.5), 0.0)

            return width

        # --- Case: alpha < 0.5 ---
        # Region 1 check (Lower hole)
        in_reg1 = (t > 0) & (t < b)
        width += np.where(in_reg1, np.clip(u, 0, a), 0.0)

        # Region 2 check (Ramp)
        inv_k = 1.0 / k
        s_min = a + (t - b) * inv_k
        s_max = a + t * inv_k
        s_start = np.clip(s_min, a, 1.0 - a)
        s_end = np.clip(s_max, a, 1.0 - a)
        w_start = np.minimum(s_start, u)
        w_end = np.minimum(s_end, u)
        width += np.maximum(0, w_end - w_start)

        # Region 3 check (Upper hole)
        in_reg3 = (t > 1.0 - b) & (t < 1.0)
        len3 = np.maximum(0, u - (1.0 - a))
        width += np.where(in_reg3, len3, 0.0)

        return width

    def _density_t(self, t):
        """Computes f_T(t)."""
        t = np.asarray(t, dtype=float)
        a, b, A, K, C_val, _, _ = self._get_constants()
        res = np.zeros_like(t)

        m1 = t < b
        res[m1] = A - K * t[m1]
        m2 = (t >= b) & (t <= 1.0 - b)
        res[m2] = C_val
        m3 = t > 1.0 - b
        res[m3] = A - K * (1.0 - t[m3])
        return res

    def _calc_h_u(self, u, t):
        """h(u, t) = min(beta, max(0, t - psi(u)))."""
        a, b, _, _, _, _, k = self._get_constants()
        psi_val = self._psi_vec(u, a, b, k)
        return np.minimum(b, np.maximum(0, t - psi_val))

    def cdf(self, u=None, v=None):
        """
        Cumulative distribution function C(u,v).
        """
        # Return the NotImplemented error if accessed as property/no-args
        # (per user constraint on _cdf_expr)
        if u is None and v is None:
            return self._cdf_expr

        return self.cdf_vectorized(u, v)

    def pdf(self, u=None, v=None):
        """
        Probability Density Function c(u,v).
        """
        # If no args, return a wrapper that can be called by plotting tools
        if u is None and v is None:
            return NumericWrapper(self.pdf_vectorized)

        return self.pdf_vectorized(u, v)

    def cond_distr_1(self, u=None, v=None):
        """P(V <= v | U = u) = dC/du."""
        if u is None and v is None:
            return NumericWrapper(self.cond_distr_1)

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        _, b, _, _, _, _, _ = self._get_constants()

        t_vals = self._quantile_t(v)
        h_u_vals = self._calc_h_u(u, t_vals)

        res = (t_vals - h_u_vals) / (1.0 - b)
        return np.clip(res, 0.0, 1.0)

    def cond_distr_2(self, u=None, v=None):
        """P(U <= u | V = v) = dC/dv."""
        if u is None and v is None:
            return NumericWrapper(self.cond_distr_2)

        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        _, b, _, _, _, _, _ = self._get_constants()

        t_vals = self._quantile_t(v)
        ft_vals = self._density_t(t_vals)
        dh_dt = self._calc_hole_width_at_t(u, t_vals)

        # Guard small density
        safe_ft = np.where(ft_vals < 1e-12, 1e-12, ft_vals)

        term1 = (u - dh_dt) / (1.0 - b)
        res = term1 * (1.0 / safe_ft)

        return np.clip(res, 0.0, 1.0)

    def cdf_vectorized(self, u, v):
        """Exact numeric Copula CDF."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        _, b, _, _, _, _, _ = self._get_constants()

        t_vals = self._quantile_t(v)
        h_val = self._calc_H(u, t_vals)

        u_cl = np.clip(u, 0, 1)
        t_cl = np.clip(t_vals, 0, 1)

        res = (u_cl * t_cl - h_val) / (1.0 - b)
        return np.clip(res, 0.0, 1.0)

    def pdf_vectorized(self, u, v):
        """Exact numeric Copula PDF."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        a, b, _, _, _, _, k = self._get_constants()

        t_vals = self._quantile_t(v)
        ft_vals = self._density_t(t_vals)
        psi_u = self._psi_vec(u, a, b, k)

        in_hole = (t_vals > psi_u) & (t_vals < psi_u + b)

        res = np.zeros_like(t_vals)
        scale = 1.0 / (1.0 - b)

        mask_valid = ~in_hole
        if np.any(mask_valid):
            safe_ft = np.where(ft_vals[mask_valid] < 1e-12, 1e-12, ft_vals[mask_valid])
            res[mask_valid] = scale * (1.0 / safe_ft)

        return res

    @property
    def _cdf_expr(self):
        """
        Symbolic expression not available.
        """
        raise NotImplementedError(
            "The analytical CDF for this density is implemented in cdf_vectorized."
        )

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return False


# Type alias
DiagonalStripCopula: TypeAlias = XiPsiApproxLowerBoundaryCopula

if __name__ == "__main__":
    # quick smoke test & plots
    pairs = [(0.2, 0.30), (0.3, 0.50), (0.40, 0.50)]
    for a, b in pairs:
        cop = XiPsiApproxLowerBoundaryCopula(alpha=a, beta=b)
        cop.plot_pdf(plot_type="contour", levels=999, grid_size=100)
        print(cop.cdf(0.4, 0.6))
        print(cop.cond_distr_1(0.4, 0.6))
        print(cop.cond_distr_2(0.4, 0.6))
        print(cop.pdf(0.4, 0.6))
