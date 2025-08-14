import sympy as sp
import numpy as np
from functools import lru_cache

from copul.families.core.biv_copula import BivCopula
from copul.wrapper.cdf_wrapper import CDFWrapper  # noqa: F401 – kept for users
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper  # noqa: F401


# ----------------------------------------------------------------------
# Helper – Cardano in SymPy (real root of z³+p z + k = 0)
# ----------------------------------------------------------------------
def _cardano_real_root(p, k):
    """
    Return the unique real root of  z³ + p z + k = 0  (Δ < 0 here).
    Works for symbolic ‘p’, ‘k’ (SymPy).
    """
    Δ = (k / 2) ** 2 + (p / 3) ** 3

    def cbrt(z):
        return sp.real_root(z, 3)  # SymPy’s real cube-root

    return cbrt(-k / 2 + sp.sqrt(Δ)) + cbrt(-k / 2 - sp.sqrt(Δ))


class OptimalAffineJumpCopula(BivCopula):
    # ------------------------------------------------------------------
    # 1.  Class-level symbols
    # ------------------------------------------------------------------
    a, c = sp.symbols("a c", real=True)
    params = [a, c]
    intervals = {
        "a": sp.Interval.Lopen(-sp.Rational(1, 2), 0),
        "c": sp.Interval(-sp.Rational(1, 2), 1),
    }
    u, v = sp.symbols("u v", positive=True)

    # ------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=None)
    def _d_numeric(a: float, c: float) -> float:
        """
        Fast numeric evaluation of d⋆ for (a,c) ∈ (−½,0)×[−½,1].
        Chooses the *smallest* positive real root of
            4 q² S³ − 15 q S² + 20(1−c) = 0 ,   S = 1 + d  > 0.
        """
        if not (-0.5 < a < 0.0):
            raise ValueError("a must lie in (−0.5,0)")
        if not (-0.5 <= c <= 1.0):
            raise ValueError("c must lie in [−0.5,1]")

        if c == 1.0:  # wedge maximum ⇒ boundary d = −1
            return -1.0

        q = -a
        coeffs = [4 * q * q, -15 * q, 0.0, 20.0 * (1.0 - c)]  # cubic in S
        roots = np.roots(coeffs)
        # keep only real & positive roots
        S_pos = np.real(roots[np.isreal(roots) & (roots.real > 0)])
        if S_pos.size == 0:
            raise RuntimeError("No positive real root found for S")
        S = S_pos.min()  # the correct one
        return float(S - 1.0)  # d⋆ = S − 1

    # ------------------------------------------------------------------
    @staticmethod
    def _d_expr(a, c):
        """
        Symbolic closed-form for d⋆ using Cardano.
        Falls back on the numeric routine when a or c are non-symbolic
        (keeps performance acceptable).
        """
        if a.is_number and c.is_number:
            return OptimalAffineJumpCopula._d_numeric(float(a), float(c))

        q = -a
        if sp.Eq(c, 1):
            return -1

        # Cardano for   4 q² S³ − 15 q S² + 20(1−c) = 0
        α = 15 / (4 * q)
        β = 5 * (1 - c) / q**2
        p = -(α**2) / 3
        k = β - 2 * α**3 / 27
        z = _cardano_real_root(p, k)
        S = α / 3 + z
        return S - 1

    # ------------------------------------------------------------------
    # 3.  Symbolic CDF  (re-written to include B-0 / B-1 split)
    # ------------------------------------------------------------------
    @property
    def _cdf_expr(self):
        a_s, c_s, u_s, v_s = self.a, self.c, self.u, self.v
        q_s = -a_s
        d_s = self._d_expr(a_s, c_s)
        b_s = 1 / q_s

        # break-points in v
        d_crit = 1 / q_s - 1
        v1 = q_s * (1 + d_s) / 2
        v2 = 1 - v1
        v0 = 1 - 1 / (2 * q_s * (1 + d_s))

        s_v = sp.Piecewise(
            # ---------- mixed‑ramp geometry (d ≤ dcrit) ---------------
            (sp.sqrt(2 * q_s * (1 + d_s) * v_s), sp.Le(d_s, d_crit) & sp.Le(v_s, v1)),
            (v_s + q_s * (1 + d_s) / 2, sp.Le(d_s, d_crit) & sp.Le(v_s, v2)),
            (
                1 + q_s * (1 + d_s) - sp.sqrt(2 * q_s * (1 + d_s) * (1 - v_s)),
                sp.Le(d_s, d_crit),
            ),  # residual v
            # ---------- fully truncated geometry (d > dcrit) ----------
            (
                sp.Rational(1, 2) + q_s * (1 + d_s) * v_s,
                sp.Gt(d_s, d_crit) & sp.Le(v_s, v0),
            ),  # B‑0
            (
                1 + q_s * (1 + d_s) - sp.sqrt(2 * q_s * (1 + d_s) * (1 - v_s)),
                True,
            ),  # B‑1
        )

        # break-points in t for the four pieces of h*
        t1 = sp.Max(0, s_v - q_s * (1 + d_s))  # inner plateau length
        t0i = s_v - q_s * d_s  # end of inner ramp
        t1o = s_v - q_s  # start of outer ramp
        t0o = s_v  # end of outer ramp

        def g1(t):
            return b_s * (s_v * t - t**2 / 2)

        def g2(t):
            return g1(t) - d_s * t

        # ---- integrate the four rectangles/triangles ------------------
        inner_plateau = sp.Max(0, sp.Min(u_s, v_s, t1))

        lo_in = sp.Max(0, t1)
        hi_in = sp.Min(u_s, v_s, t0i)
        inner_ramp = sp.Piecewise((g2(hi_in) - g2(lo_in), hi_in > lo_in), (0, True))

        outer_plateau = sp.Max(0, sp.Min(u_s, t1o) - v_s)

        lo_out = sp.Max(v_s, t1o)
        hi_out = sp.Min(u_s, t0o)
        outer_ramp = sp.Piecewise((g1(hi_out) - g1(lo_out), hi_out > lo_out), (0, True))

        return inner_plateau + inner_ramp + outer_plateau + outer_ramp

    # ------------------------------------------------------------------
    # 5.  Vectorised CDF  (updated to match the symbolic logic)
    # ------------------------------------------------------------------
    def cdf_vectorized(self, u, v):
        a, c = self.a, self.c
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        d = self._d_numeric(a, c)
        q = -a
        b = 1.0 / q

        v1 = q * (1 + d) / 2.0
        v2 = 1.0 - v1
        v0 = 1.0 - 1.0 / (2.0 * q * (1.0 + d))

        # --- s_v --------------------------------------------------------
        d_crit = 1.0 / q - 1.0
        if d <= d_crit:  # mixed‑ramp
            sv = np.where(
                v <= v1,
                np.sqrt(2 * q * (1 + d) * v),  # A‑1
                np.where(
                    v <= v2,
                    v + q * (1 + d) / 2.0,  # A‑2
                    1.0 + q * (1 + d) - np.sqrt(2 * q * (1 + d) * (1 - v)),  # A‑3
                ),
            )
        else:  # fully truncated
            v0 = 1.0 - 1.0 / (2 * q * (1 + d))
            sv = np.where(
                v <= v0,
                0.5 + q * (1 + d) * v,  # B‑0
                1.0 + q * (1 + d) - np.sqrt(2 * q * (1 + d) * (1 - v)),  # B‑1
            )

        # --- t–breaks ---------------------------------------------------
        t1 = np.maximum(0.0, sv - q * (1 + d))
        t0i = sv - q * d
        t1o = sv - q
        t0o = sv

        def integrate_ramp(lo, hi, s, offset):
            return b * (s * (hi - lo) - 0.5 * (hi**2 - lo**2)) - offset * (hi - lo)

        inner_plateau = np.maximum(0.0, np.minimum.reduce([u, v, t1]))

        lo_in = np.maximum(0.0, t1)
        hi_in = np.minimum.reduce([u, v, t0i])
        inner_ramp = np.where(hi_in > lo_in, integrate_ramp(lo_in, hi_in, sv, d), 0.0)

        outer_plateau = np.maximum(0.0, np.minimum(u, t1o) - v)

        lo_out = np.maximum(v, t1o)
        hi_out = np.minimum(u, t0o)
        outer_ramp = np.where(
            hi_out > lo_out, integrate_ramp(lo_out, hi_out, sv, 0.0), 0.0
        )

        return inner_plateau + inner_ramp + outer_plateau + outer_ramp

    # ------------------------------------------------------------------
    # 6.  Convenience / diagnostics  (unchanged)
    # ------------------------------------------------------------------
    @property
    def _pdf_expr(self):
        return self._cdf_expr.diff(self.u).diff(self.v)

    def footrule(self, numeric=False, grid=401):
        if not numeric:
            raise NotImplementedError("Symbolic footrule is too heavy.")
        u_lin = np.linspace(0.5 / grid, 1 - 0.5 / grid, grid)
        v_lin = u_lin.copy()
        uu, vv = np.meshgrid(u_lin, v_lin)
        Cvals = self.cdf_vectorized(uu, vv)
        return float(12 * np.mean(np.abs(Cvals - uu * vv)))

    def footrule_numeric(cop, grid=2001):
        """ψ(C)=6∫₀¹ C(u,u)du − 2"""
        u = np.linspace(0.0, 1.0, grid, dtype=float)
        cuu = cop.cdf_vectorized(u, u)  # C(u,u)
        return float(6.0 * np.trapz(cuu, u) - 2.0)

    @property
    def is_absolutely_continuous(self):
        return True

    @property
    def is_symmetric(self):
        return False


# ----------------------------------------------------------------------
# Tiny self-check
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     a_param, c_param = -0.02, 0.50
#     print(f"Testing  a={a_param}, c={c_param}")

#     d_star = OptimalAffineJumpCopula._d_numeric(a_param, c_param)
#     print("numeric  d* =", d_star)

#     cop = OptimalAffineJumpCopula(a=a_param, c=c_param)
#     print("CDF(0.5,0.6) =", cop.cdf_vectorized(0.5, 0.6))

#     print("Numeric footrule ≈", cop.footrule_numeric())
#     ccop = cop.to_checkerboard()
#     ccop_footrule = ccop.footrule()
#     ccop_rho = ccop.rho()
#     print("Checkerboard footrule ≈", ccop_footrule)
#     print("Checkerboard rho =", ccop_rho)

if __name__ == "__main__":
    # parameters that previously failed hard
    a_values = [-0.02, -0.05, -0.1, -0.49]
    c_values = [0.80, 0.50, 0.20, -0.1, -0.4, -0.5]
    for a in a_values:
        for c in c_values:
            # create copula instance
            cop = OptimalAffineJumpCopula(a=a, c=c)
            # sanity: ψ(C)=c   (grid version of (†))
            psi_num = cop.footrule_numeric()
            print(f"(a={a:+.3f}, c={c:+.3f})  ψ ≈ {psi_num:+.12f}")
            # should be |ψ-c| < 2e‑3 already with grid = 2001
