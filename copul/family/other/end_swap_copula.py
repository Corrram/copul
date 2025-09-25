import sympy as sp
import numpy as np

from copul.exceptions import PropertyUnavailableException
from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper


class EndSwapCopula(BivCopula):
    r"""
    End–swap copula family tracing the **lower boundary** of the
    :math:`(\psi,\nu)` region (Spearman's footrule :math:`\psi` vs Blest's :math:`\nu`).

    Construction:
      - Let :math:`d \in [0,1/2]`. Define the measure-preserving "end swap"
        :math:`S_d` that mirrors the end blocks and keeps the middle fixed:
        \[
          S_d(t) = \begin{cases}
             1 - t, & t \in [0,d]\ \cup\ [1-d,1],\\
             t, & t \in (d, 1-d).
          \end{cases}
        \]
      - The extreme-point conditional is
        :math:`h(u,v)=\mathbb{1}\{S_d(u)\le v\}`, and the copula is
        :math:`C_d(u,v) = \int_0^u h(t,v)\,dt`.

    Correct section sets for fixed :math:`v`:
      \[
        h(u,v)=1 \iff
        \begin{cases}
          u \le v, & u \in (d,\,1-d),\\
          u \ge 1-v, & u \in [0,d]\cup[1-d,1].
        \end{cases}
      \]
      Hence
      \[
      C_d(u,v) = \lambda\big((d,\min\{u,\,\min(v,1-d)\}] \big)
      + \lambda\big([ \max(0,1-v), \min(u,d)]\big)
      + \lambda\big([\max(1-d,1-v), \min(u,1)]\big).
      \]

    Closed-form boundary (lower curve):
    .. math::
        \psi(d) \;=\; 1 - 6(d - d^2),\qquad
        \nu(d) \;=\; 1 - 12d + 24d^2 - 16d^3,\qquad d\in[0,1/2].

    Inversion (from a given footrule :math:`\psi\in[-1/2,1]`):
    .. math::
        d \;=\; \frac{1 - \sqrt{(1+2\psi)/3}}{2} \in [0,1/2].

    Special cases:
      - :math:`d=0`: :math:`C_d = M` (upper Fréchet), :math:`(\psi,\nu)=(1,1)`.
      - :math:`d=1/2`: :math:`C_d = W` (lower Fréchet), :math:`(\psi,\nu)=(-1/2,-1)`.
    """

    # SymPy symbols & meta
    d = sp.symbols("d", real=True)
    params = [d]
    intervals = {
        "d": sp.Interval(0, sp.Rational(1, 2), left_open=False, right_open=False)
    }
    special_cases = {0: UpperFrechet, sp.Rational(1, 2): LowerFrechet}

    u, v = sp.symbols("u v", real=True)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["d"] = args[0]
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Rank measures (closed form)
    # ------------------------------------------------------------------

    def spearmans_footrule(self):
        """Spearman's footrule: ψ(d) = 1 - 6(d - d^2) on [-1/2, 1]."""
        d = float(self.d)
        return 1.0 - 6.0 * (d - d * d)

    def blests_nu(self):
        """Blest's ν: ν(d) = 1 - 12 d + 24 d^2 - 16 d^3."""
        d = float(self.d)
        return 1.0 - 12.0 * d + 24.0 * d * d - 16.0 * d**3

    @classmethod
    def from_psi(cls, psi_target: float):
        """
        Construct the boundary copula at a given footrule ψ ∈ [-1/2, 1].

        Inversion: d = (1 - sqrt((1 + 2 ψ)/3))/2, clamped to [0, 1/2].
        """
        if not (-0.5 <= psi_target <= 1.0):
            raise ValueError("psi_target must be in [-1/2, 1].")
        val = (1.0 + 2.0 * float(psi_target)) / 3.0
        val = max(0.0, min(1.0, val))
        d = (1.0 - np.sqrt(val)) / 2.0
        d = max(0.0, min(0.5, d))
        return cls(d=d)

    # ------------------------------------------------------------------
    # Symbolic CDF & conditional distributions
    # ------------------------------------------------------------------

    @property
    def cdf(self):
        r"""
        Symbolic CDF based on exact lengths of the three contributing blocks:
          C(u,v) = L_left + L_mid + L_right where
            L_left  = max(0, min(u, d)     - max(0, 1 - v)),
            L_mid   = max(0, min(u, m)     - d) with m := min(v, 1 - d),
            L_right = max(0, min(u, 1)     - max(1 - d, 1 - v)).
        """
        u, v, d = self.u, self.v, self.d

        m = sp.Min(v, 1 - d)

        L_left = sp.Max(0, sp.Min(u, d) - sp.Max(0, 1 - v))
        L_mid = sp.Max(0, sp.Min(u, m) - d)
        L_right = sp.Max(0, sp.Min(u, 1) - sp.Max(1 - d, 1 - v))

        expr = sp.simplify(L_left + L_mid + L_right)
        return CDFWrapper(expr)

    def cond_distr_1(self, u=None, v=None):
        r"""
        Conditional cdf in u: h(u,v) = ∂₁C(u,v) = 1{S_d(u) ≤ v}.

        Implemented directly from the definition:
          - If u ∈ (d, 1-d):  h(u,v) = 1{u ≤ v}.
          - If u ∈ [0,d] ∪ [1-d,1]: h(u,v) = 1{u ≥ 1 - v}.
        """
        d = self.d
        u_sym, v_sym = self.u, self.v

        def H(x):
            # Heaviside with H(0)=0 to avoid measure-zero ambiguities
            return sp.Heaviside(x, 0)

        in_mid = H(u_sym - d) * H((1 - d) - u_sym)  # 1{d < u < 1-d}
        in_ends = 1 - in_mid

        h_mid = in_mid * H(v_sym - u_sym)  # 1{u<=v}
        h_ends = in_ends * H(u_sym - (1 - v_sym))  # 1{u >= 1-v}

        expr = sp.simplify(h_mid + h_ends)
        return CD2Wrapper(expr)(u, v)

    # (Optional) ∂₂C(u,v) omitted: the exact a.e. expression is piecewise but
    # quite involved; numerics/plotting can use finite differences below.

    # ------------------------------------------------------------------
    # Vectorized numerics
    # ------------------------------------------------------------------

    def cdf_vectorized(self, u, v):
        """
        Vectorized C(u,v) using the length formulas. Shapes of u and v are broadcast.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d = float(self.d)

        u_b, v_b = np.broadcast_arrays(u, v)

        # Helpers: max(0, x) and min for arrays
        def pos(x):
            return np.maximum(0.0, x)

        m = np.minimum(v_b, 1.0 - d)

        L_left = pos(np.minimum(u_b, d) - np.maximum(0.0, 1.0 - v_b))
        L_mid = pos(np.minimum(u_b, m) - d)
        L_right = pos(np.minimum(u_b, 1.0) - np.maximum(1.0 - d, 1.0 - v_b))

        return L_left + L_mid + L_right

    def pdf_vectorized(self, u, v):
        """
        Vectorized "density" c(u,v) = ∂²C/∂u∂v via finite difference in v.
        (The family has singular components; this is for plotting/numerics.)
        """
        u, v = np.atleast_1d(u), np.atleast_1d(v)
        pdf_vals = np.zeros_like(u, dtype=float)
        eps = 1e-7
        for idx in np.ndindex(u.shape):
            u_i, v_i = u[idx], v[idx]
            C_plus = self.cdf_vectorized(u_i, min(1.0, v_i + eps))
            C = self.cdf_vectorized(u_i, v_i)
            pdf_vals[idx] = (C_plus - C) / eps
        return pdf_vals.reshape(np.asarray(u).shape)

    @property
    def pdf(self):
        raise PropertyUnavailableException(
            "This copula has singular components; no purely absolutely-continuous PDF."
        )


if __name__ == "__main__":
    # Quick smoke test
    for d in [0.1, 0.25, 0.4]:
        cop = EndSwapCopula(d=d)
        print(
            f"d={d:.3f} -> psi={cop.spearmans_footrule():.6f}, nu={cop.blests_nu():.6f}"
        )
        cop.plot_cond_distr_1(plot_type="contour", grid_size=500)
