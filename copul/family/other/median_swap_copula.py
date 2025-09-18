import sympy as sp
import numpy as np

from copul.exceptions import PropertyUnavailableException
from copul.family.core.biv_copula import BivCopula
from copul.family.other.upper_frechet import UpperFrechet
from copul.family.other.lower_frechet import LowerFrechet
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper


class MedianSwapCopula(BivCopula):
    r"""
    Chevron / median–swap copula family tracing the **upper boundary** of the
    :math:`(\beta,\nu)` region (Blomqvist's :math:`\beta` vs Blest's :math:`\nu`).

    Construction:
      - Let :math:`\delta \in [0,1/2]`. Define the measure-preserving "median swap"
        :math:`T_\delta` that swaps two adjacent blocks of length :math:`\delta`
        around :math:`t=1/2`.
      - The extreme-point conditional is
        :math:`h(u,v)=\mathbb{1}\{T_\delta(u)\le v\}`, and the copula is
        :math:`C_\delta(u,v) = \int_0^u h(t,v)\,dt`.

    Correct (piecewise) sections in :math:`u` for fixed :math:`v`:

      Let :math:`\tfrac12-\delta<\tfrac12<\tfrac12+\delta`. Define
      :math:`A(v)=\{\,t\in[0,1]:T_\delta(t)\le v\,\}`. Then

      * If :math:`0\le v\le\tfrac12-\delta`:  :math:`A(v)=[0,v]`.
      * If :math:`\tfrac12-\delta< v\le\tfrac12`:
        :math:`A(v)=[0,\tfrac12-\delta]\cup(\tfrac12,\,v+\delta]`.
      * If :math:`\tfrac12< v\le\tfrac12+\delta`:
        :math:`A(v)=[0,\,v-\delta]\cup(\tfrac12,\,\tfrac12+\delta]`.
      * If :math:`\tfrac12+\delta< v\le 1`: :math:`A(v)=[0,v]`.

      Hence :math:`C_\delta(u,v)=\lambda(A(v)\cap[0,u])`, which yields a valid copula
      with uniform margins.

    Closed-form boundary (upper curve):
    .. math::
        \beta(\delta) \;=\; 1-4\delta,\qquad
        \nu(\delta) \;=\; 1-6\delta^2-8\delta^4,\qquad \delta\in[0,1/2].

    Special cases:
      - :math:`\delta=0`: :math:`C_\delta = M` (upper Fréchet), :math:`(\beta,\nu)=(1,1)`.
      - :math:`\delta=1/2`: :math:`C_\delta = W` (lower Fréchet), :math:`(\beta,\nu)=(-1,-1)`.
    """

    # SymPy symbols & meta
    delta = sp.symbols("delta", real=True)
    params = [delta]
    intervals = {
        "delta": sp.Interval(0, sp.Rational(1, 2), left_open=False, right_open=False)
    }
    special_cases = {
        0: UpperFrechet,
        sp.Rational(1, 2): LowerFrechet,
    }

    u, v = sp.symbols("u v", real=True)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["delta"] = args[0]
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Rank measures (closed form)
    # ------------------------------------------------------------------

    def blomqvists_beta(self):
        """Blomqvist's beta: β(δ) = 1 - 4 δ."""
        d = float(self.delta)
        return 1.0 - 4.0 * d

    def blests_nu(self):
        """Blest's ν: ν(δ) = 1 - 6 δ^2 - 8 δ^4."""
        d = float(self.delta)
        return 1.0 - 6.0 * d * d - 8.0 * d**4

    @classmethod
    def from_beta(cls, beta_target: float):
        """
        Construct the boundary copula at a given Blomqvist's β ∈ [-1, 1].

        Inversion: δ = (1 - β)/4, clamped to [0, 1/2].
        """
        if not (-1.0 <= beta_target <= 1.0):
            raise ValueError("beta_target must be in [-1, 1].")
        delta = (1.0 - float(beta_target)) / 4.0
        delta = max(0.0, min(0.5, delta))
        return cls(delta=delta)

    # ------------------------------------------------------------------
    # Symbolic CDF & conditional distributions (Frechet-style wrappers)
    # ------------------------------------------------------------------

    @property
    def cdf(self):
        r"""
        Correct symbolic CDF from the definition
        :math:`C(u,v)=\int_0^u \mathbf 1\{T_\delta(t)\le v\}\,dt`,
        expressed piecewise in :math:`v`.
        """
        u, v, d = self.u, self.v, self.delta
        half = sp.Rational(1, 2)

        # Case A: v <= 1/2 - d   -> C(u,v) = min(u, v)
        expr_caseA = sp.Min(u, v)

        # Case B: 1/2 - d < v <= 1/2
        #   C(u,v) =
        #     u,                 0 <= u <= 1/2 - d
        #     1/2 - d,           1/2 - d < u <= 1/2
        #     u - d,             1/2 < u <= v + d
        #     v,                 v + d < u <= 1
        expr_caseB = sp.Piecewise(
            (u, u <= half - d),
            (half - d, u <= half),
            (u - d, u <= v + d),
            (v, True),
        )

        # Case C: 1/2 < v <= 1/2 + d
        #   C(u,v) =
        #     u,                         0 <= u <= v - d
        #     v - d,                     v - d < u <= 1/2
        #     u + v - d - 1/2,           1/2 < u <= 1/2 + d
        #     v,                         1/2 + d < u <= 1
        expr_caseC = sp.Piecewise(
            (u, u <= v - d),
            (v - d, u <= half),
            (u + v - d - half, u <= half + d),
            (v, True),
        )

        # Case D: v >= 1/2 + d   -> C(u,v) = min(u, v)
        expr_caseD = sp.Min(u, v)

        expr = sp.Piecewise(
            (expr_caseA, v <= half - d),
            (expr_caseB, v <= half),
            (expr_caseC, v <= half + d),
            (expr_caseD, True),
        )
        return CDFWrapper(expr)

    def cond_distr_1(self, u=None, v=None):
        r"""
        Correct :math:`h(u,v)=\partial_1 C(u,v)=\mathbf 1\{T_\delta(u)\le v\}`.
        Implemented piecewise in :math:`v` using Heaviside indicators.
        """
        half = sp.Rational(1, 2)
        d = self.delta
        u_sym, v_sym = self.u, self.v

        # Heaviside with H(0)=0 to avoid ambiguous measure-zero overlaps
        def H(x):
            return sp.Heaviside(x, 0)

        # Case A: v <= 1/2 - d  -> h = 1{u <= v}
        hA = H(v_sym - u_sym)

        # Case B: 1/2 - d < v <= 1/2 -> h = 1 on [0,1/2 - d] ∪ (1/2, v + d]
        hB = H((half - d) - u_sym) + H(u_sym - half) * H((v_sym + d) - u_sym)

        # Case C: 1/2 < v <= 1/2 + d -> h = 1 on [0, v - d] ∪ (1/2, 1/2 + d]
        hC = H((v_sym - d) - u_sym) + H(u_sym - half) * H((half + d) - u_sym)

        # Case D: v >= 1/2 + d  -> h = 1{u <= v}
        hD = H(v_sym - u_sym)

        expr = sp.Piecewise(
            (hA, v_sym <= half - d),
            (hB, v_sym <= half),
            (hC, v_sym <= half + d),
            (hD, True),
        )
        return CD2Wrapper(expr)(u, v)

    def cond_distr_2(self, u=None, v=None):
        r"""
        Conditional cdf in v: ∂₂ C(u,v) = F_{U|V=v}(u) for a.e. v.

        From the correct piecewise form of C(u,v):

        - Case A: v ≤ 1/2 − δ,  C(u,v) = min(u, v)
            ⇒ ∂₂ C(u,v) = 1{u > v}  (a.e.)

        - Case B: 1/2 − δ < v ≤ 1/2,  C(u,v) is piecewise and depends on v only
          when u > v + δ
            ⇒ ∂₂ C(u,v) = 1{u > v + δ}

        - Case C: 1/2 < v ≤ 1/2 + δ,  C(u,v) is piecewise and depends on v
          when u > v − δ
            ⇒ ∂₂ C(u,v) = 1{u > v − δ}

        - Case D: v ≥ 1/2 + δ,  C(u,v) = min(u, v)
            ⇒ ∂₂ C(u,v) = 1{u > v}

        We implement these as Heavisides with H(0)=0 (choice on a null set).
        """
        half = sp.Rational(1, 2)
        d = self.delta
        u_sym, v_sym = self.u, self.v

        def H(x):
            return sp.Heaviside(x, 0)  # 1{x >= 0} with H(0)=0

        # Case A and D: ∂₂C = 1{u > v} = H(u - v)
        dA = H(u_sym - v_sym)
        dD = H(u_sym - v_sym)

        # Case B: ∂₂C = 1{u > v + d} = H(u - (v + d))
        dB = H(u_sym - (v_sym + d))

        # Case C: ∂₂C = 1{u > v - d} = H(u - (v - d))
        dC = H(u_sym - (v_sym - d))

        expr = sp.Piecewise(
            (dA, v_sym <= half - d),
            (dB, v_sym <= half),
            (dC, v_sym <= half + d),
            (dD, True),
        )
        return CD2Wrapper(expr)(u, v)

    # ------------------------------------------------------------------
    # Vectorized numerics
    # ------------------------------------------------------------------

    def cdf_vectorized(self, u, v):
        """
        Correct vectorized C(u,v) using the piecewise cases. Shapes of u and v
        are broadcast.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d = float(self.delta)
        half = 0.5

        u_b, v_b = np.broadcast_arrays(u, v)
        C = np.empty_like(u_b, dtype=float)

        # Masks for v
        mA = v_b <= (half - d)
        mB = (v_b > (half - d)) & (v_b <= half)
        mC = (v_b > half) & (v_b <= (half + d))
        mD = v_b > (half + d)

        # Case A: min(u, v)
        C[mA] = np.minimum(u_b[mA], v_b[mA])

        # Case B
        if np.any(mB):
            uB, vB = u_b[mB], v_b[mB]
            out = np.empty_like(uB)
            m1 = uB <= (half - d)
            out[m1] = uB[m1]
            m2 = (~m1) & (uB <= half)
            out[m2] = half - d
            m3 = (~m1) & (~m2) & (uB <= (vB + d))
            out[m3] = uB[m3] - d
            out[(~m1) & (~m2) & (~m3)] = vB[(~m1) & (~m2) & (~m3)]
            C[mB] = out

        # Case C
        if np.any(mC):
            uC, vC = u_b[mC], v_b[mC]
            out = np.empty_like(uC)
            m1 = uC <= (vC - d)
            out[m1] = uC[m1]
            m2 = (~m1) & (uC <= half)
            out[m2] = vC[m2] - d
            m3 = (~m1) & (~m2) & (uC <= (half + d))
            out[m3] = uC[m3] + vC[m3] - d - half
            out[(~m1) & (~m2) & (~m3)] = vC[(~m1) & (~m2) & (~m3)]
            C[mC] = out

        # Case D: min(u, v)
        C[mD] = np.minimum(u_b[mD], v_b[mD])

        return C

    def pdf_vectorized(self, u, v):
        """
        Vectorized "density" c(u,v) = ∂^2 C/∂u∂v computed via finite difference in v.
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
    for delta in [0.1, 0.25, 0.4]:
        cop = MedianSwapCopula(delta=delta)
        print(
            f"delta={delta:.3f} -> beta={cop.blomqvists_beta():.6f}, nu={cop.blests_nu():.6f}"
        )
        # parent class plotting calls work with the wrappers:
        # cop.plot_cdf()
        cop.plot_cond_distr_1(plot_type="contour", grid_size=500)
