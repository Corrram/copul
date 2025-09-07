# file: copul/families/other/biv_counterexample_band_copula.py
import sympy as sp
import numpy as np

from copul.families.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CounterexampleBandCopula(BivCopula):
    r"""
    Singular “band” copula defined implicitly by
    \[
      h_v(t) := \partial_1 C(t,v)
      \;=\; \mathbf{1}_{\{t < v/2\}} \;+\; v\,\mathbf{1}_{(v/2,\,(v+1)/2)}(t),
    \]
    and explicitly by
    \[
      C(u,v) \;=\;
      \begin{cases}
        u, & 0 \le u \le v/2,\\[0.6ex]
        \dfrac{v}{2} + v\!\left(u - \dfrac{v}{2}\right)
        \;=\; v\,u + \dfrac{v(1-v)}{2}, & v/2 < u \le (v+1)/2,\\[1.0ex]
        v, & (v+1)/2 < u \le 1.
      \end{cases}
    \]
    This copula is a valid (non‐symmetric) copula with absolutely continuous
    part of density 1 on the strip \(\{(u,v): 2u-1 < v < 2u\}\) and a singular
    part supported on the two boundary line segments \(v=2u\) for
    \(u\in[0,\tfrac12]\) and \(v=2u-1\) for \(u\in[\tfrac12,1]\).

    Basic functionals:
      • Chatterjee's \(\xi(C)=\tfrac12\).
      • Spearman's footrule \(\psi(C)=\tfrac12\).

    Notes
    -----
    • This class is parameter‐free.
    • The returned PDF corresponds to the absolutely continuous part only.
    """

    # No parameters
    params = []
    intervals = {}

    # convenience symbols
    u, v = sp.symbols("u v", real=True)

    # -------- Symbolic CDF -------- #
    @property
    def _cdf_expr(self):
        u, v = self.u, self.v
        v_over_2 = v / 2
        vp1_over_2 = (v + 1) / 2
        middle = v_over_2 + v * (u - v_over_2)  # = v*u + v*(1-v)/2
        return sp.Piecewise(
            (u, u <= v_over_2),
            (middle, u <= vp1_over_2),
            (v, True),
        )

    # -------- Symbolic PDF (a.c. part only) -------- #
    def _pdf_expr(self):
        u, v = self.u, self.v
        # density = 1 on the open strip (2u-1, 2u), 0 elsewhere (ignoring singular lines)
        expr = sp.Piecewise(
            (1, sp.And(v > 2 * u - 1, v < 2 * u)),
            (0, True),
        )
        return SymPyFuncWrapper(expr)

    # -------- Vectorized CDF -------- #
    def cdf_vectorized(self, u, v):
        """
        Fast numpy implementation of the piecewise CDF.
        """
        u = np.asarray(u)
        v = np.asarray(v)

        a = v / 2.0
        b = (v + 1.0) / 2.0
        middle = a + v * (u - a)  # = v*u + v*(1-v)/2

        return np.select([u <= a, u <= b], [u, middle], default=v)

    # -------- Properties / metadata -------- #
    @property
    def is_absolutely_continuous(self) -> bool:
        # There is a singular component on two line segments.
        return False

    @property
    def is_symmetric(self) -> bool:
        # C(u,v) ≠ C(v,u) in general.
        return False

    # -------- Measures requested -------- #
    def xi(self):
        """Chatterjee's rank correlation ξ(C) = 1/2."""
        return sp.Rational(1, 2)

    def psi(self):
        """Spearman's footrule ψ(C) = 1/2."""
        return sp.Rational(1, 2)


if __name__ == "__main__":
    cop = CounterexampleBandCopula()
    # Quick sanity checks
    cop.plot_cdf()
    cop.plot_cond_distr_1()
    cop.plot_cond_distr_2()
