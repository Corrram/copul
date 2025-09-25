# frechet_multi.py
import sympy as sp
import numpy as np

from copul.family.core.copula import Copula
from copul.wrapper.cdf_wrapper import CDFWrapper


class _FrechetMultiMixin:
    r"""
    Multivariate Fréchet-type family.

    In dimension d=2:
        C(u) = α M(u) + (1-α-β) Π(u) + β W(u),
        with α,β ≥ 0 and α+β ≤ 1.

    In dimension d≥3:
        The classical lower Fréchet bound W is not a copula.
        We therefore *force* β = 0 and use the valid mixture
        C(u) = α M(u) + (1-α) Π(u).

    Parameters
    ----------
    alpha : α ∈ [0,1]
        Weight of the upper Fréchet bound M(u)=min(u_1,...,u_d).
    beta : β ∈ [0,1]
        Weight of the lower Fréchet bound.
        - Allowed only in d=2 (with α+β≤1).
        - In d≥3, β is fixed to 0.

    Notes
    -----
    Π(u) = ∏_i u_i,  M(u)=min_i u_i,  W(u)=max(∑ u_i - d + 1, 0) (only d=2 valid).
    """

    # symbolic params (class-level)
    _alpha, _beta = sp.symbols("alpha beta", nonnegative=True)
    params = [_alpha, _beta]
    intervals = {
        "alpha": sp.Interval(0, 1, left_open=False, right_open=False),
        "beta": sp.Interval(0, 1, left_open=False, right_open=False),
    }

    # ---------- properties ----------
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        # tighten beta interval dynamically
        if self.dim == 2:
            self.intervals["beta"] = sp.Interval(0, 1 - float(self.alpha), False, False)
        else:
            # d≥3: always β=0
            self.intervals["beta"] = sp.Interval(0, 0, False, False)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        # d≥3: enforce β=0
        if getattr(self, "dim", None) and self.dim >= 3 and value != 0:
            raise ValueError("In d≥3 the lower Fréchet weight β must be 0.")
        self._beta = value
        if self.dim == 2:
            self.intervals["alpha"] = sp.Interval(0, 1 - float(self.beta), False, False)
        else:
            self.intervals["alpha"] = sp.Interval(0, 1, False, False)

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def is_absolutely_continuous(self) -> bool:
        # α>0 or β>0 introduce singular components on manifolds; Π-part is absolutely continuous.
        return (self.alpha == 0) and (self.beta == 0)

    # ---------- helpers ----------
    def _M_expr(self):
        # M(u) = min(u_1,...,u_d)
        return sp.Min(*self.u_symbols)

    def _Pi_expr(self):
        # Π(u) = product u_i
        expr = 1
        for ui in self.u_symbols:
            expr *= ui
        return expr

    def _W_expr(self):
        # W(u) = max(sum u_i - d + 1, 0). Valid copula only in d=2.
        s = sum(self.u_symbols) - self.dim + 1
        return sp.Max(s, 0)

    # ---------- core: CDF ----------
    @property
    def cdf(self):
        r"""
        C(u) = α M + (1-α-β) Π + β W  (d=2),
        C(u) = α M + (1-α) Π          (d≥3; β forced to 0).
        """
        a = self.alpha
        b = self.beta
        M = self._M_expr()
        Pi = self._Pi_expr()

        if self.dim == 2:
            # validate α+β ≤ 1
            if (float(a) + float(b)) > 1 + 1e-12:
                raise ValueError(
                    "Parameter constraint violated: alpha + beta ≤ 1 (d=2)."
                )
            W = self._W_expr()
            expr = a * M + (1 - a - b) * Pi + b * W
        else:
            if b != 0:
                raise ValueError("In d≥3, beta must be 0.")
            expr = a * M + (1 - a) * Pi

        return CDFWrapper(expr)

    # ---------- optional vectorized CDF ----------
    # ---------- optional vectorized CDF ----------
    def cdf_vectorized(self, *args):
        """
        Vectorized CDF evaluation.

        Supported call patterns
        -----------------------
        (2D fast path used by Checkerboarder):
            cdf_vectorized(U, V)
                U, V : array_like broadcastable to a common 2D shape
                Returns an array of that shape.

        (General d-dimensional batch):
            cdf_vectorized(P)
                P : array_like of shape (n, d)
                Returns a 1D array of shape (n,).

        Notes
        -----
        - In d=2 we implement the closed-form directly on broadcasted arrays to
          match the checkerboard fast path.
        - In d>=3 we accept only the (n, d) form.
        """

        # --- 2D signature: cdf_vectorized(U, V) ---
        if len(args) == 2:
            if self.dim != 2:
                raise ValueError("cdf_vectorized(U, V) is only valid for d=2.")
            U, V = np.asarray(args[0], dtype=float), np.asarray(args[1], dtype=float)
            if np.any((U < 0) | (U > 1)) or np.any((V < 0) | (V > 1)):
                raise ValueError("Inputs must lie in [0,1].")

            a = float(self.alpha)
            b = float(self.beta)
            if a + b > 1 + 1e-12:
                raise ValueError(
                    "Parameter constraint violated: alpha + beta ≤ 1 (d=2)."
                )

            frechet_upper = np.minimum(U, V)  # M(u,v)
            frechet_lower = np.maximum(U + V - 1.0, 0.0)  # W(u,v)
            independence = U * V  # Π(u,v)

            return a * frechet_upper + (1 - a - b) * independence + b * frechet_lower

        # --- general signature: cdf_vectorized(P) with P shape (n, d) ---
        if len(args) != 1:
            raise TypeError(
                "cdf_vectorized expects either (U, V) in d=2 or a single (n,d) array."
            )

        P = np.asarray(args[0], dtype=float)
        if P.ndim == 1:
            P = P[None, :]
        if P.shape[1] != self.dim:
            raise ValueError(f"Expected points of shape (n,{self.dim}).")
        if np.any((P < 0) | (P > 1)):
            raise ValueError("All coordinates must lie in [0,1].")

        a = float(self.alpha)
        b = float(self.beta)

        Pi = np.prod(P, axis=1)
        M = np.min(P, axis=1)

        if self.dim == 2:
            W = np.maximum(P.sum(axis=1) - 1.0, 0.0)
            if a + b > 1 + 1e-12:
                raise ValueError(
                    "Parameter constraint violated: alpha + beta ≤ 1 (d=2)."
                )
            return a * M + (1 - a - b) * Pi + b * W

        # d >= 3: β must be 0 and W is not a copula
        if abs(b) > 1e-15:
            raise ValueError("In d≥3, beta must be 0.")
        return a * M + (1 - a) * Pi


class MVFrechet(_FrechetMultiMixin, Copula):
    """
    Multivariate Fréchet family for arbitrary dimension d >= 2.

    Usage:
        C = MVFrechet(dimension=3, alpha=0.4)        # beta auto-forced to 0
        C = MVFrechet(dimension=2, alpha=0.3, beta=0.2)

    Notes:
        - Intervals for alpha/beta are auto-tightened based on (dimension, other parameter).
        - In d≥3, attempting to set beta!=0 raises ValueError.
    """

    def __init__(self, dimension, *args, **kwargs):
        # default params
        self._alpha = kwargs.pop("alpha", 0.0)
        self._beta = kwargs.pop("beta", 0.0)
        # initialize Copula (sets dim, u_symbols, etc.)
        super().__init__(dimension, *args, **kwargs)
        # finalize intervals according to dim/params
        if self.dim == 2:
            self.intervals["alpha"] = sp.Interval(
                0, 1 - float(self._beta), False, False
            )
            self.intervals["beta"] = sp.Interval(
                0, 1 - float(self._alpha), False, False
            )
        else:
            # β is locked to 0
            self._beta = 0.0
            self.intervals["alpha"] = sp.Interval(0, 1, False, False)
            self.intervals["beta"] = sp.Interval(0, 0, False, False)


if __name__ == "__main__":
    copula = MVFrechet(dimension=3, alpha=0.8)
    checkerboard = copula.to_checkerboard(5)
    checkerboard.scatter_plot()
