import numpy as np
import sympy

from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula


class Galambos(BivExtremeValueCopula):
    r"""
    Galambos extreme value copula with parameter :math:`\delta > 0`.

    CDF:
        C(u,v) = u v exp( ((-log u)^(-delta) + (-log v)^(-delta))^(-1/delta) )

    for (u,v) in (0,1]^2, with the usual boundary extensions.
    """

    delta = sympy.symbols("delta", positive=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, sympy.oo, left_open=True, right_open=True)}

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        expr = 1 - (self.t ** (-self.delta) + (1 - self.t) ** (-self.delta)) ** (
            -1 / self.delta
        )
        return sympy.Piecewise(
            (1, sympy.Or(sympy.Eq(self.t, 0), sympy.Eq(self.t, 1))),
            (expr, True),
        )

    @property
    def _cdf_expr(self):
        u = self.u
        v = self.v
        delta = self.delta
        return (
            u
            * v
            * sympy.exp(
                (sympy.log(1 / u) ** (-delta) + sympy.log(1 / v) ** (-delta))
                ** (-1 / delta)
            )
        )

    # ------------------------------------------------------------------
    # Fast CDF: route concrete evaluations to the vectorized implementation
    # instead of symbolic SymPy evaluation.
    # ------------------------------------------------------------------

    def cdf(self, u=None, v=None, **kwargs):
        """Evaluate the CDF numerically via *cdf_vectorized*."""
        if u is None:
            u = kwargs.pop("u", None)
        if v is None:
            v = kwargs.pop("v", None)
        if u is None or v is None:
            raise TypeError("cdf() requires keyword arguments u and v")

        scalar_in = np.ndim(u) == 0 and np.ndim(v) == 0
        result = self.cdf_vectorized(
            np.atleast_1d(np.asarray(u, dtype=float)),
            np.atleast_1d(np.asarray(v, dtype=float)),
        )
        return float(result[0]) if scalar_in else result

    # ------------------------------------------------------------------
    # Fast PDF: numerical mixed-partial via cdf_vectorized.
    # ------------------------------------------------------------------

    @property
    def pdf(self):
        """Numerical PDF via finite differences on *cdf_vectorized*."""
        outer = self

        class _GalambosPDF:
            def __call__(self, u=None, v=None, **kw):
                if u is None:
                    u = kw.get("u")
                if v is None:
                    v = kw.get("v")
                if u is None or v is None:
                    raise TypeError("pdf() requires u and v")

                scalar_in = np.ndim(u) == 0 and np.ndim(v) == 0
                result = outer._pdf_numerical(u, v)
                if scalar_in:
                    return float(np.asarray(result).reshape(-1)[0])
                return result

        return _GalambosPDF()

    def _pdf_numerical(self, u, v, h: float = 1e-5):
        """
        Mixed partial derivative ∂²C/∂u∂v via central differences.

        Supports scalar or array inputs.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        scalar_in = np.ndim(u) == 0 and np.ndim(v) == 0

        shape = np.broadcast(u, v).shape
        u = np.broadcast_to(u, shape).astype(float, copy=False)
        v = np.broadcast_to(v, shape).astype(float, copy=False)

        u_flat = np.atleast_1d(u).ravel()
        v_flat = np.atleast_1d(v).ravel()
        out = np.zeros_like(u_flat, dtype=float)

        interior = (u_flat > 0.0) & (u_flat < 1.0) & (v_flat > 0.0) & (v_flat < 1.0)
        if np.any(interior):
            ui = u_flat[interior]
            vi = v_flat[interior]

            hi = np.minimum.reduce(
                [
                    np.full_like(ui, h, dtype=float),
                    ui / 2.0,
                    (1.0 - ui) / 2.0,
                    vi / 2.0,
                    (1.0 - vi) / 2.0,
                ]
            )

            out[interior] = (
                self.cdf_vectorized(ui + hi, vi + hi)
                - self.cdf_vectorized(ui + hi, vi - hi)
                - self.cdf_vectorized(ui - hi, vi + hi)
                + self.cdf_vectorized(ui - hi, vi - hi)
            ) / (4.0 * hi**2)

        result = out.reshape(shape)
        if scalar_in:
            return float(result.ravel()[0])
        return result

    # ------------------------------------------------------------------
    # Sub-expression helpers (symbolic building blocks of the CDF).
    # ------------------------------------------------------------------

    def _eval_sub_expr_3(self, delta, u, v):
        """Inner power sum: (-log u)^{-delta} + (-log v)^{-delta}."""
        return sympy.log(1 / u) ** (-delta) + sympy.log(1 / v) ** (-delta)

    def _eval_sub_expr(self, delta, u, v):
        """Outer power: (sub_expr_3)^{-1/delta}."""
        return self._eval_sub_expr_3(delta, u, v) ** (-1 / delta)

    def _eval_sub_expr_2(self, delta, u, v):
        """Full CDF expression: u * v * exp(sub_expr)."""
        return u * v * sympy.exp(self._eval_sub_expr(delta, u, v))
