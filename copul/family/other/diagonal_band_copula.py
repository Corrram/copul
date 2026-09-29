import sympy as sp

from copul.family.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class DiagonalBandCopula(BivCopula):
    r"""Bojarski-type *diagonal band copula* (uniform band along :math:`y=x`).

    A stripe of half-width :math:`\alpha` is laid along the main diagonal and
    *wrapped/reflected* at the unit square’s borders so that both marginals remain
    uniform.

    Following Bojarski (2002, *J. Math. Sci.*, Eq. (1)) with a **constant** base
    density

    .. math::

       f(z) \;=\; \frac{1}{2\alpha}\,\mathbf{1}\{|z|\le \alpha\}, \quad z\in\mathbb{R},

    supported on :math:`[-\alpha,\alpha]`. Using a different symmetric base density
    (e.g., rescaled Beta) is a straightforward extension, but the uniform band
    already reproduces the classical diagonal-band example discussed in the paper.

    Parameters
    ----------
    \alpha : float in (0, 1]
        Half-width of the diagonal band.
    """

    alpha = sp.symbols("alpha", positive=True)
    params = [alpha]
    intervals = {"alpha": sp.Interval(0, 1, left_open=True, right_open=False)}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _validate_alpha(self, val):
        if val <= 0 or val > 1:
            raise ValueError(f"alpha must be in (0,1], got {val}")

    # base density  f(z)  (uniform on [-α, α])
    def _f(self, z):
        return sp.Piecewise(
            (1 / (2 * self.alpha), sp.Abs(z) <= self.alpha),
            (0, True),
        )

    # ------------------------------------------------------------------
    # constructor + call
    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["alpha"] = args[0]
        if "alpha" in kwargs:
            self._validate_alpha(kwargs["alpha"])
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["alpha"] = args[0]
        if "alpha" in kwargs:
            self._validate_alpha(kwargs["alpha"])
        return super().__call__(**kwargs)

    # ------------------------------------------------------------------
    # basic flags
    # ------------------------------------------------------------------
    @property
    def is_absolutely_continuous(self):
        return True

    @property
    def is_symmetric(self):
        return True

    # ------------------------------------------------------------------
    # PDF   g_α(u,v)
    # ------------------------------------------------------------------
    @property
    def pdf(self):
        r"""Piecewise density :math:`g_\alpha(u,v)` of the diagonal-band construction.

        With the base density :math:`f(z)=\tfrac{1}{2\alpha}\mathbf{1}\{|z|\le \alpha\}`,
        the copula density is

        .. math::

           g_\alpha(u,v)
           \;=\;
           \begin{cases}
             f(u-v) + f(u+v), & u+v \le \alpha,\\[0.5ex]
             f(u-v),          & \alpha < u+v < 2-\alpha,\\[0.5ex]
             f(u-v) + f(u+v-2), & u+v \ge 2-\alpha,
           \end{cases}

        which enforces uniform margins by wrapping the diagonal band near the corners.
        """

        u, v, a = self.u, self.v, self.alpha
        term1 = self._f(u - v)
        pdf_expr = sp.Piecewise(
            # region close to lower‑left corner: x+y ≤ α
            (term1 + self._f(u + v), u + v - a <= 0),
            # region close to upper‑right corner: x+y ≥ 2-α
            (term1 + self._f(u + v - 2), u + v - 2 + a >= 0),
            # central band
            (term1, True),
        )
        return SymPyFuncWrapper(sp.simplify(pdf_expr))

    # ------------------------------------------------------------------
    # CDF   C(u,v)  (closed form)
    # ------------------------------------------------------------------
    def _F(self, z):
        """CDF of the base distribution, uniform on [-alpha, alpha]."""
        a = self.alpha
        return sp.Piecewise((0, z <= -a), ((z + a) / (2 * a), z <= a), (1, True))

    def _G(self, z):
        """Integrated base CDF, G(z) = int_{-oo}^z F(s) ds."""
        a = self.alpha
        return sp.Piecewise((0, z <= -a), ((z + a) ** 2 / (4 * a), z <= a), (z, True))

    @property
    def _cdf_expr(self):
        r"""Closed-form CDF.

        Integrating the density twice gives, with the integrated base CDF
        :math:`G(z)=\int_{-\infty}^z F(s)\,ds`,

        .. math::

           C(u,v) = G(u+v) - G(u-v) + G(-v) - G(v) + G(u+v-2)
                    - G(u-2) - G(v-2) + G(-2).
        """
        u, v, G = self.u, self.v, self._G
        return G(u + v) - G(u - v) + G(-v) - G(v) + G(u + v - 2) - G(u - 2) - G(v - 2) + G(-2)

    # ------------------------------------------------------------------
    # Conditional  F_{U|V}(u|v)
    # ------------------------------------------------------------------
    def cond_distr_2(self, u=None, v=None):
        r""":math:`\partial_2 C = F(u-v)-F(-v)+F(u+v)-F(v)+F(u+v-2)-F(v-2)`."""
        x, y, F = self.u, self.v, self._F
        cd2 = F(x - y) - F(-y) + F(x + y) - F(y) + F(x + y - 2) - F(y - 2)
        return SymPyFuncWrapper(cd2)(u, v)

    def _numeric_callables(self):
        """Vectorized closed forms of the diagonal band copula."""
        import numpy as np

        a = float(self.alpha)

        def F(z):
            return np.clip((z + a) / (2 * a), 0.0, 1.0)

        def G(z):
            return np.where(z <= -a, 0.0, np.where(z <= a, (z + a) ** 2 / (4 * a), z))

        def f(z):
            return np.where(np.abs(z) <= a, 1.0 / (2 * a), 0.0)

        def cdf(u, v):
            return G(u + v) - G(u - v) + G(-v) - G(v) + G(u + v - 2) - G(u - 2) - G(v - 2) + G(-2.0)

        def h2(u, v):
            return F(u - v) - F(-v) + F(u + v) - F(v) + F(u + v - 2) - F(v - 2)

        def pdf(u, v):
            return f(u - v) + f(u + v) + f(u + v - 2)

        return {"cdf": cdf, "h1": lambda u, v: h2(v, u), "h2": h2, "pdf": pdf}


if __name__ == "__main__":
    # Example usage
    x = 0.05
    copula = DiagonalBandCopula(x)
    # copula.plot_cdf()
    # copula.plot_cond_distr_1()
    # copula.plot_cond_distr_2()
    # copula.scatter_plot()
    copula.plot_pdf(title=f"Diagonal Band Copula (delta={x})", plot_type="contour")
    # copula.survival_copula().plot_pdf(
    #     title=f"Diagonal Band Survival Copula (delta={x})", plot_type="contour"
    # )
