import sympy as sp

from copul.family.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class DiagonalBandCopula(BivCopula):
    r"""Bojarski‐type **Diagonal Band Copula** (uniform band along *y=x*).

    A stripe of half‑width ``α`` is laid along the main diagonal and *wrapped/
    reflected* at the square’s borders so that the marginals stay **uniform**.

    The construction follows Bojarski (2002, *J. Math. Sci.*) Eq (1) with
    a *constant* base density
    :math:`f(z)=\frac{1}{2α}\mathbf1\{|z|\le α\}` on ``[-α,α]``.  Setting a
    different, symmetric base density (e.g. a scaled Beta) is a straightforward
    extension, but the uniform band already reproduces the classical diagonal
    band copula discussed in the paper’s example.

    **Parameter**
    ----------
    α : float in (0, 1]
        Half‑width of the band.
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
        """Piece‑wise density *g_α(u,v)* from Bojarski (2002)."""
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
    # CDF   C(u,v)  (symbolic integration w.r.t. first coordinate)
    # ------------------------------------------------------------------
    @property
    def _cdf_expr(self):
        t = sp.symbols("t", nonnegative=True)
        g = self.pdf.func  # underlying sympy Expr from wrapper
        # substitute u -> t to integrate over the first coordinate
        g_sub = g.subs(self.u, t)
        expr = sp.integrate(g_sub, (t, 0, self.u))
        return expr

    # ------------------------------------------------------------------
    # Conditional  F_{U|V}(u|v)
    # ------------------------------------------------------------------
    def cond_distr_2(self, u=None, v=None):
        t = sp.symbols("t", nonnegative=True)
        g = self.pdf.func.subs(self.u, t)
        cd2 = sp.integrate(g, (t, 0, self.u))
        return SymPyFuncWrapper(sp.simplify(cd2))(u, v)


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
