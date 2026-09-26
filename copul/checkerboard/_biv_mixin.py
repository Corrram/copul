"""
Mixin providing the common, exact API of all bivariate checkerboard copulas.

Subclasses must provide ``self.matr`` (the normalised ``m x n`` mass matrix)
and override :meth:`_kernel_signs` (0 = Pi, 1 = Min, -1 = W per cell).

Call conventions (all evaluation methods): ``f(u, v)`` with scalars returns a
``float``; ``f(u, v)`` with broadcastable arrays or ``f(P)`` with an
``(N, 2)`` array returns an ``ndarray``; ``f(u=..., v=...)`` is accepted too.
"""

from __future__ import annotations

from copul.checkerboard import _biv_engine as eng


class BivCheckerboardMixin:
    """Shared exact numerics for BivCheckPi / BivCheckMin / BivCheckW / Mixed."""

    # ------------------------------------------------------------------
    # kernel description
    # ------------------------------------------------------------------
    def _kernel_signs(self):
        """Per-cell kernel sign matrix (``None`` means independence everywhere)."""
        return None

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def cdf(self, *args, **kwargs):
        """Distribution function, see module docstring for call conventions."""
        u, v, scalar = eng.parse_uv(args, kwargs)
        return eng.finish(eng.cdf(self.matr, self._kernel_signs(), u, v), scalar)

    def cdf_vectorized(self, u, v):
        """Vectorised cdf at broadcastable ``u, v`` (always returns an array)."""
        return eng.cdf(self.matr, self._kernel_signs(), u, v)

    def cond_distr(self, i, *args, **kwargs):
        """Conditional distribution ``F_{U_{-i} | U_i}`` (``i`` in ``{1, 2}``)."""
        if i not in (1, 2):
            raise ValueError(f"Dimension {i} out of range 1..2")
        u, v, scalar = eng.parse_uv(args, kwargs)
        out = eng.cond_distr(self.matr, self._kernel_signs(), i, u, v)
        return eng.finish(out, scalar)

    def cond_distr_1(self, *args, **kwargs):
        """``P(V <= v | U = u)``."""
        return self.cond_distr(1, *args, **kwargs)

    def cond_distr_2(self, *args, **kwargs):
        """``P(U <= u | V = v)``."""
        return self.cond_distr(2, *args, **kwargs)

    def rvs(self, n=1, random_state=None, **kwargs):
        """Draw ``n`` samples; ``random_state`` is an int, Generator or None.

        ``None`` uses NumPy's global generator without reseeding it.
        """
        if "size" in kwargs and kwargs["size"] is not None:
            n = kwargs["size"]
        return eng.rvs(self.matr, self._kernel_signs(), n, random_state)

    # ------------------------------------------------------------------
    # closed-form dependence measures
    # ------------------------------------------------------------------
    def spearmans_rho(self, *args, **kwargs) -> float:
        """Spearman's rho (exact)."""
        return eng.spearmans_rho(self.matr, self._kernel_signs())

    def kendalls_tau(self, *args, **kwargs) -> float:
        """Kendall's tau (exact)."""
        return eng.kendalls_tau(self.matr, self._kernel_signs())

    def chatterjees_xi(self, *, condition_on_y: bool = False) -> float:
        """Chatterjee's xi (exact); ``condition_on_y=True`` gives xi(U | V)."""
        return eng.chatterjees_xi(self.matr, self._kernel_signs(), condition_on_y=condition_on_y)

    def blests_nu(self, *args, **kwargs) -> float:
        """Blest's nu (exact)."""
        return eng.blests_nu(self.matr, self._kernel_signs())

    def spearmans_footrule(self, *args, **kwargs) -> float:
        """Spearman's footrule (exact, any ``m x n`` grid)."""
        return eng.spearmans_footrule(self.matr, self._kernel_signs())

    def ginis_gamma(self, *args, **kwargs) -> float:
        """Gini's gamma (exact, any ``m x n`` grid)."""
        return eng.ginis_gamma(self.matr, self._kernel_signs())

    # names used by the generic core copula API
    def spearman_footrule(self, *args, **kwargs) -> float:
        return self.spearmans_footrule()

    def gini_gamma(self, *args, **kwargs) -> float:
        return self.ginis_gamma()

    def blomqvists_beta(self, *args, **kwargs) -> float:
        """Blomqvist's beta ``4 C(1/2, 1/2) - 1``."""
        return eng.blomqvists_beta(self.matr, self._kernel_signs())

    def lambda_L(self):
        """Lower tail dependence (``Delta_00 min(m, n)`` for a Min corner cell)."""
        return eng.lambda_L(self.matr, self._kernel_signs())

    def lambda_U(self):
        """Upper tail dependence (``Delta_mn min(m, n)`` for a Min corner cell)."""
        return eng.lambda_U(self.matr, self._kernel_signs())

    # ------------------------------------------------------------------
    # exact dependence properties
    # ------------------------------------------------------------------
    def cis_direction(self, i: int = 1):
        """Exact ``(is_SI, is_SD)`` of the conditional distribution ``i``."""
        return eng.cis_direction(self.matr, self._kernel_signs(), which=i)

    def is_cis(self, i: int = 1) -> bool:
        """Stochastically increasing w.r.t. conditioning variable ``i`` (exact)."""
        return self.cis_direction(i)[0]

    def is_si(self, i: int = 1) -> bool:
        """Alias of :meth:`is_cis`."""
        return self.is_cis(i)

    def is_cds(self, i: int = 1) -> bool:
        """Stochastically decreasing w.r.t. conditioning variable ``i`` (exact)."""
        return self.cis_direction(i)[1]

    def is_ltd(self, *args, **kwargs) -> bool:
        """Left tail decreasing LTD(V|U) (exact)."""
        return eng.tail_monotonicity(self.matr, self._kernel_signs(), "ltd")

    def is_lti(self, *args, **kwargs) -> bool:
        return eng.tail_monotonicity(self.matr, self._kernel_signs(), "lti")

    def is_rti(self, *args, **kwargs) -> bool:
        """Right tail increasing RTI(V|U) (exact)."""
        return eng.tail_monotonicity(self.matr, self._kernel_signs(), "rti")

    def is_rtd(self, *args, **kwargs) -> bool:
        return eng.tail_monotonicity(self.matr, self._kernel_signs(), "rtd")

    def is_pqd(self, *args, tol: float = 1e-12, **kwargs) -> bool:
        """Positive quadrant dependence ``C >= uv`` (exact)."""
        return eng.quadrant_dependence(self.matr, self._kernel_signs(), positive=True, tol=tol)

    def is_nqd(self, *args, tol: float = 1e-12, **kwargs) -> bool:
        """Negative quadrant dependence ``C <= uv`` (exact)."""
        return eng.quadrant_dependence(self.matr, self._kernel_signs(), positive=False, tol=tol)
