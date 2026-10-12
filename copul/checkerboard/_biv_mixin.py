"""
Mixin providing the common, exact API of all bivariate checkerboard copulas.

Subclasses must provide ``self.matr`` (the normalised ``m x n`` mass matrix)
and override :meth:`_kernel_signs` (0 = Pi, 1 = Min, -1 = W per cell).

Call conventions (all evaluation methods): ``f(u, v)`` with scalars returns a
``float``; ``f(u, v)`` with broadcastable arrays or ``f(P)`` with an
``(N, 2)`` array returns an ``ndarray``; ``f(u=..., v=...)`` is accepted too.
"""

from __future__ import annotations

import numpy as np

from copul.checkerboard import _biv_engine as eng
from copul.checkerboard import _rank_measures as rank


class BivCheckerboardMixin:
    """Shared exact numerics for BivCheckPi / BivCheckMin / BivCheckW / Mixed."""

    #: evaluation methods are exact and vectorized already (the numerical API
    #: of :mod:`copul.family.core.numeric_api` passes calls through)
    _numeric_native = True

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

    def blests_eta(self) -> float:
        """Symmetrized Blest coefficient (exact)."""
        return rank.blests_eta(self.matr, self._kernel_signs())

    def siburg_stoimenov(self) -> float:
        """Square root of the mean of the two directional xi coefficients (exact)."""
        return rank.siburg_stoimenov(self.matr, self._kernel_signs())

    def trutschnig_zeta(self) -> float:
        """Trutschnig's zeta1 (exact, including singular cells)."""
        return rank.trutschnig_zeta(self.matr, self._kernel_signs())

    def hoeffdings_d(self) -> float:
        """Hoeffding's Lebesgue L² index Phi² (exact)."""
        return rank.hoeffdings_d(self.matr, self._kernel_signs())

    def hoeffding_d(self) -> float:
        """Hoeffding's D, integrated against the copula measure (exact)."""
        return rank.hoeffding_d(self.matr, self._kernel_signs())

    def blum_kiefer_rosenblatt(self) -> float:
        """Legacy name for the copula-measure integral D (exact)."""
        return rank.hoeffding_d(self.matr, self._kernel_signs())

    def bergsma_dassios(self) -> float:
        """Bergsma--Dassios sign covariance, normalized to maximum 2/3 (exact)."""
        return rank.bergsma_dassios(self.matr, self._kernel_signs())

    def schweizer_wolff_sigma(self) -> float:
        """L¹ distance to independence (exact polynomial/logarithmic formula)."""
        return rank.schweizer_wolff_sigma(self.matr, self._kernel_signs())

    def uniform_distance(self) -> float:
        """Uniform distance to independence (exact)."""
        return rank.uniform_distance(self.matr, self._kernel_signs())

    def normal_score_correlation(self) -> float:
        """Closed for Pi cells and aligned global Min/W diagonals; otherwise numerical."""
        return rank.normal_score_correlation(self.matr, self._kernel_signs())

    def savage_score_correlation(self) -> float:
        """Savage scores (exact cell integrals using logarithms and dilogarithms)."""
        return rank.savage_score_correlation(self.matr, self._kernel_signs())

    def correlation_ratio(self, *, condition_on_y=False) -> float:
        """Squared correlation ratio (exact); optionally condition on V."""
        return rank.correlation_ratio(
            self.matr, self._kernel_signs(), condition_on_y=condition_on_y
        )

    def integrated_r2(self, *, condition_on_y=False) -> float:
        """Integrated R² (exact logarithmic formula); optionally condition on V."""
        return rank.integrated_r2(self.matr, self._kernel_signs(), condition_on_y=condition_on_y)

    def maximal_correlation(self, *, max_grid=256) -> float:
        """Exact finite matrix norms; independent of the numerical ``max_grid`` option."""
        return rank.maximal_correlation(self.matr, self._kernel_signs())

    def linfoot_correlation(self) -> float:
        """Linfoot correlation (exact), equal to one with positive singular mass."""
        return rank.linfoot_correlation(self.matr, self._kernel_signs())

    def hellinger_correlation(self) -> float:
        """Hellinger correlation (exact, accounting for singular mass)."""
        return rank.hellinger_correlation(self.matr, self._kernel_signs())

    def silvey_delta(self) -> float:
        """Silvey's total variation from independence (exact)."""
        return rank.silvey_delta(self.matr, self._kernel_signs())

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

    def _tail_monotonicity(self, kind: str, i: int = 1) -> bool:
        P = self.matr
        S = self._kernel_signs()
        if i == 2:  # (U|V): the same property of the transposed copula
            P = np.asarray(P, dtype=float)
            S = eng._signs(P, S)
            P, S = P.T, (None if S is None else S.T)
        elif i != 1:
            raise ValueError("i must be 1 or 2")
        return eng.tail_monotonicity(P, S, kind)

    def is_ltd(self, *args, i: int = 1, **kwargs) -> bool:
        """Left tail decreasing LTD(V|U) (``i=1``) or LTD(U|V) (``i=2``) (exact)."""
        return self._tail_monotonicity("ltd", i)

    def is_lti(self, *args, i: int = 1, **kwargs) -> bool:
        """Left tail increasing LTI(V|U) (``i=1``) or LTI(U|V) (``i=2``) (exact)."""
        return self._tail_monotonicity("lti", i)

    def is_rti(self, *args, i: int = 1, **kwargs) -> bool:
        """Right tail increasing RTI(V|U) (``i=1``) or RTI(U|V) (``i=2``) (exact)."""
        return self._tail_monotonicity("rti", i)

    def is_rtd(self, *args, i: int = 1, **kwargs) -> bool:
        """Right tail decreasing RTD(V|U) (``i=1``) or RTD(U|V) (``i=2``) (exact)."""
        return self._tail_monotonicity("rtd", i)

    def is_pqd(self, *args, tol: float = 1e-12, **kwargs) -> bool:
        """Positive quadrant dependence ``C >= uv`` (exact)."""
        return eng.quadrant_dependence(self.matr, self._kernel_signs(), positive=True, tol=tol)

    def is_nqd(self, *args, tol: float = 1e-12, **kwargs) -> bool:
        """Negative quadrant dependence ``C <= uv`` (exact)."""
        return eng.quadrant_dependence(self.matr, self._kernel_signs(), positive=False, tol=tol)
