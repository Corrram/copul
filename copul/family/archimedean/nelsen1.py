from typing import TypeAlias

import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.lower_frechet import LowerFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class BivClayton(BivArchimedeanCopula):
    """
    Bivariate Clayton Copula.

    The Clayton copula is defined by its generator:
    φ(t) = (t^(-θ) - 1) / θ for θ > 0
    φ(t) = -log(t) for θ = 0
    φ(t) = (t^(-θ) - 1) / θ for θ ∈ [-1, 0) with restricted range

    It allows for asymptotic lower tail dependence but no upper tail dependence.

    Parameters
    ----------
    theta : float
        The parameter controlling the strength of dependence.
        θ > 0 indicates positive dependence, θ = 0 gives independence,
        and θ ∈ [-1, 0) gives negative dependence.
    """

    theta_interval = sympy.Interval(-1, np.inf, left_open=False, right_open=True)
    # Define special cases as a dictionary mapping parameter values to classes
    special_cases = {-1: LowerFrechet, 0: BivIndependenceCopula}

    @property
    def _generator_at_0(self):
        """
        Value of the generator at t=0, which depends on θ.

        Returns
        -------
        sympy expression
            ∞ for θ ≥ 0, -1/θ for θ < 0
        """
        return sympy.Piecewise(
            (sympy.oo, self.theta >= 0),  # For θ ≥ 0, limit is infinity
            (-1 / self.theta, True),  # For θ < 0, limit is -1/θ
        )

    @property
    def _raw_generator(self):
        """
        Raw generator function for Clayton copula.

        Returns
        -------
        sympy expression
            The Clayton generator
        """
        # Regular case expression for theta != 0
        regular_expr = ((1 / self.t) ** self.theta - 1) / self.theta
        # Logarithmic generator for theta = 0
        log_expr = -sympy.log(self.t)

        # Return appropriate expression based on theta
        return sympy.Piecewise(
            (log_expr, self.theta == 0),  # Independence case (θ = 0)
            (regular_expr, True),  # Regular case (θ ≠ 0)
        )

    @property
    def _raw_inv_generator(self):
        """
        Raw inverse generator function for Clayton copula.

        Returns
        -------
        sympy expression
            The inverse generator
        """
        # Handle independence case (θ = 0)
        if self.theta == 0:
            return sympy.exp(-self.y)
        # Regular case (θ ≠ 0)
        return (self.theta * self.y + 1) ** (-1 / self.theta)

    @property
    def _cdf_expr(self):
        """
        Cumulative distribution function of the bivariate Clayton copula.

        Returns
        -------
        CDFWrapper
            The CDF function C(u,v)
        """
        u = self.u
        v = self.v
        theta = self.theta

        # Special case for theta = 0 (Independence Copula)
        if theta == 0:
            return CDFWrapper(u * v)

        # Regular formula for Clayton copula
        cdf = sympy.Max((u ** (-theta) + v ** (-theta) - 1), 0) ** (-1 / theta)
        return cdf

    def _numeric_callables(self):
        r"""Closed forms for the numerical API.

        With :math:`s=u^{-\theta}+v^{-\theta}-1`:
        :math:`C=s_+^{-1/\theta}`,
        :math:`\partial_1C=(1+u^{\theta}(v^{-\theta}-1))_+^{-1-1/\theta}`,
        :math:`\log c=\log(1+\theta)-(1+\theta)\log(uv)-(2+1/\theta)\log s`
        (log-sum-exp for :math:`\log s`), the conditional quantile
        :math:`v=(1+u^{-\theta}(w^{-\theta/(1+\theta)}-1))^{-1/\theta}` and, for
        :math:`\theta>0`, Marshall--Olkin sampling with a Gamma(1/θ) frailty.
        """
        from copul.family.archimedean import _frailty

        th = float(self.theta)
        if th == 0:
            return {}

        def cdf(u, v):
            s = u ** (-th) + v ** (-th) - 1.0
            return np.where(s > 0, np.maximum(s, 0.0) ** (-1.0 / th), 0.0)

        def _h(a, b):
            base = 1.0 + a**th * (b ** (-th) - 1.0)
            return np.where(base > 0, np.maximum(base, 0.0) ** (-1.0 - 1.0 / th), 0.0)

        def logpdf(u, v):
            lu, lv = np.log(u), np.log(v)
            if th > 0:
                a, b = -th * lu, -th * lv
                m = np.maximum(a, b)
                log_s = m + np.log(np.exp(a - m) + np.exp(b - m) - np.exp(-m))
            else:
                s = u ** (-th) + v ** (-th) - 1.0
                log_s = np.where(s > 0, np.log(np.maximum(s, 1e-300)), -np.inf)
            val = np.log1p(th) - (1.0 + th) * (lu + lv) - (2.0 + 1.0 / th) * log_s
            if th < 0:
                val = np.where(np.isfinite(log_s), val, -np.inf)
            return val

        def pdf(u, v):
            return np.exp(logpdf(u, v))

        def _h_inv(a, w):
            return (1.0 + a ** (-th) * (w ** (-th / (1.0 + th)) - 1.0)) ** (-1.0 / th)

        out = {
            "cdf": cdf,
            "h1": _h,
            "h2": lambda u, v: _h(v, u),
            "pdf": pdf,
            "logpdf": logpdf,
            "h1_inv": _h_inv,
            "h2_inv": _h_inv,
        }
        if th > 0:

            def rvs(n, rng):
                frailty = _frailty.gamma_frailty(n, rng, 1.0 / th)
                return _frailty.marshall_olkin(lambda t: np.exp(-np.log1p(t) / th), frailty, rng)

            out["rvs"] = rvs
        return out

    def cond_distr_1(self, u=None, v=None):
        """
        First conditional distribution function: ∂C(u,v)/∂u

        Parameters
        ----------
        u, v : float or None, optional
            Values to evaluate at. If None, returns the symbolic expression.

        Returns
        -------
        CD1Wrapper
            The conditional distribution
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return v  # For independence copula, conditional distribution is just v

        # Formula for Clayton copula
        cond_distr = sympy.Heaviside(-1 + self.u ** (-theta) + self.v ** (-theta)) / (
            self.u
            * self.u**theta
            * (-1 + self.u ** (-theta) + self.v ** (-theta))
            * (-1 + self.u ** (-theta) + self.v ** (-theta)) ** (1 / theta)
        )
        return CD1Wrapper(cond_distr)(u, v)

    def cond_distr_2(self, u=None, v=None):
        """
        Second conditional distribution function: ∂C(u,v)/∂v

        Parameters
        ----------
        u, v : float or None, optional
            Values to evaluate at. If None, returns the symbolic expression.

        Returns
        -------
        CD2Wrapper
            The conditional distribution
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return u  # For independence copula, conditional distribution is just u

        # Formula for Clayton copula
        cond_distr = sympy.Heaviside(
            (-1 + self.v ** (-theta) + self.u ** (-theta)) ** (-1 / theta)
        ) / (
            self.v
            * self.v**theta
            * (-1 + self.v ** (-theta) + self.u ** (-theta))
            * (-1 + self.v ** (-theta) + self.u ** (-theta)) ** (1 / theta)
        )
        return CD2Wrapper(cond_distr)(u, v)

    def _squared_cond_distr_1(self, u, v):
        """
        Second partial derivative of CDF with respect to u.

        Parameters
        ----------
        u, v : float
            Values to evaluate at

        Returns
        -------
        sympy expression
            The second derivative
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return 0  # For independence copula, second derivative is 0

        # Formula for Clayton copula
        return sympy.Heaviside((-1 + v ** (-theta) + u ** (-theta)) ** (-1 / theta)) / (
            u**2
            * u ** (2 * theta)
            * (-1 + v ** (-theta) + u ** (-theta)) ** 2
            * (-1 + v ** (-theta) + u ** (-theta)) ** (2 / theta)
        )

    @property
    def pdf(self):
        """
        Probability density function of the bivariate Clayton copula.

        Returns
        -------
        SymPyFuncWrapper
            The PDF function c(u,v)
        """
        theta = self.theta

        # Handle special case for theta = 0
        if theta == 0:
            return SymPyFuncWrapper(1)  # Uniform density for independence copula

        # Formula for Clayton copula
        result = (
            (self.u ** (-theta) + self.v ** (-theta) - 1) ** (-2 - 1 / theta)
            * self.u ** (-theta - 1)
            * self.v ** (-theta - 1)
            * (theta + 1)
        )
        return SymPyFuncWrapper(result)

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        Returns
        -------
        bool
            True for θ > -1 (for -1 < θ < 0 the zero set carries no mass since
            the generator has infinite slope at 0, Nelsen 2006, Thm. 4.3.3);
            θ = -1 is the lower Fréchet bound.
        """
        return self.theta > -1

    # ------------------------------------------------------------------
    # Additional dependence measures
    # ------------------------------------------------------------------

    def blomqvists_beta(self, *args, **kwargs):
        r"""
        Blomqvist's :math:`\beta` for the Clayton copula.

        .. math::

           \beta = 4\,(2^{-\theta}+2^{-\theta}-1)^{-1/\theta}\cdot
                   \bigl[(\text{at } u=v=\tfrac12)\bigr] - 1

        For :math:`\theta > 0`:

        .. math::

           C(\tfrac12,\tfrac12) = (2^{1-\theta}-1)^{-1/\theta} \cdot 2^{-1}
                                \;\Longrightarrow\;
           \beta = 4\,(2\cdot 2^{-\theta}-1)^{-1/\theta} - 1
        """
        self._set_params(args, kwargs)
        theta = float(self.theta)
        if theta == 0:
            return 0
        # C(1/2,1/2) = (2·2^θ - 1)^{-1/θ}
        c_half = (2.0 * 2.0**theta - 1.0) ** (-1.0 / theta)
        return 4.0 * c_half - 1.0

    def kendalls_tau(self, *args, **kwargs):
        r"""Kendall's :math:`\tau = \theta/(\theta+2)` of the Clayton copula."""
        self._set_params(args, kwargs)
        return self.theta / (self.theta + 2)

    def schweizer_wolff_sigma(self, *args, **kwargs):
        r"""
        Schweizer–Wolff :math:`\sigma` for the Clayton copula.

        For :math:`\theta \ge 0` the Clayton copula is PQD so
        :math:`\sigma = \rho_S`.  For :math:`\theta \in [-1,0)` it is NQD
        so :math:`\sigma = -\rho_S`.  In both cases
        :math:`\sigma = |\rho_S|`.
        """
        self._set_params(args, kwargs)
        rho = self.spearmans_rho()
        return abs(rho)

    def lambda_L(self):
        """
        Lower tail dependence coefficient.

        For Clayton copula with θ > 0, this is 2^(-1/θ).

        Returns
        -------
        float or sympy expression
            The lower tail dependence coefficient
        """
        # Clayton with θ = 0 has no tail dependence
        if self.theta == 0:
            return 0
        # Formula for Clayton with θ > 0
        return 2 ** (-1 / self.theta)

    def lambda_U(self):
        """
        Upper tail dependence coefficient.

        For Clayton copula, this is always 0.

        Returns
        -------
        float
            Always 0 for Clayton
        """
        return 0


# Type alias for backwards compatibility
Clayton: TypeAlias = BivClayton
Nelsen1: TypeAlias = BivClayton


if __name__ == "__main__":
    # Example usage
    copula = BivClayton(theta=1)
    result = copula.cdf(u=0.5)
    # Test the new rvs method
    print("Generating 5 samples with the fast rvs method:")
    samples = copula.rvs(5, random_state=42)
    print(samples)

    copula_neg = BivClayton(theta=-0.8)
    print("\nGenerating 5 samples with a negative theta:")
    samples_neg = copula_neg.rvs(5, random_state=42)
    print(samples_neg)

    print("\nDone!")
