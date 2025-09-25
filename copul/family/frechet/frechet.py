import copy

import sympy

from copul.exceptions import PropertyUnavailableException
from copul.family.core.biv_copula import BivCopula
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper


class Frechet(BivCopula):
    r"""
    Bivariate Fréchet copula (a convex combination of the upper/lower
    Fréchet bounds and independence).

    Parameters
    ----------
    alpha : :math:`\alpha \in [0,1]`
        Weight of the upper Fréchet bound :math:`\min(u,v)`.
    beta : :math:`\beta \in [0,1]`
        Weight of the lower Fréchet bound :math:`\max(u+v-1,0)`.

    Notes
    -----
    The CDF is

    .. math::

       C(u,v)
       \;=\;
       \alpha\,\min(u,v)
       \;+\; (1-\alpha-\beta)\,u\,v
       \;+\; \beta\,\max(u+v-1,0).

    The parameter domain must satisfy :math:`\alpha\ge 0`, :math:`\beta\ge 0`
    and :math:`\alpha+\beta \le 1`.

    The copula is absolutely continuous iff :math:`\alpha=\beta=0`.
    """

    @property
    def is_symmetric(self) -> bool:
        """Whether :math:`C(u,v)=C(v,u)` (always ``True`` for this family)."""
        return True

    _alpha, _beta = sympy.symbols("alpha beta", nonnegative=True)
    params = [_alpha, _beta]
    intervals = {
        "alpha": sympy.Interval(0, 1, left_open=False, right_open=False),
        "beta": sympy.Interval(0, 1, left_open=False, right_open=False),
    }

    @property
    def is_absolutely_continuous(self) -> bool:
        return (self.alpha == 0) & (self.beta == 0)

    @property
    def alpha(self):
        if isinstance(self._alpha, property):
            return self._alpha.fget(self)
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def beta(self):
        if isinstance(self._beta, property):
            return self._beta.fget(self)
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value

    def __init__(self, *args, **kwargs):
        if args and len(args) == 2:
            kwargs["alpha"] = args[0]
            kwargs["beta"] = args[1]
        if "alpha" in kwargs:
            self._alpha = kwargs["alpha"]
            self.intervals["beta"] = sympy.Interval(
                0, 1 - self.alpha, left_open=False, right_open=False
            )
            self.params = [param for param in self.params if str(param) != "alpha"]
            del kwargs["alpha"]
        if "beta" in kwargs:
            self._beta = kwargs["beta"]
            self.intervals["alpha"] = sympy.Interval(
                0, 1 - self.beta, left_open=False, right_open=False
            )
            self.params = [param for param in self.params if str(param) != "beta"]
            del kwargs["beta"]
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        if args and len(args) == 2:
            kwargs["alpha"] = args[0]
            kwargs["beta"] = args[1]
        if "alpha" in kwargs:
            new_copula = copy.deepcopy(self)
            new_copula._alpha = kwargs["alpha"]
            new_copula.intervals["beta"] = sympy.Interval(
                0, 1 - new_copula.alpha, left_open=False, right_open=False
            )
            new_copula.params = [
                param for param in new_copula.params if param != self._alpha
            ]
            del kwargs["alpha"]
            return new_copula.__call__(**kwargs)
        if "beta" in kwargs:
            new_copula = copy.deepcopy(self)
            new_copula._beta = kwargs["beta"]
            new_copula.intervals["alpha"] = sympy.Interval(
                0, 1 - new_copula.beta, left_open=False, right_open=False
            )
            new_copula.params = [
                param for param in new_copula.params if param != self._beta
            ]
            del kwargs["beta"]
            return new_copula.__call__(**kwargs)
        return super().__call__(**kwargs)

    @property
    def cdf(self):
        r"""
        Cumulative distribution function

        .. math::

           C(u,v)
           \;=\;
           \alpha\,\min(u,v)
           \;+\; (1-\alpha-\beta)\,u\,v
           \;+\; \beta\,\max(u+v-1,0).
        """
        frechet_upper = sympy.Min(self.u, self.v)
        frechet_lower = sympy.Max(self.u + self.v - 1, 0)
        cdf = (
            self._alpha * frechet_upper
            + (1 - self._alpha - self._beta) * self.u * self.v
            + self._beta * frechet_lower
        )
        return CDFWrapper(cdf)

    def cdf_vectorized(self, u, v):
        r"""
        Vectorized CDF on many points.

        Parameters
        ----------
        u, v : array_like
            Uniform marginals in :math:`[0,1]`.

        Returns
        -------
        numpy.ndarray
            Values of :math:`C(u,v)`.

        Notes
        -----
        Uses NumPy broadcasting; implements the same formula as above.
        """
        import numpy as np

        # Convert inputs to numpy arrays if they aren't already
        u = np.asarray(u)
        v = np.asarray(v)

        # Ensure inputs are within [0, 1]
        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Marginals must be in [0, 1]")

        # Handle scalar inputs by broadcasting to the same shape
        if u.ndim == 0 and v.ndim > 0:
            u = np.full_like(v, u.item())
        elif v.ndim == 0 and u.ndim > 0:
            v = np.full_like(u, v.item())

        # Get parameter values as floats
        alpha = float(self.alpha)
        beta = float(self.beta)

        # Compute the three components of the Frechet copula using vectorized operations
        frechet_upper = np.minimum(u, v)
        frechet_lower = np.maximum(u + v - 1, 0)
        independence = u * v

        # Combine the components with the weights
        cdf_values = (
            alpha * frechet_upper
            + (1 - alpha - beta) * independence
            + beta * frechet_lower
        )

        return cdf_values

    def cond_distr_1(self, u=None, v=None):
        cond_distr = (
            self._alpha * sympy.Heaviside(self.v - self.u)
            + self._beta * sympy.Heaviside(self.u + self.v - 1)
            + self.v * (-self._alpha - self._beta + 1)
        )
        return CD2Wrapper(cond_distr)(u, v)

    def cond_distr_2(self, u=None, v=None):
        cond_distr = (
            self._alpha * sympy.Heaviside(self.u - self.v)
            + self._beta * sympy.Heaviside(self.u + self.v - 1)
            + self.u * (-self._alpha - self._beta + 1)
        )
        return CD2Wrapper(cond_distr)(u, v)

    def spearmans_rho(self, *args, **kwargs):
        r"""
        Spearman's rank correlation

        .. math:: \rho_S \;=\; \alpha \;-\; \beta.
        """
        self._set_params(args, kwargs)
        return self._alpha - self._beta

    def kendalls_tau(self, *args, **kwargs):
        r"""
        Kendall's :math:`\tau`

        .. math::

           \tau \;=\; \frac{(\alpha-\beta)\,\bigl(2+\alpha+\beta\bigr)}{3}.
        """
        self._set_params(args, kwargs)
        return (self._alpha - self._beta) * (2 + self._alpha + self._beta) / 3

    @property
    def lambda_L(self):
        return self._alpha

    @property
    def lambda_U(self):
        return self._alpha

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return (self.alpha - self.beta) ** 2 + self.alpha * self.beta

    def blests_nu(self, *args, **kwargs):
        """
        Blest's measure of rank correlation ν.
        For the Fréchet copula: ν = α − β.
        """
        self._set_params(args, kwargs)
        return self.alpha - self.beta

    @property
    def pdf(self):
        raise PropertyUnavailableException("Frechet copula does not have a pdf")

    def spearmans_footrule(self, *args, **kwargs):
        r"""
        Spearman's footrule
        :math:`\psi \;=\; \mathbb{E}\,\lvert U-V\rvert`.

        Closed form:

        .. math:: \psi \;=\; \alpha \;-\; \tfrac{1}{2}\,\beta.
        """
        self._set_params(args, kwargs)
        return self.alpha - self.beta / 2

    def ginis_gamma(self, *args, **kwargs):
        r"""
        Gini's gamma :math:`\gamma`.

        For the Fréchet copula,

        .. math:: \gamma \;=\; \alpha \;-\; \beta,

        which coincides with Spearman's :math:`\rho_S` for this family.
        """
        self._set_params(args, kwargs)
        return self.alpha - self.beta


# B11 = lambda: Frechet(beta=0)
if __name__ == "__main__":
    # Example usage
    frechet_copula = Frechet(alpha=0.55, beta=0)
    xi = frechet_copula.chatterjees_xi()
    ccop = frechet_copula.to_checkerboard()
    xi_ccop = ccop.chatterjees_xi()
    rho_ccop = ccop.spearmans_rho()
    print(
        f"Frechet Copula: xi = {xi}, Checkerboard xi = {xi_ccop}, Checkerboard rho = {rho_ccop}"
    )
    gamma = frechet_copula.ginis_gamma()
    ccop_gamma = ccop.ginis_gamma()
    footrule = frechet_copula.spearmans_footrule()
    ccop_footrule = ccop.spearmans_footrule()
    print(f"Gini's Gamma: {gamma}, Checkerboard Gini's Gamma: {ccop_gamma}")
    print(f"Footrule: {footrule}, Checkerboard Footrule: {ccop_footrule}")
    print("Done!")
