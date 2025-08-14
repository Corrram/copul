import copy

import sympy

from copul.exceptions import PropertyUnavailableException
from copul.families.core.biv_copula import BivCopula
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper


class Frechet(BivCopula):
    @property
    def is_symmetric(self) -> bool:
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
        frechet_upper = sympy.Min(self.u, self.v)
        frechet_lower = sympy.Max(self.u + self.v - 1, 0)
        cdf = (
            self._alpha * frechet_upper
            + (1 - self._alpha - self._beta) * self.u * self.v
            + self._beta * frechet_lower
        )
        return CDFWrapper(cdf)

    def cdf_vectorized(self, u, v):
        """
        Vectorized implementation of the cumulative distribution function for Frechet copula.

        This method evaluates the CDF at multiple points simultaneously, which is more efficient
        than calling the scalar CDF function repeatedly.

        Parameters
        ----------
        u : array_like
            First uniform marginal, should be in [0, 1].
        v : array_like
            Second uniform marginal, should be in [0, 1].

        Returns
        -------
        numpy.ndarray
            The CDF values at the specified points.

        Notes
        -----
        This implementation uses numpy for vectorized operations, providing significant
        performance improvements for large inputs. The formula used is:
            C(u,v) = α·min(u,v) + (1-α-β)·u·v + β·max(u+v-1,0)
        where α and β are the parameters of the Frechet copula.
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

    def rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return self._alpha - self._beta

    def tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return (self._alpha - self._beta) * (2 + self._alpha + self._beta) / 3

    @property
    def lambda_L(self):
        return self._alpha

    @property
    def lambda_U(self):
        return self._alpha

    def xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        return (self.alpha - self.beta) ** 2 + self.alpha * self.beta

    @property
    def pdf(self):
        raise PropertyUnavailableException("Frechet copula does not have a pdf")

    def footrule(self, *args, **kwargs):
        """
        Calculates Spearman's Footrule (psi) for the Frechet copula.

        The closed-form formula is psi = alpha - beta / 2.
        """
        self._set_params(args, kwargs)
        return self.alpha - self.beta / 2

    def ginis_gamma(self, *args, **kwargs):
        """
        Calculates Gini's Gamma (gamma) for the Frechet copula.

        The closed-form formula is gamma = alpha - beta, which is identical to
        Spearman's Rho for this family.
        """
        self._set_params(args, kwargs)
        return self.alpha - self.beta


# B11 = lambda: Frechet(beta=0)
if __name__ == "__main__":
    # Example usage
    frechet_copula = Frechet(alpha=0.55, beta=0)
    xi = frechet_copula.xi()
    ccop = frechet_copula.to_checkerboard()
    xi_ccop = ccop.xi()
    rho_ccop = ccop.rho()
    print(
        f"Frechet Copula: xi = {xi}, Checkerboard xi = {xi_ccop}, Checkerboard rho = {rho_ccop}"
    )
    gamma = frechet_copula.ginis_gamma()
    ccop_gamma = ccop.ginis_gamma()
    footrule = frechet_copula.footrule()
    ccop_footrule = ccop.footrule()
    print(f"Gini's Gamma: {gamma}, Checkerboard Gini's Gamma: {ccop_gamma}")
    print(f"Footrule: {footrule}, Checkerboard Footrule: {ccop_footrule}")
    print("Done!")
