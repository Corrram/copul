import numpy as np
import sympy

from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class Joe(BivArchimedeanCopula):
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)
    special_cases = {1: BivIndependenceCopula}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _raw_generator(self):
        return -sympy.log(1 - (1 - self.t) ** self.theta)

    @property
    def _raw_inv_generator(self):
        return 1 - (1 - sympy.exp(-self.y)) ** (1 / self.theta)

    @property
    def _cdf_expr(self):
        theta = self.theta
        return 1 - (-((1 - self.u) ** theta - 1) * ((1 - self.v) ** theta - 1) + 1) ** (1 / theta)

    def _numeric_callables(self):
        r"""Closed forms for the numerical API.

        With :math:`a=(1-u)^\theta`, :math:`b=(1-v)^\theta`, :math:`s=a+b-ab`:
        :math:`C=1-s^{1/\theta}`,
        :math:`\partial_1C=(1-u)^{\theta-1}(1-b)s^{1/\theta-1}`,
        :math:`c=s^{1/\theta-2}(1-u)^{\theta-1}(1-v)^{\theta-1}(\theta-1+s)`;
        exact sampling with a Sibuya(1/θ) frailty (Marshall--Olkin).
        """
        from copul.family.archimedean import _frailty

        th = float(self.theta)
        alpha = 1.0 / th

        def parts(u, v):
            a = (1.0 - u) ** th
            b = (1.0 - v) ** th
            return a, b, a + b - a * b

        def cdf(u, v):
            *_, s = parts(u, v)
            return 1.0 - s**alpha

        def _h(x, y):
            _, b, s = parts(x, y)
            return (1.0 - x) ** (th - 1.0) * (1.0 - b) * s ** (alpha - 1.0)

        def logpdf(u, v):
            *_, s = parts(u, v)
            return (
                (alpha - 2.0) * np.log(s)
                + (th - 1.0) * (np.log1p(-u) + np.log1p(-v))
                + np.log(th - 1.0 + s)
            )

        def pdf(u, v):
            return np.exp(logpdf(u, v))

        def rvs(n, rng):
            frailty = _frailty.sibuya(n, rng, alpha)
            return _frailty.marshall_olkin(lambda t: 1.0 - (-np.expm1(-t)) ** alpha, frailty, rng)

        return {
            "cdf": cdf,
            "h1": _h,
            "h2": lambda u, v: _h(v, u),
            "pdf": pdf,
            "logpdf": logpdf,
            "rvs": rvs,
        }

    def cdf_vectorized(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Vectorized implementation of the cumulative distribution function for the Joe copula.

        This method uses the explicit mathematical formula for the Joe copula, which is
        significantly faster than the generic generator-based approach.

        Parameters
        ----------
        u : array_like
            First uniform marginal, must be in [0, 1].
        v : array_like
            Second uniform marginal, must be in [0, 1].

        Returns
        -------
        numpy.ndarray
            The CDF values at the specified points.
        """
        theta_val = float(self.theta)

        # Handle the independence case
        if np.isclose(theta_val, 1):
            return np.asarray(u) * np.asarray(v)

        # Convert inputs to numpy arrays for vectorized operations
        u = np.asarray(u)
        v = np.asarray(v)

        # Initialize result array. The default of 0 correctly handles C(0,v) and C(u,0).
        result = np.zeros_like(u, dtype=float)

        # Identify points that require the full computation (not on the boundaries)
        interior_mask = (u > 0) & (u < 1) & (v > 0) & (v < 1)

        if np.any(interior_mask):
            u_int, v_int = u[interior_mask], v[interior_mask]

            # Use the standard formula for the Joe copula for better numerical stability
            # C(u,v) = 1 - [ (1-u)^θ + (1-v)^θ - (1-u)^θ * (1-v)^θ ]^(1/θ)
            term_u = (1 - u_int) ** theta_val
            term_v = (1 - v_int) ** theta_val
            base = term_u + term_v - term_u * term_v
            result[interior_mask] = 1 - base ** (1 / theta_val)

        # Handle the u=1 and v=1 boundary cases using the mask
        result[u == 1] = v[u == 1]
        result[v == 1] = u[v == 1]

        return result

    def cond_distr_1(self, u=None, v=None):
        theta = self.theta
        cond_distr_1 = (
            -((1 - self.u) ** theta)
            * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1) ** (1 / theta)
            * ((1 - self.v) ** theta - 1)
            / ((1 - self.u) * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1))
        )
        return SymPyFuncWrapper(cond_distr_1)(u, v)

    def cond_distr_2(self, u=None, v=None):
        theta = self.theta
        cond_distr_2 = (
            (1 - self.v) ** theta
            * (1 - (1 - self.u) ** theta)
            * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1) ** (1 / theta)
            / ((1 - self.v) * ((1 - (1 - self.u) ** theta) * ((1 - self.v) ** theta - 1) + 1))
        )
        return SymPyFuncWrapper(cond_distr_2)(u, v)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)


Nelsen6 = Joe

# B5 = Joe


if __name__ == "__main__":
    copula = Nelsen6(theta=2)
    print(copula.rvs(5))
    for _i in range(1000):
        copula.rvs(1, approximate=False)
    print(copula.cdf(0.5, 0.5))
    print(copula.cond_distr_1(0.5, 0.5))
    print(copula.cond_distr_2(0.5, 0.5))
    print(copula.lambda_L())
    print(copula.lambda_U())
