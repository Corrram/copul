import numpy as np
import sympy
from scipy.stats import multivariate_t
from scipy.stats import t as student_t

from copul.family.elliptical.elliptical_copula import EllipticalCopula
from copul.family.other import LowerFrechet, UpperFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper
from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class StudentT(EllipticalCopula):
    """
    Student's t Copula implementation.

    The Student's t copula is an elliptical copula derived from the multivariate t-distribution.
    It is characterized by a correlation parameter rho in [-1, 1] and a degrees of freedom
    parameter nu > 0.

    Special cases:
    - rho = -1: Lower Fréchet bound (countermonotonicity)
    - rho = 1: Upper Fréchet bound (comonotonicity)
    - nu → ∞: Approaches the Gaussian copula
    """

    @property
    def is_symmetric(self) -> bool:
        return True

    rho = sympy.symbols("rho")
    nu = sympy.symbols("nu", positive=True)
    modified_bessel_function = sympy.Function("K")(nu)
    gamma_function = sympy.Function("gamma")(nu / 2)
    params = [rho, nu]
    intervals = {
        "rho": sympy.Interval(-1, 1, left_open=False, right_open=False),
        "nu": sympy.Interval(0, sympy.oo, left_open=True, right_open=True),
    }

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 1:
            kwargs["rho"] = args[0]
        if args is not None and len(args) == 2:
            kwargs["rho"] = args[0]
            kwargs["nu"] = args[1]

        if "rho" in kwargs:
            # Handle special cases
            if kwargs["rho"] == -1:
                # Don't pass 'nu' parameter to LowerFrechet
                new_kwargs = kwargs.copy()
                if "nu" in new_kwargs:
                    del new_kwargs["nu"]
                if "rho" in new_kwargs:
                    del new_kwargs["rho"]
                return LowerFrechet()(**new_kwargs)
            elif kwargs["rho"] == 1:
                # Don't pass 'nu' parameter to UpperFrechet
                new_kwargs = kwargs.copy()
                if "nu" in new_kwargs:
                    del new_kwargs["nu"]
                if "rho" in new_kwargs:
                    del new_kwargs["rho"]
                return UpperFrechet()(**new_kwargs)

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def rvs(self, n=1, **kwargs):
        """
        Generate random samples from the Student's t copula.

        Args:
            n (int): Number of samples to generate

        Returns:
            numpy.ndarray: Array of shape (n, 2) containing the samples
        """
        from statsmodels.distributions.copula.elliptical import StudentTCopula

        return StudentTCopula(self.rho, df=self.nu).rvs(n)

    def _calculate_student_t_cdf(self, u, v, rho_val, nu_val):
        """Calculate Student's t CDF at point (u, v)."""
        if u <= 0 or v <= 0:
            return 0
        if u >= 1:
            return v
        if v >= 1:
            return u

        mvt = multivariate_t(df=nu_val, shape=[[1, rho_val], [rho_val, 1]])
        z_u = student_t.ppf(u, nu_val)
        z_v = student_t.ppf(v, nu_val)
        return mvt.cdf([z_u, z_v])

    @property
    def cdf(self):
        """
        Compute the cumulative distribution function of the Student's t copula.

        Returns:
            callable: Function that computes the CDF at given points
        """
        # Store the parameters to avoid capturing 'self' in the lambda
        rho_val = self.rho
        nu_val = self.nu

        # Use a reference to the method, not self
        cdf_calc = self._calculate_student_t_cdf

        def student_t_copula_cdf(u, v):
            return cdf_calc(u, v, rho_val, nu_val)

        return lambda u, v: CDFWrapper(sympy.S(student_t_copula_cdf(u, v)))

    def _conditional_distribution(self, u, v):
        """
        Compute the conditional distribution function of the Student's t copula.

        Args:
            u (float, optional): First marginal value
            v (float, optional): Second marginal value

        Returns:
            callable or sympy.Expr: Conditional distribution function or value
        """

        def conditional_func(primary, secondary):
            # Y | X=x  ~  rho*x + sqrt((nu+x^2)(1-rho^2)/(nu+1)) * t_{nu+1}
            x = student_t.ppf(primary, self.nu)
            cdf = student_t.cdf(
                student_t.ppf(secondary, self.nu),
                self.nu + 1,
                loc=self.rho * x,
                scale=((1 - self.rho**2) * (self.nu + x**2) / (self.nu + 1)) ** 0.5,
            )
            if isinstance(cdf, float):
                return sympy.S(cdf)
            return sympy.S(cdf(u, v))

        if u is None and v is None:
            return conditional_func
        elif u is not None and v is not None:
            return conditional_func(u, v)
        elif u is not None:
            return lambda v_: conditional_func(u, v_)
        else:
            return lambda u_: conditional_func(u_, v)

    def cond_distr_1(self, u=None, v=None):
        """
        Compute the first conditional distribution C(v|u).

        Args:
            u (float, optional): Conditioning value
            v (float, optional): Value at which to evaluate

        Returns:
            CD1Wrapper: Wrapped conditional distribution function or value
        """
        if v in [0, 1]:
            return CD1Wrapper(sympy.S(v))
        cd1 = self._conditional_distribution(u, v)
        return CD1Wrapper(cd1)

    def cond_distr_2(self, u=None, v=None):
        """
        Compute the second conditional distribution C(u|v).

        Args:
            u (float, optional): Value at which to evaluate
            v (float, optional): Conditioning value

        Returns:
            CD2Wrapper: Wrapped conditional distribution function or value
        """
        if u in [0, 1]:
            return CD2Wrapper(sympy.S(u))
        cd2 = self._conditional_distribution(v, u)
        return CD2Wrapper(cd2)

    @property
    def pdf(self):
        """
        Compute the probability density function of the Student's t copula.

        Returns:
            callable: Function that computes the PDF at given points
        """
        from statsmodels.distributions.copula.elliptical import StudentTCopula

        return lambda u, v: SymPyFuncWrapper(
            sympy.S(StudentTCopula(self.rho, df=self.nu).pdf([u, v]))
        )

    def _numeric_callables(self):
        r"""Vectorized ingredients for the measures engine.

        With :math:`x=t_\nu^{-1}(u)`, :math:`y=t_\nu^{-1}(v)`,

        .. math::

           \partial_1 C(u,v) = t_{\nu+1}\!\Bigl(\frac{y-\rho x}
               {\sqrt{(\nu+x^2)(1-\rho^2)/(\nu+1)}}\Bigr),

        the density is the ratio of the bivariate and univariate t
        densities, and :math:`C(u,v)=\int_0^u\partial_1C(s,v)\,ds` is evaluated
        by 48-point Gauss--Legendre quadrature after the substitution
        :math:`s=u\,w^4` (which removes the algebraic endpoint behaviour).
        The engine prefers the h-function formulas for rho and nu.
        """
        from scipy.special import gammaln, stdtr, stdtrit

        r = float(self.rho)
        nu = float(self.nu)
        s = np.sqrt(1.0 - r * r)

        def h(a, b):
            x = stdtrit(nu, a)
            y = stdtrit(nu, b)
            return stdtr(nu + 1.0, (y - r * x) / (s * np.sqrt((nu + x * x) / (nu + 1.0))))

        def h1(u, v):
            return h(u, v)

        def h2(u, v):
            return h(v, u)

        logk = gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2)

        def pdf(u, v):
            x = stdtrit(nu, u)
            y = stdtrit(nu, v)
            q = (x * x - 2 * r * x * y + y * y) / (nu * (1 - r * r))
            lg = (
                logk
                - np.log(s)
                - (nu + 2) / 2 * np.log1p(q)
                + (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu))
            )
            return np.exp(lg)

        gx, gw = np.polynomial.legendre.leggauss(48)
        gx = 0.5 * (gx + 1.0)
        gw = 0.5 * gw
        w4 = gx**4
        jac = 4 * gx**3 * gw

        def cdf(u, v):
            u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
            shape = u.shape
            uf, vf = u.ravel(), v.ravel()
            out = np.empty(uf.size)
            step = 20_000
            for i in range(0, uf.size, step):
                uu = uf[i : i + step, None]
                vv = vf[i : i + step, None]
                vals = h(uu * w4[None, :], np.broadcast_to(vv, (vv.shape[0], w4.size)))
                out[i : i + step] = uu[:, 0] * (vals @ jac)
            return out.reshape(shape)

        return {"cdf": cdf, "h1": h1, "h2": h2, "pdf": pdf, "prefer_h": True}

    # ------------------------------------------------------------------
    # Analytical dependence measures
    # ------------------------------------------------------------------

    def lambda_L(self):
        r"""Lower tail dependence coefficient for the Student-t copula.

        .. math::

           \lambda_L = 2\,t_{\nu+1}\!\left(
               -\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}
           \right)

        where :math:`t_{\nu+1}` is the CDF of the univariate Student-t
        distribution with :math:`\nu + 1` degrees of freedom.

        Returns
        -------
        float
            Lower tail dependence coefficient in :math:`[0, 1]`.

        References
        ----------
        Demarta & McNeil (2005), *The t Copula and Related Copulas*,
        International Statistical Review 73(1), 111--129.
        """
        rho_val = float(self.rho)
        nu_val = float(self.nu)
        arg = -np.sqrt((nu_val + 1.0) * (1.0 - rho_val) / (1.0 + rho_val))
        return 2.0 * student_t.cdf(arg, df=nu_val + 1.0)

    def lambda_U(self):
        r"""Upper tail dependence coefficient for the Student-t copula.

        The Student-t copula is radially symmetric, so
        :math:`\lambda_U = \lambda_L`.

        Returns
        -------
        float
            Upper tail dependence coefficient in :math:`[0, 1]`.
        """
        return self.lambda_L()

    def tail_order(self):
        r"""Tail order for the Student-t copula.

        Since :math:`\lambda_L = \lambda_U > 0` for all
        :math:`\rho \in (-1, 1]` and :math:`\nu < \infty`, the tail order is
        :math:`\kappa_L = \kappa_U = 1`, see Hua & Joe (2011), JMVA 102.

        Returns
        -------
        dict
        """
        if float(self.rho) <= -1.0:
            return {"lower": float("inf"), "upper": float("inf")}
        return {"lower": 1.0, "upper": 1.0}

    def kendalls_tau(self, *args, **kwargs):
        r"""Kendall's :math:`\tau` for the Student-t copula.

        .. math::

           \tau = \frac{2}{\pi}\,\arcsin(\rho)

        (same formula as the Gaussian copula — independent of :math:`\nu`).

        Returns
        -------
        float
        """
        self._set_params(args, kwargs)
        rho_val = float(self.rho)
        return (2.0 / np.pi) * np.arcsin(rho_val)

    def blests_nu(self, *args, **kwargs):
        r"""Blest's rank correlation :math:`\nu` for the Student-t copula.

        The Student-t copula is radially symmetric and
        :math:`\nu(C)+\nu(\hat C)=2\rho_S(C)` for any copula, so
        :math:`\nu = \rho_S`.
        """
        return self.spearmans_rho(*args, **kwargs)

    def blomqvists_beta(self, *args, **kwargs):
        r"""Blomqvist's :math:`\beta` for the Student-t copula.

        .. math::

           \beta = \frac{2}{\pi}\,\arcsin(\rho)

        (same formula as the Gaussian copula).

        Returns
        -------
        float
        """
        self._set_params(args, kwargs)
        rho_val = float(self.rho)
        return (2.0 / np.pi) * np.arcsin(rho_val)

    def tail_dependence_function(self, t, lower=True):
        r"""Evaluate the tail dependence function at :math:`t \in [0,1]`.

        For the Student-t copula:

        .. math::

           b_L(t) = (1-t)\,t_{\nu+1}\!\left(
                    -\sqrt{\frac{(\nu+1)(1 - \rho_{t})}{1 + \rho_{t}}}
                    \right)
                  + t\,t_{\nu+1}\!\left(
                    -\sqrt{\frac{(\nu+1)(1 - \tilde\rho_{t})}{1 + \tilde\rho_{t}}}
                    \right)

        where the mixed-quantile correlations involve the parameter.
        The simple diagonal case is :math:`b_L(1/2) = \lambda_L / 2`.

        Parameters
        ----------
        t : float or array_like
            Point(s) in :math:`[0, 1]`.
        lower : bool
            If True, evaluate the lower TDF. If False, upper TDF.

        Returns
        -------
        float or numpy.ndarray
        """
        # For the symmetric Student-t copula, the full bivariate TDF has a
        # closed form.  We use the numerically stable diagonal approach:
        #   b(t) = lim_{s→0+} C(s·(1-t), s·t) / s         (lower)
        #   b(t) = lim_{s→0+} Ĉ(s·(1-t), s·t) / s         (upper)
        # Evaluated via the R(t) representation using the Pickands-like
        # decomposition.  For the t-copula, the full analytical form is
        # non-trivial, so we use a stable numerical approximation.
        t = np.asarray(t, dtype=float)
        rho_val = float(self.rho)
        nu_val = float(self.nu)

        eps = 1e-7
        out = np.empty_like(t)

        for i in np.ndindex(t.shape):
            ti = t[i]
            if ti <= 0 or ti >= 1:
                out[i] = 0.0
            else:
                u_s = eps * (1 - ti)
                v_s = eps * ti
                if lower:
                    c_val = self._calculate_student_t_cdf(u_s, v_s, rho_val, nu_val)
                else:
                    c_val = (
                        u_s
                        + v_s
                        - 1
                        + self._calculate_student_t_cdf(1 - u_s, 1 - v_s, rho_val, nu_val)
                    )
                out[i] = c_val / eps

        if out.ndim == 0:
            return float(out)
        return out
