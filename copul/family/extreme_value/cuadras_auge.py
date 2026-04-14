import logging
import sympy as sp
from copul.exceptions import PropertyUnavailableException
from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.wrapper.cd1_wrapper import CD1Wrapper

log = logging.getLogger(__name__)


class CuadrasAuge(BivExtremeValueCopula):
    """
    Cuadras-Auge copula, special case of the Marshall-Olkin copula.
    """

    def __new__(cls, *args, **kwargs):
        # Handle delta value passed as positional argument
        if args and len(args) > 0:
            if args[0] == 0:
                # Copy kwargs without the delta parameter
                new_kwargs = kwargs.copy()
                return BivIndependenceCopula(**new_kwargs)
            elif args[0] == 1:
                # Copy kwargs without the delta parameter
                new_kwargs = kwargs.copy()
                return UpperFrechet(**new_kwargs)
            # Parameter validation for positional args
            elif args[0] < 0 or args[0] > 1:
                raise ValueError(f"delta parameter must be in [0,1], got {args[0]}")

        # Handle delta value passed as keyword argument
        elif "delta" in kwargs:
            if kwargs["delta"] == 0:
                # Copy kwargs without the delta parameter
                new_kwargs = kwargs.copy()
                del new_kwargs["delta"]
                return BivIndependenceCopula(**new_kwargs)
            elif kwargs["delta"] == 1:
                # Copy kwargs without the delta parameter
                new_kwargs = kwargs.copy()
                del new_kwargs["delta"]
                return UpperFrechet(**new_kwargs)
            # Parameter validation for kwargs
            elif kwargs["delta"] < 0 or kwargs["delta"] > 1:
                raise ValueError(
                    f"delta parameter must be in [0,1], got {kwargs['delta']}"
                )

        # If we get here, continue with normal initialization
        return super().__new__(cls)

    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sp.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sp.Interval(0, 1, left_open=False, right_open=False)}

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["delta"] = args[0]

        if "delta" in kwargs:
            # Parameter validation
            if kwargs["delta"] < 0 or kwargs["delta"] > 1:
                raise ValueError(
                    f"delta parameter must be in [0,1], got {kwargs['delta']}"
                )

            if kwargs["delta"] == 0:
                del kwargs["delta"]
                return BivIndependenceCopula()(**kwargs)

            if kwargs["delta"] == 1:
                del kwargs["delta"]
                return UpperFrechet()(**kwargs)

        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self):
        return self.delta == 0

    @property
    def _pickands(self):
        return 1 - self.delta * sp.Min(1 - self.t, self.t)

    @property
    def _cdf_expr(self):
        return sp.Min(self.u, self.v) ** self.delta * (self.u * self.v) ** (
            1 - self.delta
        )

    @property
    def pdf(self):
        raise PropertyUnavailableException("Cuadras-Auge copula does not have a pdf")

    def cond_distr_1(self, u=None, v=None):
        delta = self.delta
        cond_distr_1 = (
            self.v ** (1 - delta)
            * (
                delta * self.u * sp.Heaviside(-self.u + self.v)
                - delta * sp.Min(self.u, self.v)
                + sp.Min(self.u, self.v)
            )
            * sp.Min(self.u, self.v) ** (delta - 1)
            / self.u**delta
        )
        return CD1Wrapper(cond_distr_1)(u, v)

    def _squared_cond_distr_1(self, v, u):
        delta = self.delta
        func = (
            (u * v) ** (2 - 2 * delta)
            * (delta * u * sp.Heaviside(-u + v) - (delta - 1) * sp.Min(u, v)) ** 2
            * sp.Min(u, v) ** (2 * delta - 2)
            / u**2
        )
        return sp.simplify(func)

    def _xi_int_1(self, v):
        delta = self.delta
        u = self.u
        func_u_lower_v = (
            (u * v) ** (2 - 2 * delta)
            * (delta * u - (delta - 1) * u) ** 2
            * u ** (2 * delta - 2)
            / u**2
        )
        func_u_greater_v = (delta - 1) ** 2 * v**2 / u ** (2 * delta)
        int1 = sp.simplify(sp.integrate(func_u_lower_v, (u, 0, v)))
        # int2 = sp.simplify(sp.integrate(func_u_greater_v, (u, v, 1)))
        int2 = sp.integrate(func_u_greater_v, (u, v, 1))
        # int2 = -v**2*v**(1 - 2*delta)*(delta - 1)**2/(1 - 2*delta) + v**2*(delta - 1)**2/(1 - 2*delta)
        log.debug("sub int1 sp: ", int1)
        log.debug("sub int1: ", sp.latex(int1))
        log.debug("sub int2 sp: ", int2)
        log.debug("sub int2: ", sp.latex(int2))
        return sp.simplify(int1 + int2)

    def chatterjees_xi(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # Handle the edge case for delta=0
        if self.delta == 0:
            return 0
        return self.delta**2 / (2 - self.delta)

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # Handle the edge case for delta=0
        if self.delta == 0:
            return 0
        return 3 * self.delta / (4 - self.delta)

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        # Handle the edge case for delta=0
        if self.delta == 0:
            return 0
        return self.delta / (2 - self.delta)

    # ------------------------------------------------------------------
    # Additional dependence measures — closed forms
    # ------------------------------------------------------------------

    def blomqvists_beta(self, *args, **kwargs):
        r"""
        Blomqvist's :math:`\beta` for the Cuadras-Augé copula.

        .. math::

           \beta = 4\,C(\tfrac12,\tfrac12) - 1
                 = 4\cdot 2^{-\delta}\cdot 2^{-(2-2\delta)} - 1
                 = 2^{2-\delta} \cdot 2^{-2+2\delta}\cdot 4 - 1
                 \;=\; 2^{\delta} - 1
        """
        self._set_params(args, kwargs)
        d = float(self.delta)
        if d == 0:
            return 0
        return 2.0**d - 1.0

    def schweizer_wolff_sigma(self, *args, **kwargs):
        r"""
        Schweizer–Wolff :math:`\sigma` for the Cuadras-Augé copula.

        The CA copula is PQD for all :math:`\delta \in (0,1]`, so
        :math:`\sigma = \rho_S = 3\delta/(4-\delta)`.
        """
        self._set_params(args, kwargs)
        d = float(self.delta)
        if d == 0:
            return 0
        return 3 * d / (4 - d)

    def hoeffdings_d(self, *args, **kwargs):
        r"""
        Hoeffding's :math:`D` for the Cuadras-Augé copula.

        Expanding the double integral
        :math:`90\iint(C-uv)^2\,du\,dv` over the two regions
        :math:`\{v \le u\}` and :math:`\{u < v\}` (using symmetry) gives:

        .. math::

           D(\delta) = \frac{10\,\delta^{2}}
                            {18 - 9\,\delta + \delta^{2}}

        which satisfies :math:`D(0)=0` and :math:`D(1)=1`.
        """
        self._set_params(args, kwargs)
        d = float(self.delta)
        if d == 0:
            return 0
        return 10 * d**2 / (18 - 9 * d + d**2)

    def gini_gamma(self, *args, **kwargs):
        r"""
        Gini's :math:`\gamma` for the Cuadras-Augé copula.

        .. math::

           \gamma = 4\!\left[\int_0^1 C(t,t)\,dt
                     + \int_0^1 C(t,1{-}t)\,dt\right] - 2

        For the CA copula, :math:`C(t,t) = t^{2-\delta}` (since
        :math:`\min(t,t)=t`), so the diagonal integral is
        :math:`1/(3-\delta)`.
        The anti-diagonal :math:`C(t,1-t)` requires splitting at
        :math:`t=1/2`.
        """
        self._set_params(args, kwargs)
        d = float(self.delta)
        if d == 0:
            return 0
        # Diagonal: int_0^1 t^{2-delta} dt = 1/(3-delta)
        I_diag = 1.0 / (3 - d)
        # Anti-diagonal: C(t, 1-t) = min(t,1-t)^delta * (t(1-t))^{1-delta}
        # For t in [0,1/2]: min(t,1-t) = t, so C = t^delta * (t(1-t))^{1-d} = t * (1-t)^{1-d}
        # For t in [1/2,1]: min(t,1-t) = 1-t, so C = (1-t) * t^{1-d}
        # By substitution s=1-t, the two halves are equal.
        # I_anti = 2 * int_0^{1/2} t * (1-t)^{1-d} dt
        # = 2 * [B_{1/2}(2, 2-d)]  (incomplete beta)
        # We can compute via the regularised incomplete beta:
        from scipy.special import betainc as _betainc_reg
        from scipy.special import beta as _beta_fn

        I_anti = 2 * _betainc_reg(2, 2 - d, 0.5) * _beta_fn(2, 2 - d)
        return float(4 * (I_diag + I_anti) - 2)

    def spearman_footrule(self, *args, **kwargs):
        r"""
        Spearman's footrule :math:`\psi` for the Cuadras-Augé copula.

        .. math::

           \psi = 6\int_0^1 C(t,t)\,dt - 2
                = \frac{6}{3-\delta} - 2
                = \frac{2\delta}{3-\delta}
        """
        self._set_params(args, kwargs)
        d = float(self.delta)
        if d == 0:
            return 0
        return 2 * d / (3 - d)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    cop = CuadrasAuge()
    cop.delta = 0.5
    print(cop.spearmans_rho() ** 2 - cop.chatterjees_xi())
    print("Done!")
