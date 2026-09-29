import sympy

from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.upper_frechet import UpperFrechet
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class B11(BivCopula):
    """
    B11 Copula - a special case of the Fréchet copula family.

    This is a convex combination of the upper Fréchet bound (min function)
    and the independence copula:
    C(u,v) = delta * min(u,v) + (1-delta) * u*v

    Parameters:
    -----------
    delta : float, 0 ≤ delta ≤ 1
        Mixing parameter that determines the weight of the upper Fréchet bound.
        delta = 0 gives the independence copula.
        delta = 1 gives the upper Fréchet bound.
    """

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def is_absolutely_continuous(self) -> bool:
        # the min(u, v) component puts mass delta on the diagonal
        return self.delta == 0

    def _numeric_callables(self):
        """Closed forms and exact mixture sampling of delta*M + (1-delta)*Pi."""
        from copul.family.frechet.frechet import fr_mixture_callables

        return fr_mixture_callables(float(self.delta), 0.0)

    # Define parameter
    delta = sympy.symbols("delta", nonnegative=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, 1, left_open=False, right_open=False)}

    def __init__(self, *args, **kwargs):
        """Initialize the B11 copula with parameter validation."""
        if args and len(args) == 1:
            kwargs["delta"] = args[0]

        if "delta" in kwargs:
            # Validate delta parameter
            delta_val = kwargs["delta"]
            if delta_val < 0 or delta_val > 1:
                raise ValueError(f"Parameter delta must be between 0 and 1, got {delta_val}")

        super().__init__(**kwargs)

    def __call__(self, **kwargs):
        """Handle special cases when calling the instance."""
        if "delta" in kwargs:
            # Validate delta parameter
            delta_val = kwargs["delta"]
            if delta_val < 0 or delta_val > 1:
                raise ValueError(f"Parameter delta must be between 0 and 1, got {delta_val}")

            # Special cases
            if delta_val == 0:
                del kwargs["delta"]
                return BivIndependenceCopula()(**kwargs)
            if delta_val == 1:
                del kwargs["delta"]
                return UpperFrechet()(**kwargs)

        return super().__call__(**kwargs)

    @property
    def _cdf_expr(self):
        return self.delta * sympy.Min(self.u, self.v) + (1 - self.delta) * self.u * self.v

    @property
    def cdf(self):
        """
        Cumulative distribution function of the copula.

        C(u,v) = delta * min(u,v) + (1-delta) * u*v
        """
        return SymPyFuncWrapper(self._cdf_expr)

    def spearmans_rho(self, *args, **kwargs):
        """
        Calculate Spearman's rho for the B11 copula.

        For B11, rho = delta
        """
        self._set_params(args, kwargs)
        return self.delta

    def kendalls_tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the B11 copula.

        For the Fréchet mixture delta*M + (1-delta)*Pi, tau = delta*(2 + delta)/3
        (Nelsen 2006, Example 5.5 with beta = 0).
        """
        self._set_params(args, kwargs)
        return self.delta * (2 + self.delta) / 3

    def lambda_U(self):
        """
        Upper tail dependence coefficient, lambda_U = delta.

        (1 - 2t + C(t, t)) / (1 - t) = delta + (1 - delta)(1 - t) -> delta.
        """
        return self.delta

    def lambda_L(self):
        """
        Lower tail dependence coefficient, lambda_L = delta.

        C(t, t) / t = delta + (1 - delta) t -> delta.
        """
        return self.delta
