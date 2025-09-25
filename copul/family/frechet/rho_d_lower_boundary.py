# lower_boundary.py
import sympy

from copul.family.frechet.frechet import Frechet


class RhoDLowerBoundary(Frechet):
    r"""
    One-parameter family attaining the *lower* boundary of the (rho, D) region:
        C = (1 - |rho|) * Pi + |rho| * B_rho,
    where B_rho = M if rho >= 0 and B_rho = W if rho < 0.

    This can be encoded as the Frechet mixture with:
        alpha(rho) = max(rho, 0),
        beta(rho)  = max(-rho, 0),
        independence weight = 1 - |rho|.

    Parameters
    ----------
    rho : sympy symbol or float in [-1, 1]
        Target Spearman's rho. For this family, rho(C) = rho.

    Notes
    -----
    We expose only the single parameter 'rho' and compute alpha, beta
    piecewise via sympy.Max to keep expressions symbolic when needed.
    """

    _rho = sympy.symbols("rho", real=True)
    params = [_rho]
    intervals = {
        "rho": sympy.Interval(-1, 1, left_open=False, right_open=False),
    }

    def __init__(self, *args, **kwargs):
        # Allow positional or keyword initialization: RhoDLowerBoundary(rho=-0.3) or (-0.3)
        if args and len(args) == 1:
            kwargs["rho"] = args[0]
        if "rho" in kwargs:
            self._rho = kwargs["rho"]
            del kwargs["rho"]
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        # Functional style: obj(rho=...) returns a cloned instance with the new parameter
        if args and len(args) == 1:
            kwargs["rho"] = args[0]
        if "rho" in kwargs:
            new = sympy.deepcopy(self) if hasattr(sympy, "deepcopy") else None
            if new is None:
                import copy as _copy

                new = _copy.deepcopy(self)
            new._rho = kwargs["rho"]
            del kwargs["rho"]
            return new.__call__(**kwargs)
        return super().__call__(**kwargs)

    @property
    def rho(self):
        return self._rho

    @property
    def alpha(self):
        # alpha = max(rho, 0)
        return sympy.Max(self._rho, 0)

    @property
    def beta(self):
        # beta = max(-rho, 0)
        return sympy.Max(-self._rho, 0)

    # Frechet.cdf already uses self._alpha/self._beta in the symbolic cdf,
    # so we override cdf to ensure it always uses the (piecewise) alpha/beta *properties*.
    @property
    def cdf(self):
        r"""
        C(u,v) = alpha * min(u,v) + (1 - alpha - beta) * u v + beta * max(u+v-1, 0),
        where 1 - alpha - beta = 1 - |rho|.
        """
        frechet_upper = sympy.Min(self.u, self.v)
        frechet_lower = sympy.Max(self.u + self.v - 1, 0)
        a = self.alpha
        b = self.beta
        independence = self.u * self.v
        return self.CDFWrapper(
            a * frechet_upper + (1 - a - b) * independence + b * frechet_lower
        )
