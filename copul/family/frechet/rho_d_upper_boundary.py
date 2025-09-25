# upper_boundary.py
import sympy

from copul.family.frechet.frechet import Frechet


class RhoDUpperBoundary(Frechet):
    r"""
    One-parameter family attaining the *upper* boundary of the (rho, D) region:
        C = alpha * M + (1 - alpha) * W,   with   alpha = (1 + rho)/2.

    Hence the independence weight is zero and beta = 1 - alpha = (1 - rho)/2.
    This family satisfies Spearman's rho(C) = rho.

    Parameters
    ----------
    rho : sympy symbol or float in [-1, 1]
        Target Spearman's rho.

    Notes
    -----
    We implement this as a Frechet subclass but expose only the single
    parameter 'rho'. Internally:
        alpha(rho) = (1 + rho)/2,
        beta(rho)  = (1 - rho)/2.
    """

    _rho = sympy.symbols("rho", real=True)
    params = [_rho]
    intervals = {
        "rho": sympy.Interval(-1, 1, left_open=False, right_open=False),
    }

    def __init__(self, *args, **kwargs):
        # Allow positional or keyword initialization: RhoDUpperBoundary(rho=0.2) or (0.2)
        if args and len(args) == 1:
            kwargs["rho"] = args[0]
        if "rho" in kwargs:
            self._rho = kwargs["rho"]
            # Remove to avoid Frechet.__init__ trying to treat it as alpha/beta
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
        # alpha = (1 + rho)/2
        return (1 + self._rho) / 2

    @property
    def beta(self):
        # beta = (1 - rho)/2
        return (1 - self._rho) / 2

    @property
    def cdf(self):
        r"""
        C(u,v) = alpha * min(u,v) + (1 - alpha - beta) * u v + beta * max(u+v-1, 0)
               = alpha * M + beta * W   (since 1 - alpha - beta = 0 in this family).
        """
        frechet_upper = sympy.Min(self.u, self.v)
        frechet_lower = sympy.Max(self.u + self.v - 1, 0)
        a = self.alpha
        b = self.beta
        # Independence weight is identically zero.
        return self.CDFWrapper(a * frechet_upper + b * frechet_lower)


if __name__ == "__main__":
    copula = RhoDUpperBoundary()
