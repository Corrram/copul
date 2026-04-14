import numpy as np
import sympy

from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.family.extreme_value.galambos import Galambos
from copul.family.extreme_value.gumbel_hougaard import (
    GumbelHougaardEV as GumbelHougaard,
)
from copul.family.frechet.upper_frechet import UpperFrechet


class BB5(BivExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    theta, delta = sympy.symbols("theta delta", positive=True)
    params = [theta, delta]
    intervals = {
        "theta": sympy.Interval(1, np.inf, left_open=False, right_open=True),
        "delta": sympy.Interval(0, np.inf, left_open=True, right_open=True),
    }

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 2:
            self.theta = args[0]
            self.delta = args[1]
        elif args:
            raise ValueError("BB5 copula requires two parameters")
        if "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            return Galambos(delta=self.delta)(**kwargs)
        elif "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return GumbelHougaard(self.theta)(**kwargs)
        elif "delta" in kwargs and kwargs["delta"] == sympy.oo:
            del kwargs["delta"]
            return UpperFrechet(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        theta = self.theta
        t = self.t
        return (
            t**theta
            + (1 - t) ** theta
            - ((1 - t) ** (-theta * self.delta) + t ** (-theta * self.delta))
            ** (-1 / self.delta)
        ) ** (1 / theta)

    @property
    def _cdf_expr(self):
        theta = self.theta
        u = self.u
        v = self.v
        delta = self.delta
        return sympy.exp(
            -(
                (
                    sympy.log(1 / v) ** theta
                    + sympy.log(1 / u) ** theta
                    - (
                        sympy.log(1 / u) ** (-delta * theta)
                        + sympy.log(1 / v) ** (-delta * theta)
                    )
                    ** (-1 / delta)
                )
                ** (1 / theta)
            )
        )

    # ------------------------------------------------------------------
    # Fast CDF: route keyword-arg calls to the vectorised implementation
    # so that axiom tests don't trigger slow SymPy evaluation.
    # ------------------------------------------------------------------

    def cdf(self, u=None, v=None, **kwargs):
        """Evaluate the CDF numerically via *cdf_vectorized*.

        The symbolic CDF of the BB5 copula contains deeply nested
        logarithms and powers that are prohibitively slow to evaluate
        through SymPy's ``evalf``.  This override routes all concrete
        (u, v) evaluations to the fast numpy path.
        """
        if u is None:
            u = kwargs.pop("u", None)
        if v is None:
            v = kwargs.pop("v", None)
        if u is None or v is None:
            raise TypeError("cdf() requires keyword arguments u and v")
        scalar_in = not hasattr(u, "__len__")
        result = self.cdf_vectorized(
            np.atleast_1d(np.asarray(u, dtype=float)),
            np.atleast_1d(np.asarray(v, dtype=float)),
        )
        return float(result[0]) if scalar_in else result

    # ------------------------------------------------------------------
    # Fast PDF: numerical mixed-partial via cdf_vectorized.
    # The symbolic Pickands for BB5 is far too complex for SymPy to
    # differentiate in reasonable time, so we use finite differences.
    # ------------------------------------------------------------------

    @property
    def pdf(self):
        """Numerical PDF via finite-difference on *cdf_vectorized*."""
        outer = self  # capture reference for closure

        class _BB5PDF:
            """Callable wrapper that behaves like SymPyFuncWrapper."""

            def __call__(self, u=None, v=None, **kw):
                if u is None:
                    u = kw.get("u")
                if v is None:
                    v = kw.get("v")
                if u is None or v is None:
                    raise TypeError("pdf() requires u and v")
                return outer._pdf_numerical(float(u), float(v))

            # Let the axiom test's `hasattr(result, "evalf")` return False
            # so it goes through the float() path.

        return _BB5PDF()

    def _pdf_numerical(self, u: float, v: float, h: float = 1e-5) -> float:
        """Mixed partial derivative ∂²C/∂u∂v via central differences."""
        if u <= 0 or v <= 0 or u >= 1 or v >= 1:
            return 0.0
        # Clamp to avoid stepping outside (0, 1)
        h = min(h, u / 2, v / 2, (1 - u) / 2, (1 - v) / 2)
        ua = np.array([u + h, u + h, u - h, u - h])
        va = np.array([v + h, v - h, v + h, v - h])
        c = self.cdf_vectorized(ua, va)
        return float((c[0] - c[1] - c[2] + c[3]) / (4.0 * h * h))
