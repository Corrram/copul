import copy

import numpy as np
import sympy

from copul.family.extreme_value.biv_extreme_value_copula import BivExtremeValueCopula
from copul.family.extreme_value.galambos import Galambos
from copul.family.extreme_value.marshall_olkin import MarshallOlkin
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula


class JoeEV(BivExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return self.alpha_1 == self.alpha_2

    alpha_1, alpha_2 = sympy.symbols("alpha_1 alpha_2", nonnegative=True)
    delta = sympy.symbols("delta", positive=True)
    params = [alpha_1, alpha_2, delta]

    intervals = {
        "alpha_1": sympy.Interval(0, 1, left_open=False, right_open=False),
        "alpha_2": sympy.Interval(0, 1, left_open=False, right_open=False),
        "delta": sympy.Interval(0, np.inf, left_open=True, right_open=True),
    }

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) == 3:
            self.alpha_1 = args[0]
            self.alpha_2 = args[1]
            self.delta = args[2]
        elif args:
            raise ValueError("Tawn copula requires three parameters")
        if (
            "alpha_1" in kwargs
            and kwargs["alpha_1"] == 1
            and "alpha_2" in kwargs
            and kwargs["alpha_2"] == 1
        ):
            del kwargs["alpha_1"]
            del kwargs["alpha_2"]
            return Galambos(**kwargs)
        elif "alpha_1" in kwargs and kwargs["alpha_1"] == 1:
            del kwargs["alpha_1"]
            if self.alpha_2 == 1:
                if "alpha_2" in kwargs:
                    del kwargs["alpha_2"]
                return Galambos()(**kwargs)
            new_copula = copy.deepcopy(self)
            new_copula.alpha_1 = 1
            return new_copula(**kwargs)
        elif "alpha_2" in kwargs and kwargs["alpha_2"] == 1:
            del kwargs["alpha_2"]
            if self.alpha_1 == 1:
                if "alpha_1" in kwargs:
                    del kwargs["alpha_1"]
                return Galambos()(**kwargs)
            new_copula = copy.deepcopy(self)
            new_copula.alpha_2 = 1
            return new_copula(**kwargs)
        elif "alpha_1" in kwargs and kwargs["alpha_1"] == 0:
            del kwargs["alpha_1"]
            if "alpha_2" in kwargs:
                del kwargs["alpha_2"]
            if "delta" in kwargs:
                del kwargs["delta"]
            return BivIndependenceCopula()(**kwargs)
        elif "alpha_2" in kwargs and kwargs["alpha_2"] == 0:
            del kwargs["alpha_2"]
            if "alpha_1" in kwargs:
                del kwargs["alpha_1"]
            if "delta" in kwargs:
                del kwargs["delta"]
            return BivIndependenceCopula()(**kwargs)
        elif "delta" in kwargs and kwargs["delta"] == sympy.oo:
            del kwargs["delta"]
            if "alpha_1" in kwargs:
                alpha1 = kwargs["alpha_1"]
                del kwargs["alpha_1"]
            else:
                alpha1 = self.alpha_1
            if "alpha_2" in kwargs:
                alpha2 = kwargs["alpha_2"]
                del kwargs["alpha_2"]
            else:
                alpha2 = self.alpha_2
            return MarshallOlkin(**kwargs)(alpha_1=alpha1, alpha_2=alpha2)
        elif "delta" in kwargs and kwargs["delta"] == 0:
            del kwargs["delta"]
            return BivIndependenceCopula()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def _pickands(self):
        delta = self.delta
        t = self.t
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        return 1 - ((alpha_1 * (1 - t)) ** (-delta) + (alpha_2 * t) ** (-delta)) ** (
            -1 / delta
        )

    @property
    def _cdf_expr(self):
        return (
            self.u
            * self.v
            * sympy.exp(
                (
                    (self.alpha_1 * sympy.log(1 / self.u)) ** (-self.delta)
                    + (self.alpha_2 * sympy.log(1 / self.v)) ** (-self.delta)
                )
                ** (-1 / self.delta)
            )
        )

    # ------------------------------------------------------------------
    # Fast CDF: route keyword-arg calls to the vectorised implementation
    # so that axiom tests don't trigger slow SymPy evaluation.
    # ------------------------------------------------------------------

    def cdf(self, u=None, v=None, **kwargs):
        """Evaluate the CDF numerically via *cdf_vectorized*.

        The symbolic CDF of the JoeEV copula contains deeply nested
        logarithms and powers that are slow to evaluate through SymPy's
        ``evalf``.  This override routes all concrete (u, v) evaluations
        to the fast numpy path.
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
    # ------------------------------------------------------------------

    @property
    def pdf(self):
        """Numerical PDF via finite-difference on *cdf_vectorized*."""
        outer = self

        class _JoeEVPDF:
            """Callable wrapper that behaves like SymPyFuncWrapper."""

            def __call__(self, u=None, v=None, **kw):
                if u is None:
                    u = kw.get("u")
                if v is None:
                    v = kw.get("v")
                if u is None or v is None:
                    raise TypeError("pdf() requires u and v")
                return outer._pdf_numerical(float(u), float(v))

        return _JoeEVPDF()

    def _pdf_numerical(self, u: float, v: float, h: float = 1e-5) -> float:
        """Mixed partial derivative ∂²C/∂u∂v via central differences."""
        if u <= 0 or v <= 0 or u >= 1 or v >= 1:
            return 0.0
        h = min(h, u / 2, v / 2, (1 - u) / 2, (1 - v) / 2)
        ua = np.array([u + h, u + h, u - h, u - h])
        va = np.array([v + h, v - h, v + h, v - h])
        c = self.cdf_vectorized(ua, va)
        return float((c[0] - c[1] - c[2] + c[3]) / (4.0 * h * h))

    # @property
    # def pdf(self):  # original placeholder
    #     u = self.u
    #     v = self.v
    #     result = None
    #     return SymPyFunctionWrapper(result)
