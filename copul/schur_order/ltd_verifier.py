import logging

import numpy as np

log = logging.getLogger(__name__)


class LTDVerifier:
    r"""Verifier for left/right tail monotonicity properties of copulas.

    A copula :math:`C` is LTD iff, for every :math:`v\in(0,1)`, the mapping

    .. math::

       u \mapsto \frac{C(u,v)}{u}, \quad 0<u<1,

    is non-increasing in :math:`u`. LTI uses the same ratio with the opposite
    monotonicity. RTI/RTD use the upper-tail conditional probability

    .. math::

       u \mapsto \frac{1-u-v+C(u,v)}{1-u}, \quad 0<u<1.
    """

    def __init__(self):
        # Nothing to configure at the moment.
        pass

    def is_ltd(self, copul, range_min=None, range_max=None):
        r"""Check whether a copula satisfies the left-tail-decreasing property."""
        return self._check_property(copul, self._copula_is_ltd, range_min, range_max)

    def is_lti(self, copul, range_min=None, range_max=None):
        r"""Check whether a copula satisfies the left-tail-increasing property."""
        return self._check_property(copul, self._copula_is_lti, range_min, range_max)

    def is_rti(self, copul, range_min=None, range_max=None):
        r"""Check whether a copula satisfies the right-tail-increasing property."""
        return self._check_property(copul, self._copula_is_rti, range_min, range_max)

    def is_rtd(self, copul, range_min=None, range_max=None):
        r"""Check whether a copula satisfies the right-tail-decreasing property."""
        return self._check_property(copul, self._copula_is_rtd, range_min, range_max)

    def _check_property(self, copul, check_func, range_min=None, range_max=None):
        range_min = -10 if range_min is None else range_min
        range_max = 10 if range_max is None else range_max
        n_interpolate = 20  # grid on parameter axis
        grid = np.linspace(0.001, 0.999, 40)  # grid on (u,v)

        try:
            param_name = str(copul.params[0])
        except (AttributeError, IndexError, TypeError):
            return check_func(copul, grid)

        interval = copul.intervals[param_name]
        p_min = float(max(interval.inf, range_min))
        p_max = float(min(interval.sup, range_max))
        if interval.left_open:
            p_min += 0.01
        if interval.right_open:
            p_max -= 0.01

        for p in np.linspace(p_min, p_max, n_interpolate):
            C = copul(**{param_name: p})
            holds = check_func(C, grid)
            log.debug("param %s = %.4g -> %s", param_name, p, holds)
            if not holds:
                return False

        return True

    @staticmethod
    def _exact(C, kind):
        """Exact check for checkerboard copulas (``None`` if not applicable).

        For checkerboards ``u -> C(u, v)`` is piecewise linear with affine
        dependence of the pieces on the cell-local ``v``-coordinate, so the
        tail monotonicity reduces to finitely many sign conditions, see
        :func:`copul.checkerboard._biv_engine.tail_monotonicity`.
        """
        from copul.checkerboard._biv_mixin import BivCheckerboardMixin

        if isinstance(C, BivCheckerboardMixin):
            from copul.checkerboard import _biv_engine as eng

            return eng.tail_monotonicity(C.matr, C._kernel_signs(), kind)
        return None

    @staticmethod
    def _engine(C, key):
        """:func:`copul.theory.dependence.check_property` for fully specified copulas."""
        from copul.theory.dependence import _is_specified_bivariate, check_property

        if _is_specified_bivariate(C):
            return bool(check_property(C, key, i=1))
        return None

    def _copula_is_ltd(self, C, grid):
        exact = self._exact(C, "ltd")
        if exact is not None:
            return exact
        engine = self._engine(C, "LTD")
        if engine is not None:
            return engine
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: expr / u,
            numeric_ratio=lambda cdf, u, v: max(float(cdf(u, v)), 0.0) / u,
            increasing=False,
        )

    def _copula_is_lti(self, C, grid):
        exact = self._exact(C, "lti")
        if exact is not None:
            return exact
        engine = self._engine(C, "LTI")
        if engine is not None:
            return engine
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: expr / u,
            numeric_ratio=lambda cdf, u, v: max(float(cdf(u, v)), 0.0) / u,
            increasing=True,
        )

    def _copula_is_rti(self, C, grid):
        exact = self._exact(C, "rti")
        if exact is not None:
            return exact
        engine = self._engine(C, "RTI")
        if engine is not None:
            return engine
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: (1 - u - v + expr) / (1 - u),
            numeric_ratio=lambda cdf, u, v: (1 - u - v + max(float(cdf(u, v)), 0.0)) / (1 - u),
            increasing=True,
        )

    def _copula_is_rtd(self, C, grid):
        exact = self._exact(C, "rtd")
        if exact is not None:
            return exact
        engine = self._engine(C, "RTD")
        if engine is not None:
            return engine
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: (1 - u - v + expr) / (1 - u),
            numeric_ratio=lambda cdf, u, v: (1 - u - v + max(float(cdf(u, v)), 0.0)) / (1 - u),
            increasing=False,
        )

    def _check_monotone_ratio(self, C, grid, symbolic_ratio, numeric_ratio, increasing):
        tol = 1e-10

        # Evaluate the (clipped) cdf numerically; copula cdfs of families
        # defined with a positive part may evaluate to negative values if the
        # stored expression lacks the clipping, so we clip at 0.
        cdf = C.cdf
        for v in grid:
            values = (numeric_ratio(cdf, u, v) for u in grid)
            if not self._values_are_monotone(values, increasing, tol):
                return False

        return True

    @staticmethod
    def _values_are_monotone(values, increasing, tol):
        prev = None
        for val in values:
            if prev is not None:
                if increasing and val < prev - tol:
                    return False
                if not increasing and val > prev + tol:
                    return False
            prev = val
        return True
