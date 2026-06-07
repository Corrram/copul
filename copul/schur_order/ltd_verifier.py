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
        except (AttributeError, IndexError):
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

    def _copula_is_ltd(self, C, grid):
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: expr / u,
            numeric_ratio=lambda cdf, u, v: cdf(u, v) / u,
            increasing=False,
        )

    def _copula_is_lti(self, C, grid):
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: expr / u,
            numeric_ratio=lambda cdf, u, v: cdf(u, v) / u,
            increasing=True,
        )

    def _copula_is_rti(self, C, grid):
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: (1 - u - v + expr) / (1 - u),
            numeric_ratio=lambda cdf, u, v: (1 - u - v + cdf(u, v)) / (1 - u),
            increasing=True,
        )

    def _copula_is_rtd(self, C, grid):
        return self._check_monotone_ratio(
            C,
            grid,
            symbolic_ratio=lambda expr, u, v: (1 - u - v + expr) / (1 - u),
            numeric_ratio=lambda cdf, u, v: (1 - u - v + cdf(u, v)) / (1 - u),
            increasing=False,
        )

    def _check_monotone_ratio(self, C, grid, symbolic_ratio, numeric_ratio, increasing):
        tol = 1e-10

        try:
            C_expr = C.cdf.func
            u_sym, v_sym = C.u, C.v
            ratio = symbolic_ratio(C_expr, u_sym, v_sym)
            for v in grid:
                f_v = ratio.subs(v_sym, v)
                values = (f_v.subs(u_sym, u) for u in grid)
                if not self._values_are_monotone(values, increasing, tol):
                    return False
        except Exception:
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
