import logging
import numpy as np

log = logging.getLogger(__name__)


class LTDVerifier:
    """
    Verifier for the Left-Tail-Decreasing (LTD) property of copulas.
    A copula C is LTD iff, for every v in (0,1),
        u ↦ C(u, v) / u               (0 < u < 1)
    is non-increasing.
    """

    def __init__(self):
        # nothing to configure at the moment
        pass

    # ------------------------------------------------------------------ #
    #  Public driver
    # ------------------------------------------------------------------ #
    def is_ltd(self, copul, range_min=None, range_max=None):
        """
        Check whether *all* members of a one-parameter copula family
        (or a single fixed copula) satisfy LTD.

        Parameters
        ----------
        copul : BivCopula (class *or* instance)
            The copula family or concrete copula to test.
        range_min, range_max : float, optional
            Parameter range to scan if ``copul`` is a family.

        Returns
        -------
        bool
            ``True``  – LTD holds for every parameter tested
            ``False`` – at least one parameter violates LTD
        """
        range_min = -10 if range_min is None else range_min
        range_max = 10 if range_max is None else range_max
        n_interpolate = 20  # grid on parameter axis
        grid = np.linspace(0.001, 0.999, 40)  # grid on (u,v)

        # ---------- 1)  If *no* parameter → check directly ------------- #
        try:
            param_name = str(copul.params[0])
        except (AttributeError, IndexError):
            return self._copula_is_ltd(copul, grid)

        # ---------- 2)  Otherwise scan the parameter range ------------- #
        interval = copul.intervals[param_name]
        p_min = float(max(interval.inf, range_min))
        p_max = float(min(interval.sup, range_max))
        if interval.left_open:
            p_min += 0.01
        if interval.right_open:
            p_max -= 0.01

        params = np.linspace(p_min, p_max, n_interpolate)
        is_ltd_final = True

        for p in params:
            C = copul(**{param_name: p})
            is_ltd = self._copula_is_ltd(C, grid)
            log.debug("param %s = %.4g → LTD %s", param_name, p, is_ltd)
            is_ltd_final &= is_ltd  # stop *only* if we find a counter-example
            if not is_ltd_final:
                break

        return is_ltd_final

    # ------------------------------------------------------------------ #
    #  Core routine for a *concrete* copula instance
    # ------------------------------------------------------------------ #
    def _copula_is_ltd(self, C, grid):
        """
        True  ↔  C is LTD on the given grid.
        """
        tol = 1e-10
        # Try the symbolic route first (much faster if available) --------
        try:
            C_expr = C.cdf.func  # SymPy expression
            u_sym, v_sym = C.u, C.v
            frac = C_expr / u_sym
            for v in grid:
                f_v = frac.subs(v_sym, v)
                for u1, u2 in zip(grid[:-1], grid[1:]):
                    if f_v.subs(u_sym, u1) < f_v.subs(u_sym, u2) - tol:
                        return False
        # … fall back to numeric evaluation ------------------------------
        except Exception:
            cdf = C.cdf  # SymPyFuncWrapper → callable
            for v in grid:
                prev = None
                for u in grid:
                    val = cdf(u, v) / u
                    if prev is not None and val > prev + tol:
                        return False
                    prev = val
        return True
