import logging
import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Positive-Lower-Orthant-Dependence (PLOD) verifier
# --------------------------------------------------------------------------- #
class PLODVerifier:
    """
    Verifier for the Positive-Lower-Orthant-Dependence (PLOD) property.
    A copula C is PLOD iff
        C(u, v)  ≥  u · v            for all 0 < u, v < 1.
    """

    def __init__(self):
        # nothing to configure
        pass

    # ------------------------------------------------------------------ #
    #  Public driver
    # ------------------------------------------------------------------ #
    def is_plod(self, copul, range_min=None, range_max=None):
        """
        Check whether *all* members of a one-parameter copula family
        (or a single fixed copula) satisfy PLOD.

        Parameters
        ----------
        copul : BivCopula (class *or* instance)
            The copula family or concrete copula to test.
        range_min, range_max : float, optional
            Parameter range to scan if ``copul`` is a family.

        Returns
        -------
        bool
            ``True``  – PLOD holds for every parameter tested
            ``False`` – at least one parameter violates PLOD
        """
        range_min = -10 if range_min is None else range_min
        range_max = 10 if range_max is None else range_max
        n_interpolate = 20  # grid on parameter axis
        grid = np.linspace(0.001, 0.999, 40)  # grid on (u,v)

        # ---------- 1)  No parameter → check directly ------------------ #
        try:
            param_name = str(copul.params[0])
        except (AttributeError, IndexError):
            return self._copula_is_plod(copul, grid)

        # ---------- 2)  Otherwise scan the parameter range ------------- #
        interval = copul.intervals[param_name]
        p_min = float(max(interval.inf, range_min))
        p_max = float(min(interval.sup, range_max))
        if interval.left_open:
            p_min += 0.01
        if interval.right_open:
            p_max -= 0.01

        params = np.linspace(p_min, p_max, n_interpolate)
        is_plod_final = True

        for p in params:
            C = copul(**{param_name: p})
            is_plod = self._copula_is_plod(C, grid)
            log.debug("param %s = %.4g → PLOD %s", param_name, p, is_plod)
            is_plod_final &= is_plod
            if not is_plod_final:  # stop once a counter-example is found
                break

        return is_plod_final

    # ------------------------------------------------------------------ #
    #  Core routine for a *concrete* copula instance
    # ------------------------------------------------------------------ #
    def _copula_is_plod(self, C, grid):
        """
        True  ↔  C is PLOD on the given grid.
        """
        tol = 1e-10
        # ---------- symbolic route (fast if available) ----------------- #
        try:
            C_expr = C.cdf.func  # SymPy expression
            u_sym, v_sym = C.u, C.v
            diff = C_expr - u_sym * v_sym  # should be ≥ 0
            for u in grid:
                for v in grid:
                    if diff.subs({u_sym: u, v_sym: v}) < -tol:
                        return False
        # ---------- numeric fallback ----------------------------------- #
        except Exception:
            cdf = C.cdf  # SymPyFuncWrapper → callable
            for u in grid:
                for v in grid:
                    if cdf(u, v) < u * v - tol:
                        return False
        return True
