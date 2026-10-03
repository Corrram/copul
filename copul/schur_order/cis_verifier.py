import logging

import numpy as np
import sympy

log = logging.getLogger(__name__)


class CISVerifier:
    """
    Verifier for stochastic monotonicity of copulas (SI / SD, a.k.a. CI / CD).

    The copula is *SI in the conditioning variable* ``i`` if the conditional
    distribution function ``h(u, v) = P(V <= v | U = u)`` (for ``i = 1``) is
    nonincreasing in ``u`` for every ``v`` (and SD if it is nondecreasing).

    * :meth:`is_cis` returns a ``bool`` (SI),
    * :meth:`is_cds` returns a ``bool`` (SD),
    * :meth:`cis_direction` returns the tuple ``(is_SI, is_SD)``.

    Checkerboard copulas (``BivCheckPi``, ``BivCheckMin``, ``BivCheckW``,
    ``BivCheckMixed``) are checked *exactly* via their ``cis_direction``
    method; other copulas are checked numerically on an ``n_grid`` x
    ``n_grid`` grid of ``(0, 1)^2``.  For parametric families (with free
    parameters) the property must hold for every parameter on an
    ``n_interpolate``-point grid of the admissible parameter interval.
    """

    def __init__(self, cond_distr=1, n_grid: int = 50, n_interpolate: int = 20):
        """
        Parameters:
        -----------
        cond_distr : int
            Which conditional distribution to check (1 or 2)
        n_grid : int
            Number of grid points per axis for numerical checks.
        n_interpolate : int
            Number of parameter values checked for parametric families.
        """
        if cond_distr not in (1, 2):
            raise ValueError("cond_distr must be 1 or 2")
        self.cond_distr = cond_distr
        self.n_grid = n_grid
        self.n_interpolate = n_interpolate

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def is_cis(self, copul, range_min=None, range_max=None) -> bool:
        """Whether the copula is stochastically increasing (SI / CI)."""
        return self.cis_direction(copul, range_min, range_max)[0]

    def is_cds(self, copul, range_min=None, range_max=None) -> bool:
        """Whether the copula is stochastically decreasing (SD / CD)."""
        return self.cis_direction(copul, range_min, range_max)[1]

    def cis_direction(self, copul, range_min=None, range_max=None):
        """Return ``(is_SI, is_SD)``; over a parameter range both must hold
        for every checked parameter value."""
        exact = self._exact_direction(copul)
        if exact is not None:
            return exact

        linspace = np.linspace(0.001, 0.999, self.n_grid)
        try:
            param = str(copul.params[0])
        except (AttributeError, IndexError, TypeError):
            return self._is_copula_cis(copul, linspace)

        range_min = -10 if range_min is None else range_min
        interval = copul.intervals[param]
        range_min = float(max(interval.inf, range_min))
        if interval.left_open:
            range_min += 0.01
        param_range_max = 10 if range_max is None else range_max
        param_range_max = float(min(interval.end, param_range_max))
        if interval.right_open:
            param_range_max -= 0.01

        all_ci, all_cd = True, True
        for param_value in np.linspace(range_min, param_range_max, self.n_interpolate):
            my_copul = copul(**{param: param_value})
            is_ci, is_cd = self._is_copula_cis(my_copul, linspace)
            log.debug(f"param {param_value}: CI={is_ci}, CD={is_cd}")
            all_ci &= is_ci
            all_cd &= is_cd
            if not (all_ci or all_cd):
                break
        return bool(all_ci), bool(all_cd)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _exact_direction(self, copul):
        from copul.checkerboard._biv_mixin import BivCheckerboardMixin

        if isinstance(copul, BivCheckerboardMixin):
            return copul.cis_direction(self.cond_distr)
        return None

    def _cond_grid(self, my_copul, points):
        """Matrix ``H[k, l] = h(x_k, y_l)`` with conditioning value ``x_k``."""
        X, Y = np.meshgrid(points, points, indexing="ij")
        if self.cond_distr == 1:
            U, V = X, Y
            method = my_copul.cond_distr_1
        else:
            U, V = Y, X
            method = my_copul.cond_distr_2

        # 1) symbolic expression -> vectorised numpy function
        try:
            expr = method().func
            f = sympy.lambdify((my_copul.u, my_copul.v), expr, "numpy")
            with np.errstate(all="ignore"):
                H = np.asarray(f(U, V), dtype=float)
            H = np.broadcast_to(H, U.shape).astype(float)
            if np.all(np.isfinite(H)):
                return H
        except Exception:
            pass

        # 2) numerical method accepting arrays
        try:
            H = np.asarray(method(U, V), dtype=float)
            if H.shape == U.shape and np.all(np.isfinite(H)):
                return H
        except Exception:
            pass

        # 3) scalar fallback
        H = np.empty(U.shape)
        for idx in np.ndindex(U.shape):
            H[idx] = float(method(float(U[idx]), float(V[idx])))
        return H

    def _is_copula_cis(self, my_copul, points, tol: float = 1e-10):
        """
        Check a specific (fully specified) copula instance.

        Returns:
        --------
        tuple
            (is_ci, is_cd) - whether the copula is CI/CD on the grid
        """
        exact = self._exact_direction(my_copul)
        if exact is not None:
            return exact
        from copul.theory.dependence import _is_specified_bivariate, check_property

        if _is_specified_bivariate(my_copul):
            i = self.cond_distr
            return (
                bool(check_property(my_copul, "SI", i=i)),
                bool(check_property(my_copul, "SD", i=i)),
            )
        H = self._cond_grid(my_copul, points)
        d = np.diff(H, axis=0)  # along the conditioning variable
        is_ci = bool(np.all(d <= tol))
        is_cd = bool(np.all(d >= -tol))
        return is_ci, is_cd
