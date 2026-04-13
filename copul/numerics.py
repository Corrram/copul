# numerics.py
"""
Utilities for converting SymPy expressions to fast NumPy callables.

Key improvements over the naive ``sp.lambdify`` approach:
- ``NUMPY_SAFE_MAP`` maps SymPy names that numpy does not expose under the same
  name (``Max``, ``Min``, ``Heaviside``, …) plus several ``scipy.special``
  functions used in copula densities (``erf``, ``erfc``, ``gamma``, …).
- ``to_numpy_callable`` caches compiled functions by (expression-string,
  variable-names) to avoid re-invoking SymPy's lambdify for the same expression.
"""

import logging
from typing import Sequence

import numpy as np
import sympy as sp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scipy.special imports (graceful degradation if scipy not present)
# ---------------------------------------------------------------------------
try:
    from scipy import special as _sc_special

    _SCIPY_MAP = {
        # Error functions (Gaussian copula, normal distribution)
        "erf": _sc_special.erf,
        "erfc": _sc_special.erfc,
        "erfinv": _sc_special.erfinv,
        # Gamma and related (Student-t, some Archimedean densities)
        "gamma": _sc_special.gamma,
        "loggamma": _sc_special.loggamma,
        "polygamma": _sc_special.polygamma,
        "digamma": _sc_special.digamma,
        # Beta function
        "beta": _sc_special.beta,
        "betainc": _sc_special.betainc,
        # Bessel / zeta (occasionally appear in EV copula densities)
        "zeta": _sc_special.zeta,
    }
except ImportError:  # pragma: no cover
    _SCIPY_MAP = {}
    log.debug("scipy not available; scipy.special functions unavailable in lambdify")

# ---------------------------------------------------------------------------
# Master safe-function map passed to sp.lambdify
# ---------------------------------------------------------------------------
NUMPY_SAFE_MAP: dict = {
    # ---- piecewise-style operations ----------------------------------------
    # SymPy's Max/Min are not the same as numpy.max/numpy.min (those reduce
    # arrays; we need element-wise comparisons).
    "Max": (lambda *xs: np.maximum.reduce(xs)),
    "Min": (lambda *xs: np.minimum.reduce(xs)),
    "Abs": np.abs,
    # ---- distributional terms ----------------------------------------------
    # DiracDelta has measure zero; dropping it is correct a.e.
    "DiracDelta": (lambda *args, **kw: 0.0),
    # Heaviside with the conventional H(0)=1/2
    "Heaviside": (
        lambda x, H0=0.5: np.where(
            np.asarray(x) > 0, 1.0, np.where(np.asarray(x) < 0, 0.0, H0)
        )
    ),
    # ---- integer / rounding ------------------------------------------------
    "sign": np.sign,
    "floor": np.floor,
    "ceiling": np.ceil,
    # ---- expose scalar math so lambdify never falls back to Python math ----
    # (numpy's module already covers log/exp/sqrt, but being explicit avoids
    # edge cases where SymPy generates the function name with a capital letter)
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    # ---- scipy.special (if available) --------------------------------------
    **_SCIPY_MAP,
}


# ---------------------------------------------------------------------------
# Distributional-term removal
# ---------------------------------------------------------------------------


def drop_distributions(expr: sp.Expr) -> sp.Expr:
    """
    Remove distributional terms for 'a.e.' numeric evaluation (plots, grids).

    Specifically:
    - ``DiracDelta(·)``                → 0
    - ``Derivative(Heaviside(·), ·)``  → 0  (SymPy represents this as DiracDelta)
    """
    return expr.replace(sp.DiracDelta, lambda *args: sp.Integer(0))


# ---------------------------------------------------------------------------
# Lambdify cache
# ---------------------------------------------------------------------------

# Cache keyed by (expression_string, variable_names_tuple, ae_flag).
# Using the string representation of the expression as the cache key is safe
# because SymPy's __str__ is canonical for structurally identical expressions,
# and avoids hashing mutable SymPy objects.
_lambdify_cache: dict = {}


def _cache_key(expr: sp.Expr, vars: Sequence, ae: bool) -> tuple:
    return (str(expr), tuple(str(v) for v in vars), ae)


def clear_lambdify_cache() -> None:
    """
    Evict all cached lambdified functions.

    Useful in long-running sessions or after symbolic manipulations that
    produce many unique expressions.
    """
    _lambdify_cache.clear()


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def to_numpy_callable(expr: sp.Expr, vars: Sequence, *, ae: bool = True):
    """
    Compile a SymPy expression to a NumPy-callable with robust function mappings.

    Parameters
    ----------
    expr : sympy.Expr
        The symbolic expression to compile.
    vars : sequence of sympy.Symbol
        Free variables, in the order expected by the returned callable.
    ae : bool
        If ``True`` (default), drop ``DiracDelta`` terms so the result is
        valid *almost everywhere* — appropriate for density plots and grids.

    Returns
    -------
    callable
        ``f(*arrays) -> array``  fully broadcast-compatible with NumPy.

    Notes
    -----
    Compiled functions are cached by ``(str(expr), var_names, ae)``.  The
    first call for a given expression pays the SymPy compilation overhead;
    subsequent calls return the cached function instantly.
    """
    key = _cache_key(expr, vars, ae)
    cached = _lambdify_cache.get(key)
    if cached is not None:
        return cached

    if ae:
        expr = drop_distributions(expr)

    # numpy handles Piecewise/select/less/greater natively when it appears in
    # lambdified output; NUMPY_SAFE_MAP fills in the remaining gaps.
    modules = [NUMPY_SAFE_MAP, "numpy"]
    f = sp.lambdify(tuple(vars), expr, modules=modules)

    _lambdify_cache[key] = f
    log.debug("lambdified: %s (ae=%s)", str(expr)[:80], ae)
    return f
