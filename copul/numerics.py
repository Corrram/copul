# numerics.py (or as static methods on CoreCopula)
import numpy as np
import sympy as sp

NUMPY_SAFE_MAP = {
    # hard kinks → broadcast-safe numpy
    "Max": (lambda *xs: np.maximum.reduce(xs)),
    "Min": (lambda *xs: np.minimum.reduce(xs)),
    "Abs": np.abs,
    # distributions / discontinuities
    "DiracDelta": (lambda *args, **kw: 0.0),  # drop measure-zero spikes
    "Heaviside": (
        lambda x, H0=0.5: np.where(
            np.asarray(x) > 0, 1.0, np.where(np.asarray(x) < 0, 0.0, H0)
        )
    ),
    # optional: sign turns into {-1,0,1} with numpy
    "sign": np.sign,
    # optional: floor/ceil/sawtooth stay numeric
}


def drop_distributions(expr: sp.Expr) -> sp.Expr:
    """
    Remove distributional terms for 'a.e.' numeric evaluation (plots, grids).
    - DiracDelta(·) → 0
    - Derivative(Heaviside(·), ·) → 0 (SymPy represents this as DiracDelta anyway)
    """
    return expr.replace(sp.DiracDelta, lambda *args: sp.Integer(0))


def to_numpy_callable(expr: sp.Expr, vars, *, ae: bool = True):
    """
    Compile a SymPy expression to a NumPy-callable with robust mappings.

    Parameters
    ----------
    expr : sympy.Expr
    vars : sequence of sympy.Symbol
    a.e. : bool
        If True, drop distributional terms (DiracDelta) for 'almost everywhere' density.

    Returns
    -------
    callable
        f(*arrays) -> array
    """
    if ae:
        expr = drop_distributions(expr)
    # Let SymPy generate piecewise/select/less/greater; numpy handles them.
    modules = [NUMPY_SAFE_MAP, "numpy"]
    return sp.lambdify(tuple(vars), expr, modules=modules)
