import sympy as sp
import numpy as np
from copul.families.core.biv_copula import BivCopula
from copul.wrapper.cdf_wrapper import CDFWrapper


# ---------------------------------------------------------------------------
#  Markov (star) product of two instantiated copulas
# ---------------------------------------------------------------------------
def star_product(
    C: "BivCopula",
    D: "BivCopula",
    *,
    n_grid: int = 400,  # integration grid for t ∈ [0,1]
    checkerboard: bool = False,  # quick & coarse plotting
) -> "BivCopula":
    """
    Return the copula (C ⋆ D) created from two *instances* C and D.

    Parameters
    ----------
    C, D       instantiated subclasses of ``BivCopula``
    n_grid     resolution of the inner t–integral
    checkerboard  if True the resulting object's pdf() uses a fast 1/n_grid
                  checkerboard approximation – useful for contour plots

    Notes
    -----
    • If both inputs have a symbolic CDF we *keep* the unevaluated
      integral for pretty-printing but evaluate numerically when needed.
    • All numerical work is fully vectorised; no ``np.vectorize`` remains.
    """
    # ------------------------------------------------------------------ #
    # symbolic objects (if available)
    # ------------------------------------------------------------------ #
    try:
        C_expr = C.cdf().func
        D_expr = D.cdf().func

        C_dv = sp.diff(C_expr, C.v)  # ∂C/∂v   (cond. v|u)
        D_du = sp.diff(D_expr, D.u)  # ∂D/∂u   (cond. u|v)
        t_sym = sp.symbols("t")
        integrand_sym = C_dv.subs(C.v, t_sym) * D_du.subs(D.u, t_sym)
        cdf_expr = sp.Integral(integrand_sym, (t_sym, 0, 1))  # unevaluated
        symbolic_ok = True
    except Exception:
        symbolic_ok = False  # silently switch to numeric-only

    # ------------------------------------------------------------------ #
    # pre-compute grid & helpers (both branches share them)
    # ------------------------------------------------------------------ #
    t_grid = np.linspace(0.0, 1.0, n_grid)
    dt = t_grid[1] - t_grid[0]

    # These call *once*, return broadcasting-friendly arrays
    def C_cond(u, t):
        return C.cond_distr_2(u, t)

    def D_cond(t, v):
        return D.cond_distr_1(t, v)

    try:
        C_pdf_fun = C.pdf
    except Exception:
        C_pdf_fun = None
    try:
        D_pdf_fun = D.pdf
    except Exception:
        D_pdf_fun = None

    # ------------------------------------------------------------------ #
    # helper: argument handling (shared by both star classes)
    # ------------------------------------------------------------------ #
    def _process_args(args):
        if len(args) == 2:  # f(u,v)
            return np.asarray(args[0]), np.asarray(args[1])
        if len(args) == 1:  # f([u,v])  or  f([[..]])
            arr = np.asarray(args[0])
            if arr.ndim == 1 and arr.size == 2:
                return arr[0], arr[1]
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr[:, 0], arr[:, 1]
        raise ValueError("expected (u,v) or array with two columns")

    # ------------------------------------------------------------------ #
    # core integrators (vectorised)
    # ------------------------------------------------------------------ #
    def _integrate_over_t(f_of_t):
        # trapezoid along last axis (= t)
        return np.trapz(f_of_t, dx=dt, axis=-1)

    def _cdf_array(u, v):
        # broadcasting shapes:  u[...,None] × t_grid[None,:] -> (..., n_grid)
        f = C_cond(np.expand_dims(u, -1), t_grid) * D_cond(
            t_grid, np.expand_dims(v, -1)
        )
        return _integrate_over_t(f)

    # fallback derivative with central difference (vectorised)
    h = 1e-5

    def _cond1_array(u, v):
        if C_pdf_fun is not None:
            f = C_pdf_fun(np.expand_dims(u, -1), t_grid) * D_cond(
                t_grid, np.expand_dims(v, -1)
            )
            return _integrate_over_t(f)
        return (_cdf_array(u + h, v) - _cdf_array(np.maximum(u - h, 0.0), v)) / (2 * h)

    def _cond2_array(u, v):
        if D_pdf_fun is not None:
            f = C_cond(np.expand_dims(u, -1), t_grid) * D_pdf_fun(
                t_grid, np.expand_dims(v, -1)
            )
            return _integrate_over_t(f)
        return (_cdf_array(u, v + h) - _cdf_array(u, np.maximum(v - h, 0.0))) / (2 * h)

    def _pdf_array(u, v):
        if (C_pdf_fun is not None) and (D_pdf_fun is not None):
            f = C_pdf_fun(np.expand_dims(u, -1), t_grid) * D_pdf_fun(
                t_grid, np.expand_dims(v, -1)
            )
            return _integrate_over_t(f)
        return (_cond2_array(u + h, v) - _cond2_array(np.maximum(u - h, 0.0), v)) / (
            2 * h
        )

    # optional fast checkerboard – good for dense contour plots
    if checkerboard:

        def _pdf_array(u, v):
            # • map (u,v) onto nearest grid corner
            u_idx = np.minimum(np.round(u * (n_grid - 1)).astype(int), n_grid - 2)
            v_idx = np.minimum(np.round(v * (n_grid - 1)).astype(int), n_grid - 2)
            u0 = u_idx / (n_grid - 1)
            v0 = v_idx / (n_grid - 1)
            # • evaluate in the cell centre once
            val = _pdf_array.__dict__.setdefault("cache", {})
            key = (u_idx.tobytes(), v_idx.tobytes())
            if key not in val:
                val[key] = _pdf_array_raw(
                    u0 + 0.5 / (n_grid - 1), v0 + 0.5 / (n_grid - 1)
                )
            return val[key]

        _pdf_array_raw = _pdf_array  # keep exact version around

    # ------------------------------------------------------------------ #
    #   build the resulting copula class
    # ------------------------------------------------------------------ #
    class _Star(BivCopula):
        """Copula of (C ⋆ D).  Evaluations fully vectorised."""

        def __init__(self):
            super().__init__()
            if symbolic_ok:
                self._cdf_expr = cdf_expr  # pretty print
            self._n_grid = n_grid

        # ---- core API -------------------------------------------------
        def cdf(self, *args):
            if not args:
                return CDFWrapper(lambda uu, vv: _cdf_array(uu, vv))
            u, v = _process_args(args)
            out = _cdf_array(u, v)
            return out.item() if np.isscalar(out) else out

        def cond_distr_1(self, *args):
            u, v = _process_args(args)
            out = _cond1_array(u, v)
            return out.item() if np.isscalar(out) else out

        def cond_distr_2(self, *args):
            u, v = _process_args(args)
            out = _cond2_array(u, v)
            return out.item() if np.isscalar(out) else out

        def pdf(self, *args):
            u, v = _process_args(args)
            out = _pdf_array(u, v)
            return out.item() if np.isscalar(out) else out

    return _Star()
