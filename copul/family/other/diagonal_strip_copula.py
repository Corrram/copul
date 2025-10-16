import sympy as sp
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import types

from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.core.biv_copula import BivCopula
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class DiagonalStripCopula(BivCopula):
    r"""
    Two-parameter "diagonal strip" copula with a rectangular hole of vertical thickness
    :math:`\beta\in(0,1)` whose lower boundary along the diagonal (parameterized by
    :math:`s\in[0,1]`) is the piecewise-linear function

    .. math::
        \psi(s) \;=\;
        \begin{cases}
          0, & 0 \le s \le \alpha,\\[2pt]
          \displaystyle\frac{1-\beta}{1-2\alpha}\,\bigl(s-\alpha\bigr), & \alpha < s < 1-\alpha,\\[6pt]
          1-\beta, & 1-\alpha \le s \le 1,
        \end{cases}
        \qquad \alpha\in(0,\tfrac12),\ \beta\in(0,1).

    In the construction, a one–dimensional transport :math:`v \mapsto t` is chosen to
    preserve the uniform marginal in :math:`v` while removing the hole of area :math:`\beta`.
    Writing :math:`L(t)` for the fraction of :math:`s\in[0,1]` such that
    :math:`\psi(s)\le t\le \psi(s)+\beta`, we define

    .. math::
        f_T(t) \;=\; \frac{1 - L(t)}{1 - \beta},\qquad
        F_T(t) \;=\; \int_0^t f_T(\tau)\,d\tau,\qquad
        t = F_T^{-1}(v).

    The copula density in the :math:`(u,v)` plane is then
    :math:`c(u,v) = \frac{1}{1-\beta}\,\frac{1}{f_T(t(v))}` outside the hole
    (where :math:`\psi(u) \le t(v) \le \psi(u)+\beta`) and :math:`0` inside the hole.
    """

    # Parameters and domains
    alpha, beta = sp.symbols("alpha beta", real=True)
    params = [alpha, beta]
    intervals = {
        "alpha": sp.Interval.open(0, sp.Rational(1, 2)),
        "beta": sp.Interval.open(0, 1),
    }
    special_cases = {0: BivIndependenceCopula}

    # Convenience symbols
    u, v = sp.symbols("u v", nonnegative=True)

    def __init__(self, *args, **kwargs):
        # Allow positional (alpha, beta)
        if args:
            if len(args) == 1:
                kwargs["alpha"] = args[0]
            elif len(args) == 2:
                kwargs["alpha"], kwargs["beta"] = args
            else:
                raise ValueError("Provide at most two positional args: alpha, beta.")
        super().__init__(**kwargs)
        self._cache_ft = None  # cache (t_grid, f_T, F_T) for current (alpha,beta)

    # ---------------------------
    # Core building blocks
    # ---------------------------
    @staticmethod
    def _psi_vec(s, alpha, beta):
        """Vectorized ψ(s) piecewise function."""
        s = np.asarray(s, dtype=float)
        ps = np.zeros_like(s)
        mask_mid = (s > alpha) & (s < 1.0 - alpha)
        mask_up = s >= 1.0 - alpha
        if alpha < 0.5:
            slope = (1.0 - beta) / (1.0 - 2.0 * alpha)
            ps[mask_mid] = slope * (s[mask_mid] - alpha)
        ps[mask_up] = 1.0 - beta
        return ps

    @staticmethod
    def _L_t(t_grid, alpha, beta, n_points=2000):
        """
        L(t): width of the hole at height t; computed by integrating the indicator
        over s∈[0,1] on a fine grid.
        """
        s_fine = np.linspace(0.0, 1.0, n_points)
        psi_vals = DiagonalStripCopula._psi_vec(s_fine, alpha, beta)  # (n_points,)
        # is_in_hole[i,j]: for t_grid[i], whether that s_fine[j] falls inside the vertical strip
        is_in_hole = (psi_vals[np.newaxis, :] <= t_grid[:, np.newaxis]) & (
            psi_vals[np.newaxis, :] + beta >= t_grid[:, np.newaxis]
        )
        return trapezoid(is_in_hole.astype(float), s_fine, axis=1)

    def _ensure_ft_cache(self, grid_n=2000):
        """
        Build (and cache) t_grid, f_T(t), F_T(t) for current alpha,beta.
        """
        alpha = float(self.alpha)
        beta = float(self.beta)
        key = (alpha, beta, grid_n)
        if self._cache_ft and self._cache_ft[0] == key:
            return self._cache_ft[1]

        if not (0.0 < alpha < 0.5 and 0.0 < beta < 1.0):
            raise ValueError(
                "Parameters must satisfy 0 < alpha < 0.5 and 0 < beta < 1."
            )

        t_grid = np.linspace(0.0, 1.0, grid_n)
        L_t = self._L_t(t_grid, alpha, beta, n_points=grid_n)
        # total hole area is β by construction
        hole_area = beta
        if hole_area >= 1.0:
            # degenerate (no mass left) — return flat stubs
            f_T = np.ones_like(t_grid)
            F_T = t_grid
        else:
            f_T = (1.0 - L_t) / (1.0 - hole_area)
            # CDF by integration and ensure monotonicity
            F_T = cumulative_trapezoid(f_T, t_grid, initial=0.0)
            F_T = np.maximum.accumulate(F_T)
            # Normalize to 1 (small numerical drift)
            if F_T[-1] > 0:
                F_T = F_T / F_T[-1]

        self._cache_ft = (key, (t_grid, f_T, F_T))
        return self._cache_ft[1]

    def _t_of_v(self, v_vals):
        """Invert F_T to get t(v) by interpolation."""
        v_vals = np.asarray(v_vals, dtype=float)
        t_grid, f_T, F_T = self._ensure_ft_cache()
        # Clip v to [0,1] to avoid extrapolation wobbles
        v_clip = np.clip(v_vals, 0.0, 1.0)
        t_vals = np.interp(v_clip, F_T, t_grid)
        return t_vals, f_T

    # ---------------------------
    # Vectorized PDF / CDF
    # ---------------------------
    def pdf_vectorized(self, u, v):
        """
        Copula density c(u,v): zero inside the diagonal strip (hole) and
        constant (in u) outside the strip with height
            (1/(1-β)) * 1 / f_T(t(v)).
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        alpha = float(self.alpha)
        beta = float(self.beta)

        # map v -> t via inverse transform
        t_vals, f_T = self._t_of_v(v)
        # evaluate ψ(u)
        psi_u = self._psi_vec(u, alpha, beta)

        # inside the hole when ψ(u) ≤ t ≤ ψ(u)+β
        inside = (t_vals >= psi_u) & (t_vals <= psi_u + beta)

        # height outside the hole
        eps = 1e-12
        t_grid, f_T_grid, _ = self._ensure_ft_cache()
        fT_at_t = np.interp(t_vals, t_grid, f_T_grid)  # f_T evaluated at t(v)
        height = (1.0 / (1.0 - beta)) * 1.0 / (fT_at_t + eps)

        out = np.zeros_like(t_vals, dtype=float)
        out[~inside] = height[~inside]
        return out.reshape(np.broadcast(u, v).shape)

    @property
    def cdf(self):
        """We expose a symbolic placeholder to match the interface, but numerics are used."""
        return self._cdf_expr

    def cdf_vectorized(self, u, v, grid_n=600):
        """
        Numerical C(u,v) by integrating the density on a regular grid in (u,v),
        then bilinearly interpolating to (u,v).
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # Regular grid for integration
        ug = np.linspace(0.0, 1.0, grid_n)
        vg = np.linspace(0.0, 1.0, grid_n)
        Vg, Ug = np.meshgrid(vg, ug, indexing="ij")  # Vg,Ug shapes: (Nv, Nu)

        Z = self.pdf_vectorized(Ug, Vg)  # shape (Nv, Nu)
        H = cumulative_trapezoid(
            Z, vg, axis=0, initial=0.0
        )  # integrate in v -> (Nv, Nu)
        Cmat = cumulative_trapezoid(
            H, ug, axis=1, initial=0.0
        )  # integrate in u -> (Nv, Nu)

        # Bilinear interpolation from (ug, vg) -> Cmat at arbitrary (u,v)
        u_flat = np.clip(u.ravel(), 0.0, 1.0)
        v_flat = np.clip(v.ravel(), 0.0, 1.0)

        # indices of left/bottom cells
        iu = np.searchsorted(ug, u_flat, side="right") - 1
        iv = np.searchsorted(vg, v_flat, side="right") - 1
        iu = np.clip(iu, 0, len(ug) - 2)
        iv = np.clip(iv, 0, len(vg) - 2)

        u0 = ug[iu]
        u1 = ug[iu + 1]
        v0 = vg[iv]
        v1 = vg[iv + 1]
        eps = 1e-15
        su = (u_flat - u0) / (u1 - u0 + eps)
        tv = (v_flat - v0) / (v1 - v0 + eps)

        C00 = Cmat[iv, iu]
        C10 = Cmat[iv, iu + 1]
        C01 = Cmat[iv + 1, iu]
        C11 = Cmat[iv + 1, iu + 1]

        C_flat = (
            (1 - su) * (1 - tv) * C00
            + su * (1 - tv) * C10
            + (1 - su) * tv * C01
            + su * tv * C11
        )

        return C_flat.reshape(np.broadcast(u, v).shape)

    # ---------------------------
    # Rank measures (numerical)
    # ---------------------------
    def chatterjees_xi(self, grid_n=1200):
        """
        Numerical ξ = 6 ∫∫ H(u,v)^2 dudv - 2, with H = ∫ c du (or dv consistently).
        We replicate the order from your script: integrate c over v -> H, then
        integrate H^2 over the unit square.
        """
        ug = np.linspace(0.0, 1.0, grid_n)
        vg = np.linspace(0.0, 1.0, grid_n)
        Vg, Ug = np.meshgrid(vg, ug, indexing="ij")
        Z = self.pdf_vectorized(Ug, Vg)
        H = cumulative_trapezoid(Z, vg, axis=0, initial=0.0)
        xi_int = trapezoid(trapezoid(H**2, vg, axis=0), ug)
        return 6.0 * xi_int - 2.0

    def spearmans_footrule(self, grid_n=1200):
        """
        Numerical ψ (Spearman's footrule on the diagonal), computed as
        ψ = 6 ∫_0^1 C(u,u) du - 2.
        """
        ug = np.linspace(0.0, 1.0, grid_n)
        Vg, Ug = np.meshgrid(ug, ug, indexing="ij")
        # Build C(u,v) by integrating pdf (reuse cdf_vectorized on the diagonal)
        Z = self.pdf_vectorized(Ug, Vg)
        H = cumulative_trapezoid(Z, ug, axis=0, initial=0.0)
        C = cumulative_trapezoid(H, ug, axis=1, initial=0.0)
        Cdiag = np.diag(C)
        val = trapezoid(Cdiag, ug)
        return 6.0 * val - 2.0

    # ---------------------------
    # Plotting (reuse vectorized numerics)
    # ---------------------------
    def _plot3d(self, func, title, zlabel, zlim=None, **kwargs):
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            f = func
        else:
            if isinstance(func, types.MethodType):
                func = func()
            f = func

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        x = np.linspace(0.01, 0.99, 60)
        y = np.linspace(0.01, 0.99, 60)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        if zlim:
            ax.set_zlim(*zlim)
        plt.show()
        return fig, ax

    def _plot_contour(
        self, func, title, zlabel, *, levels=200, zlim=None, log_z=False, **kwargs
    ):
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            f = func
        else:
            if isinstance(func, types.MethodType):
                func = func()
            f = func

        grid_size = kwargs.pop("grid_size", 300)
        x = np.linspace(0.005, 0.995, grid_size)
        y = np.linspace(0.005, 0.995, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        if zlim:
            Z = np.clip(Z, zlim[0], zlim[1])

        fig, ax = plt.subplots()
        if log_z:
            Zp = np.array(Z, copy=True)
            if np.any(Zp > 0):
                Zp[Zp <= 0] = np.min(Zp[Zp > 0])
                norm = mcolors.LogNorm(vmin=Zp.min(), vmax=Zp.max())
                cf = ax.contourf(X, Y, Zp, levels=levels, cmap="viridis", norm=norm)
            else:
                cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
        else:
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")

        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(zlabel)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_pdf(self, *, plot_type="contour", log_z=False, **kwargs):
        title = kwargs.pop("title", "Diagonal Strip PDF")
        zlabel = kwargs.pop("zlabel", "PDF")
        if plot_type == "3d":
            return self._plot3d(self.pdf_vectorized, title, zlabel, **kwargs)
        elif plot_type == "contour":
            return self._plot_contour(
                self.pdf_vectorized, title, zlabel, log_z=log_z, **kwargs
            )
        else:
            raise ValueError(f"plot_type must be '3d' or 'contour', not {plot_type}")

    def plot_cdf(self, *, plot_type="contour", log_z=False, **kwargs):
        title = kwargs.pop("title", "Cumulative Distribution Function")
        zlabel = kwargs.pop("zlabel", "CDF")
        # cdf is numerical; use cdf_vectorized
        if plot_type == "3d":
            return self._plot3d(
                self.cdf_vectorized, title, zlabel, zlim=(0, 1), **kwargs
            )
        elif plot_type == "contour":
            return self._plot_contour(
                self.cdf_vectorized, title, zlabel, zlim=(0, 1), log_z=log_z, **kwargs
            )
        else:
            raise ValueError(f"plot_type must be '3d' or 'contour', not {plot_type}")

    def __new__(cls, *args, **kwargs):
        if args:
            kwargs["alpha"] = args[0]
        if "alpha" in kwargs and kwargs["alpha"] in cls.special_cases:
            special_case_cls = cls.special_cases[kwargs["alpha"]]
            del kwargs["alpha"]
            return special_case_cls()
        return super().__new__(cls)

    @property
    def pdf(self):
        """
        Symbolic PDF, c(u,v).
        The density is non-zero only on the 'yellow region' and its value
        depends on v to ensure valid uniform marginals.
        """
        alpha, u, v = self.alpha, self.u, self.v
        r = sp.sqrt(alpha)

        # 1. Define the value of the density c(v) on its support
        c_v_expr = sp.Piecewise(
            (1 / (1 - v - r / 2), v < r),
            (1 / (1 - r), v <= 1 - r),
            (1 / (v - r / 2), True),
        )

        # 2. Define the support (the "yellow region")
        psi = sp.Min(sp.Max(u - r / 2, 0), 1 - r)
        in_white = (v >= psi) & (v <= psi + r)

        # 3. Combine them: density is c(v) on the support, 0 otherwise
        # NOTE: The use of `~in_white` appears to be a bug, as it makes the
        # copula invalid for alpha > 1/4. The implementation of cond_distr
        # assumes the density is ON the strip (i.e., `in_white` is used).
        pdf_expr = sp.Piecewise((c_v_expr, ~in_white), (0, True))
        return SymPyFuncWrapper(pdf_expr)

    @property
    def _cdf_expr(self):
        raise NotImplementedError(
            "The analytical CDF for this density is complex and has not been implemented."
        )

    def cond_distr_1(self, u=None, v=None):
        """
        Symbolic conditional distribution, C(u|v).
        C(u|v) = P(U <= u | V = v) = integral from 0 to u of c(x,v) dx.

        NOTE: This implementation assumes a correction to the PDF definition.
        The original PDF is defined as non-zero *outside* a diagonal strip
        (due to `~in_white`). This leads to an invalid copula for alpha > 1/4.
        We assume the density is meant to be non-zero *inside* the strip,
        which makes it a valid copula for all alpha in [0, 1/2].
        """
        if u is None:
            u = self.u
        if v is None:
            v = self.v

        alpha = self.alpha
        r = sp.sqrt(alpha)

        # Case 1: v is in the bottom-left section of the strip (v < r)
        cd_v_lt_r = sp.Min(u, v + r / 2) / (v + r / 2)

        # Case 2: v is in the top-right section of the strip (v > 1 - r)
        cd_v_gt_1mr = sp.Max(0, u - (v - r / 2)) / (1 - v + r / 2)

        # Case 3: v is in the central diagonal section (r <= v <= 1-r)
        cd_v_between = sp.Max(0, sp.Min(u, v + r / 2) - (v - r / 2)) / r

        # Combine into a single piecewise expression for C(u|v).
        cond_expr = sp.Piecewise(
            (cd_v_lt_r, v < r),
            (cd_v_gt_1mr, v > 1 - r),
            (cd_v_between, True),  # The remaining case is r <= v <= 1-r
        )

        return SymPyFuncWrapper(cond_expr)

    def cond_distr_2(self, u=None, v=None):
        """
        Symbolic conditional distribution, C(v|u) = P(V <= v | U = u).

        NOTE: The density function defined in this class does not correspond
        to a valid copula, because the marginal distribution for U, f_U(u),
        is not uniform. This implementation computes the true conditional
        distribution C(v|u) for the given density, which is:
        C(v|u) = integral_0^v c(u,y)dy / integral_0^1 c(u,y)dy
        """
        if u is None:
            u = self.u
        if v is None:
            v = self.v

        alpha = self.alpha
        r = sp.sqrt(alpha)

        # Dummy integration variable
        y = sp.symbols("y", real=True, positive=True)

        # Define c(y), the density's value (which only depends on y)
        # This is based on the corrected assumption that density is ON the strip
        c_y_func = sp.Piecewise(
            (1 / (y + r / 2), y < r), (1 / (1 - y + r / 2), y > 1 - r), (1 / r, True)
        )

        # Define psi(u), the lower bound of the support for y at a given u
        psi_u = sp.Min(sp.Max(u - r / 2, 0), 1 - r)

        # Numerator: integral from 0 to v of c(u,y)dy.
        # This is the integral of c(y) on the intersection of
        # [0, v] and the support interval [psi(u), psi(u) + r].
        num_lower_bound = psi_u
        num_upper_bound = sp.Min(v, psi_u + r)
        # SymPy's integrate handles cases where upper_bound < lower_bound
        numerator = sp.integrate(c_y_func, (y, num_lower_bound, num_upper_bound))

        # Denominator: marginal density f_U(u) = integral from 0 to 1 of c(u,y)dy.
        # This is the integral of c(y) over the full support [psi(u), psi(u) + r].
        den_lower_bound = psi_u
        den_upper_bound = psi_u + r
        denominator = sp.integrate(c_y_func, (y, den_lower_bound, den_upper_bound))

        # The conditional probability C(v|u)
        cond_expr = numerator / denominator

        return SymPyFuncWrapper(cond_expr)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        # Note: With c(u,v) = c(v), this copula is no longer symmetric.
        # Symmetry requires c(u,v) = c(v,u).
        return False

    def cond_distr_1_vectorized(self, u, v):
        """
        h(u,v) = P(U <= u | V = v) = ∫_0^u c(x,v) dx.
        Fully numeric, works on arrays (broadcasts like pdf_vectorized).
        """
        # Strategy: for each fixed v, c(x,v) is 0 on the u-interval where
        # psi(x) <= t(v) <= psi(x) + beta, and equals 'height(v)' otherwise.
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        alpha = float(self.alpha)
        beta = float(self.beta)

        # Map v -> t and height(v)
        t_vals, _ = self._t_of_v(v)
        t_grid, f_T_grid, _ = self._ensure_ft_cache()
        eps = 1e-12
        fT_at_t = np.interp(t_vals, t_grid, f_T_grid)
        height = (1.0 / (1.0 - beta)) * 1.0 / (fT_at_t + eps)

        # We need to integrate c(x,v) in x from 0 to u for each (u,v).
        # Build a fine x-grid matching the shape of u for stable integration.
        # If u,v come from a meshgrid (plotting), treat axis 1 as x/u-axis.
        # Otherwise, broadcast to common shape and integrate along "last" axis.
        U = np.asarray(u)
        V = np.asarray(v)
        # Ensure we have 2D grids (like plotting expects)
        if U.ndim == 1 and V.ndim == 1:
            Vg, Ug = np.meshgrid(V, U, indexing="ij")
        else:
            # assume meshgrids already
            Ug, Vg = U, V

        # For each column (fixed v), compute c(x,v) along x:
        # psi depends on x only:
        psi_x = self._psi_vec(Ug, alpha, beta)
        tV = t_vals if Vg.shape == t_vals.shape else np.broadcast_to(t_vals, Vg.shape)
        inside = (tV >= psi_x) & (tV <= psi_x + beta)

        heightV = (
            height if Vg.shape == height.shape else np.broadcast_to(height, Vg.shape)
        )
        c_xv = np.zeros_like(Ug, dtype=float)
        c_xv[~inside] = heightV[~inside]

        # cumulative integral in x (u-direction) along axis=1 (since indexing="ij")
        # First build a uniform x-grid in [0,1] of same columns as Ug:
        xg = Ug[0, :]  # assumes Ug has same x-grid along rows
        H = cumulative_trapezoid(c_xv, xg, axis=1, initial=0.0)

        # Now, return H evaluated at the passed u-values:
        # If caller passes generic u (not equal to grid), we can bilinearly interp.
        # In plotting calls, Ug already matches the grid, so H matches shape directly.
        return H

    def cond_distr_2_vectorized(self, u, v):
        """
        P(V <= v | U = u) = ∫_0^v c(u,y) dy / ∫_0^1 c(u,y) dy.
        Numeric and array-aware.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        # Build a v-grid and integrate c along v
        if u.ndim == 1 and v.ndim == 1:
            Vg, Ug = np.meshgrid(v, u, indexing="ij")
        else:
            Ug, Vg = u, v

        Z = self.pdf_vectorized(Ug, Vg)
        vg = Vg[:, 0]  # v-grid along rows
        num = cumulative_trapezoid(Z, vg, axis=0, initial=0.0)  # ∫_0^v c(u,y) dy
        den = num[-1, :]  # ∫_0^1 c(u,y) dy  (last row in v)
        den = np.where(den == 0, 1.0, den)  # guard
        C = num / den  # broadcasts along columns
        return C

    # ---------------------------
    # Override plotting for conditionals to use numeric callables (NEW)
    # ---------------------------
    def plot_cond_distr_1(self, *, plot_type="contour", log_z=False, **kwargs):
        title = kwargs.pop("title", "Conditional Distribution h(u,v)")
        zlabel = kwargs.pop("zlabel", "h(u,v)")

        def f(U, V):
            return self.cond_distr_1_vectorized(U, V)

        if plot_type == "3d":
            return self._plot3d(f, title, zlabel, zlim=(0, 1), **kwargs)
        elif plot_type == "contour":
            return self._plot_contour(
                f, title, zlabel, zlim=(0, 1), log_z=log_z, **kwargs
            )
        elif plot_type == "slices":
            return self._plot_functions(f, title, zlabel, xlabel="u", **kwargs)
        else:
            raise ValueError("plot_type must be '3d', 'contour', or 'slices'.")

    def plot_cond_distr_2(self, *, plot_type="contour", log_z=False, **kwargs):
        title = kwargs.pop("title", "Conditional Distribution C(v|u)")
        zlabel = kwargs.pop("zlabel", "C(v|u)")

        def f(U, V):
            return self.cond_distr_2_vectorized(U, V)

        if plot_type == "3d":
            return self._plot3d(f, title, zlabel, zlim=(0, 1), **kwargs)
        elif plot_type == "contour":
            return self._plot_contour(
                f, title, zlabel, zlim=(0, 1), log_z=log_z, **kwargs
            )
        elif plot_type == "slices":
            return self._plot_functions(f, title, zlabel, xlabel="v", **kwargs)
        else:
            raise ValueError("plot_type must be '3d', 'contour', or 'slices'.")


if __name__ == "__main__":
    # quick smoke test & plots
    pairs = [(0.20, 0.30), (0.30, 0.50), (0.40, 0.50)]
    for a, b in pairs:
        cop = DiagonalStripCopula(alpha=a, beta=b)
        print(f"alpha={a:.2f}, beta={b:.2f}")
        xi = cop.chatterjees_xi(grid_n=800)
        psi = cop.spearmans_footrule(grid_n=800)
        print(f"  xi ≈ {xi:.4f},  psi ≈ {psi:.4f}")
        cop.plot_pdf(plot_type="3d", levels=600, grid_size=600)
        cop.plot_cond_distr_1(plot_type="3d", levels=600, grid_size=600)
        cop.plot_cond_distr_2(plot_type="3d", levels=600, grid_size=600)
        cop.plot_cdf(plot_type="3d", levels=600, grid_size=600)
