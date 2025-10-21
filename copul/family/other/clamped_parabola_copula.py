from typing import TypeAlias

import sympy as sp
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import types

from copul.family.core.biv_copula import BivCopula
from copul.family.frechet.biv_independence_copula import BivIndependenceCopula
from copul.family.frechet.upper_frechet import UpperFrechet

import warnings
from scipy.integrate import IntegrationWarning

warnings.filterwarnings("ignore", category=IntegrationWarning)


class XiNuBoundaryCopula(BivCopula):
    r"""
    Clamped–parabola copula parameterized by :math:`b=1/\mu>0`:

    .. math::
       h_v(t) \;=\; \mathrm{clamp}\!\left(b\big((1-t)^2 - q(v)\big),\,0,\,1\right),

    where :math:`q(v)` is uniquely determined by the marginal constraint
    :math:`\int_0^1 h_v(t)\,dt=v`.
    """

    # Parameter now: b > 0
    b = sp.symbols("b", positive=True)
    params = [b]
    intervals = {"b": sp.Interval.open(0, sp.oo)}
    # Limits: b->∞ gives M, b->0+ gives Π
    special_cases = {sp.oo: UpperFrechet, 0: BivIndependenceCopula}

    # Convenience symbols
    u, v = sp.symbols("u v", nonnegative=True)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["b"] = args[0]
        super().__init__(**kwargs)
        self._q_cache = {}

    # ---------------------------
    # Core: solve q(v)
    # ---------------------------
    @staticmethod
    def _marginal_integral_residual(q, v_target, b):
        """
        Residual F(q) = (∫_0^1 h_v(t) dt) - v for a given q, with h_v(t)=clamp(b((1-t)^2-q),0,1).
        Valid q ∈ [-1/b, 1].
        """
        if q > 1 or q < -1.0 / b:
            return 1e6

        s_v = 1.0 if q < 0 else 1.0 - np.sqrt(q)  # right zero of 0-clamp
        a_v = max(0.0, 1.0 - np.sqrt(q + 1.0 / b))  # right end of plateau (h=1)

        # ∫ h_v = plateaulen + b * ∫(x^2 - q) on the middle arc, with primitive T(x;q) = x^3/3 - q x
        integral = a_v
        val_at_s = -((1.0 - s_v) ** 3) / 3.0 - q * s_v
        val_at_a = -((1.0 - a_v) ** 3) / 3.0 - q * a_v
        integral += b * (val_at_s - val_at_a)
        return integral - v_target

    def _get_q_v(self, v_val, b_val):
        """
        Solve q(v) for a single v ∈ [0,1].
        """
        # exact endpoints
        if v_val == 0.0:
            return 1.0
        if v_val == 1.0:
            return -1.0 / b_val

        cache_key = (v_val, b_val)
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]

        lo, hi = -1.0 / b_val, 1.0
        try:
            q_val = brentq(
                self._marginal_integral_residual, lo, hi, args=(v_val, b_val)
            )
            self._q_cache[cache_key] = q_val
            return q_val
        except ValueError:
            # check boundaries
            rl = self._marginal_integral_residual(lo, v_val, b_val)
            if np.isclose(rl, 0.0):
                return lo
            rh = self._marginal_integral_residual(hi, v_val, b_val)
            if np.isclose(rh, 0.0):
                return hi
            raise RuntimeError(
                f"Failed to find q for v={v_val}, b={b_val}. "
                f"Residuals: F(-1/b)={rl:.3g}, F(1)={rh:.3g}"
            )

    def _get_q_v_vec(self, v_arr, b_val):
        v_arr = np.asarray(v_arr)
        shp = v_arr.shape
        q_flat = np.array([self._get_q_v(v, b_val) for v in v_arr.ravel()])
        return q_flat.reshape(shp)

    # ---------------------------
    # Vectorized CDF / PDF
    # ---------------------------
    @property
    def cdf(self):
        return self._cdf_expr

    # --- base (|b|) helpers used by both signs ---
    def _base_cdf_vectorized(self, u, v):
        """CDF for the upright copula with parameter b_abs = |b| > 0."""
        u, v = np.asarray(u), np.asarray(v)
        b_abs = float(abs(self.b))

        q = self._get_q_v_vec(v, b_abs)
        s = np.empty_like(q, dtype=float)
        mask = q >= 0.0
        s[~mask] = 1.0
        s[mask] = 1.0 - np.sqrt(q[mask])
        a = np.maximum(0.0, 1.0 - np.sqrt(q + 1.0 / b_abs))

        val_at_u = -((1.0 - u) ** 3) / 3.0 - q * u
        val_at_a = -((1.0 - a) ** 3) / 3.0 - q * a
        middle = a + b_abs * (val_at_u - val_at_a)

        return np.select([u <= a, u <= s], [u, middle], default=v)

    def _base_pdf_vectorized(self, u, v):
        """PDF for the upright copula with parameter b_abs = |b| > 0."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        out = np.zeros_like(u, dtype=float)
        b_abs = float(abs(self.b))

        q = self._get_q_v_vec(v, b_abs)
        a, s = self._switch_points(q, b_abs)

        vprime = self._vprime_of_q(q, b_abs)
        eps = 1e-14
        qprime = 1.0 / np.where(np.abs(vprime) < eps, np.sign(vprime) * eps, vprime)

        height = -b_abs * qprime
        mask = (u > a) & (u < s)
        out[mask] = height[mask]
        return out.reshape(u.shape)

    def cdf_vectorized(self, u, v):
        """
        If b >= 0: C(u,v) = C_base(u,v; |b|).
        If b < 0 : C(u,v) = u - C_base(u, 1-v; |b|)   (σ2 reflection).
        """
        if float(self.b) >= 0:
            return self._base_cdf_vectorized(u, v)
        else:
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            return u_arr - self._base_cdf_vectorized(u_arr, 1.0 - v_arr)

    def pdf_vectorized(self, u, v):
        """
        If b >= 0: c(u,v) = c_base(u,v; |b|).
        If b < 0 : c(u,v) = c_base(u, 1-v; |b|)       (chain rule for σ2).
        """
        if float(self.b) >= 0:
            return self._base_pdf_vectorized(u, v)
        else:
            u_arr, v_arr = np.asarray(u), np.asarray(v)
            return self._base_pdf_vectorized(u_arr, 1.0 - v_arr)

    def _switch_points(self, q, b):
        """Return a(v), s(v) for given q and b."""
        a = np.maximum(0.0, 1.0 - np.sqrt(q + 1.0 / b))
        s = np.empty_like(q, dtype=float)
        mask = q >= 0.0
        s[~mask] = 1.0
        s[mask] = 1.0 - np.sqrt(q[mask])
        return a, s

    def _vprime_of_q(self, q, b):
        """
        Piecewise derivative v'(q) with μ = 1/b.
        Returns an array matching q.
        """
        q = np.asarray(q, dtype=float)
        mu = 1.0 / float(b)
        R = np.sqrt(np.maximum(q + mu, 0.0))  # defined for q > -mu
        r = np.sqrt(np.maximum(q, 0.0))  # defined for q >= 0
        vprime = np.empty_like(q)

        # Regime A1: q < 0 and R < 1
        mask_A1 = (q < 0.0) & (R < 1.0 - 1e-14)
        vprime[mask_A1] = -R[mask_A1] / mu

        # Regime A2: q < 0 and R >= 1  (only if mu >= 1)
        mask_A2 = (q < 0.0) & ~mask_A1
        vprime[mask_A2] = -1.0 / mu

        # Regime B: 0 <= q < 1 - mu  (only if mu < 1)
        mask_B = (q >= 0.0) & (q < 1.0 - mu - 1e-14)
        Delta = R - r
        vprime[mask_B] = -(1.0 / (2.0 * R[mask_B])) * (1.0 + (Delta[mask_B] ** 2) / mu)

        # Regime C: 1 - mu <= q <= 1
        mask_C = ~(mask_A1 | mask_A2 | mask_B)
        vprime[mask_C] = (r[mask_C] - 1.0) / mu

        return vprime

    def cond_distr_1(self):
        """
        h(u,v) = ∂_u C(u,v).
        If b >= 0: h = clamp(|b|((1-u)^2 - q(v)),0,1).
        If b < 0 : h = 1 - clamp(|b|((1-u)^2 - q(1-v)),0,1).   (since C^σ2)
        NOTE: q(·) will be evaluated with |b| in the lambdify bridge.
        """
        b = sp.Abs(self.b)
        q = sp.Function("q")
        h_base = sp.Min(sp.Max(0, b * ((1 - self.u) ** 2 - q(self.v))), 1)
        h_ref = 1 - sp.Min(sp.Max(0, b * ((1 - self.u) ** 2 - q(1 - self.v))), 1)
        return sp.Piecewise((h_base, self.b >= 0), (h_ref, True))

    @property
    def _cdf_expr(self):
        # C(u,v) = ∫_0^u h(x,v) dx — this matches the σ2 wrapper as well.
        return sp.Integral(self.cond_distr_1(), (self.u, 0, self.u))

    def _pdf_expr(self):
        raise NotImplementedError("Symbolic PDF not available; use pdf_vectorized.")

    # ---------------------------
    # Build from target xi
    # ---------------------------
    @classmethod
    def from_xi(cls, x_target):
        """
        Solve for b from target ξ. Since ξ(μ) is strictly decreasing in μ,
        we solve for μ and return b=1/μ.
        """
        if not (0.0 < x_target < 1.0):
            raise ValueError("Target xi must be in (0, 1).")

        # use temporary instances to compute ξ(μ)
        def xi_of_mu(mu):
            tmp = cls(b=1.0 / mu)
            return tmp.chatterjees_xi()

        xi_at_1 = xi_of_mu(1.0)

        if x_target < xi_at_1:
            lo, hi = 1.0, 2.0
            while xi_of_mu(hi) > x_target:
                hi *= 2.0
                if hi > 1e12:
                    raise RuntimeError(
                        "from_xi bracketing failed (upper bound exploded)."
                    )
        else:
            hi, lo = 1.0, 0.5
            while xi_of_mu(lo) < x_target:
                lo *= 0.5
                if lo < 1e-14:
                    return cls(b=1.0 / lo)  # effectively b→∞

        def f(mu):
            return xi_of_mu(mu) - x_target

        mu_val = brentq(f, lo, hi, maxiter=200, xtol=1e-14, rtol=1e-12)
        return cls(b=1.0 / mu_val)

    def chatterjees_xi(self):
        import numpy as _np

        b_abs = float(abs(self.b))
        if b_abs <= 0.0:
            raise ValueError("b must be nonzero.")
        mu = 1.0 / b_abs
        s = _np.sqrt(mu)
        if mu < 1.0:
            t = _np.sqrt(1.0 - mu)
            A = _np.asinh(t / s)
            num = (
                -105 * s**8 * A
                + 183 * s**6 * t
                - 38 * s**4 * t
                - 88 * s**2 * t
                + 112 * s**2
                + 48 * t
                - 48
            )
            den = 210 * s**6
            return num / den
        else:
            return 8.0 * (7.0 * mu - 3.0) / (105.0 * mu**3)

    def blests_nu(self):
        """
        Signed Blest's ν(b). Under the σ₂ reflection (b<0) the dependence reverses,
        so ν must satisfy ν(-b) = -ν(b). We therefore return sign(b) * ν(|b|),
        where ν(|b|) is the closed form in terms of μ = 1/|b|.
        """
        import numpy as _np

        b = float(self.b)
        if b == 0.0:
            return 0.0  # limit = independence

        sgn = 1.0 if b > 0.0 else -1.0
        mu = 1.0 / abs(b)
        s = _np.sqrt(mu)

        if mu < 1.0:
            t = _np.sqrt(1.0 - mu)
            A = _np.asinh(t / s)
            num = (
                -105 * s**8 * A
                + 87 * s**6 * t
                + 250 * s**4 * t
                - 376 * s**2 * t
                + 448 * s**2
                + 144 * t
                - 144
            )
            den = 420 * s**4
            return sgn * (num / den)
        else:
            return sgn * (4.0 * (28.0 * mu - 9.0) / (105.0 * mu**2))

    # ===================================================================
    # START: Rich plotting capabilities (restored, adapted to b)
    # ===================================================================

    def _plot3d(self, func, title, zlabel, zlim=None, **kwargs):
        r"""
        Internal 3D surface plot using either a numpy-callable or a SymPy expr.
        If 'func' is a SymPy expression, we lambdify it with q(v) injected.
        """
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            # Already a numpy-callable
            f = func
        else:
            # func is a SymPy expression or bound method returning an expr
            if isinstance(func, types.MethodType):
                func = func()

            def q_func(v_val):
                return self._get_q_v_vec(v_val, float(self.b))

            f = sp.lambdify(
                (self.u, self.v),
                func,
                modules=[{"q": q_func, "Min": np.minimum, "Max": np.maximum}, "numpy"],
            )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        u_vals = np.linspace(0.01, 0.99, 50)
        v_vals = np.linspace(0.01, 0.99, 50)
        U, V = np.meshgrid(u_vals, v_vals)
        Z = f(U, V)

        ax.plot_surface(U, V, Z, cmap="viridis", edgecolor="none")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        if zlim:
            ax.set_zlim(*zlim)
        else:
            ax.set_zlim(0, None)
        plt.show()
        return fig, ax

    def _plot_contour(
        self, func, title, zlabel, *, levels=200, zlim=None, log_z=False, **kwargs
    ):
        """
        Internal contour plot using either a numpy-callable or a SymPy expr.
        If 'func' is a SymPy expression, we lambdify it with q(v) injected.
        """
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            f = func
        else:
            if isinstance(func, types.MethodType):
                func = func()

            def q_func(v_val):
                return self._get_q_v_vec(v_val, float(self.b))

            f = sp.lambdify(
                (self.u, self.v),
                func,
                modules=[{"q": q_func, "Min": np.minimum, "Max": np.maximum}, "numpy"],
            )

        grid_size = kwargs.pop("grid_size", 100)
        x = np.linspace(0.005, 0.995, grid_size)
        y = np.linspace(0.005, 0.995, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        if zlim:
            Z = np.clip(Z, zlim[0], zlim[1])

        fig, ax = plt.subplots()
        if log_z:
            # avoid zeros for LogNorm
            Zp = np.array(Z, copy=True)
            if np.any(Zp > 0):
                Zp[Zp <= 0] = np.min(Zp[Zp > 0])
                norm = mcolors.LogNorm(vmin=Zp.min(), vmax=Zp.max())
                cf = ax.contourf(X, Y, Zp, levels=levels, cmap="viridis", norm=norm)
            else:
                # fallback if everything is 0
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

    def _plot_functions(self, func, title, zlabel, xlabel="u", **kwargs):
        """
        Internal line plots (slices) using either a numpy-callable or a SymPy expr.
        If 'func' is a SymPy expression, we lambdify it with q(v) injected.
        """
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            f = func
        else:
            if isinstance(func, types.MethodType):
                func = func()

            def q_func(v_val):
                return self._get_q_v_vec(v_val, float(self.b))

            f = sp.lambdify(
                (self.u, self.v),
                func,
                modules=[{"q": q_func, "Min": np.minimum, "Max": np.maximum}, "numpy"],
            )

        u_vals = np.linspace(0.01, 0.99, 200)
        v_vals = np.linspace(0.1, 0.9, 9)
        fig, ax = plt.subplots(figsize=(6, 6))

        for v_i in v_vals:
            y_vals = f(u_vals, v_i)
            ax.plot(u_vals, y_vals, label=f"$v = {v_i:.1f}$", linewidth=2.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(zlabel)
        ax.set_title(f"{title} — {zlabel}")
        ax.grid(True)
        ax.legend(loc="best")
        fig.tight_layout()
        plt.show()
        return fig

    def plot_cdf(self, *, plot_type="3d", log_z=False, **kwargs):
        """Plot the CDF using the numerical :meth:`cdf_vectorized` implementation."""
        title = kwargs.pop("title", "Cumulative Distribution Function")
        zlabel = kwargs.pop("zlabel", "CDF")

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

    def plot_pdf(self, *, plot_type="3d", log_z=False, **kwargs):
        """Plot the PDF using the numerical :meth:`pdf_vectorized` implementation."""
        title = kwargs.pop("title", "Clamped Parabola PDF")
        zlabel = kwargs.pop("zlabel", "PDF")

        if plot_type == "3d":
            return self._plot3d(self.pdf_vectorized, title, zlabel, **kwargs)
        elif plot_type == "contour":
            return self._plot_contour(
                self.pdf_vectorized, title, zlabel, log_z=log_z, **kwargs
            )
        else:
            raise ValueError(f"plot_type must be '3d' or 'contour', not {plot_type}")

    def plot_cond_distr_1(self, *, plot_type="3d", log_z=False, **kwargs):
        """
        Plot h(u,v) = ∂_u C(u,v). Uses the symbolic expression and injects q(v)
        so the base lambdify has a valid mapping.
        """
        title = kwargs.pop("title", "Conditional Distribution h(u,v)")
        zlabel = kwargs.pop("zlabel", "h(u,v)")
        expr = self.cond_distr_1()  # SymPy expression depending on q(v)

        if plot_type == "3d":
            return self._plot3d(expr, title, zlabel, zlim=(0, 1), **kwargs)
        elif plot_type == "contour":
            return self._plot_contour(
                expr, title, zlabel, zlim=(0, 1), log_z=log_z, **kwargs
            )
        elif plot_type == "slices":
            return self._plot_functions(expr, title, zlabel, **kwargs)
        else:
            raise ValueError("plot_type must be '3d', 'contour', or 'slices'.")

    def plot_cond_distr_2(self, *, plot_type="3d", log_z=False, **kwargs):
        """Not available: q(v) is implicit and prevents a closed form."""
        raise NotImplementedError(
            "cond_distr_2 is not available due to the implicit function q(v)."
        )


ClampedParabolaCopula: TypeAlias = XiNuBoundaryCopula

if __name__ == "__main__":
    b_values = [0.5, 1, 2]  # corresponds to mu = 0.2, 0.5, 1.0, 2.0
    for b in b_values:
        copula = XiNuBoundaryCopula(b=b)
        copula.plot_pdf(plot_type="contour")
        copula.plot_cond_distr_1(plot_type="contour")
        # copula.plot_cond_distr_2(plot_type="contour")
        copula.plot_cdf(plot_type="contour")
        xi = copula.chatterjees_xi()
        nu = copula.blests_nu()

        ccop = copula.to_checkerboard(grid_size=50)
        xi_approx = ccop.chatterjees_xi()
        nu_approx = ccop.blests_nu()
        print(f"--- Copula with b = {copula.b} ---")
        print(f"Exact xi = {xi:.6f}, nu = {nu:.6f}")
        print(f"Approx xi = {xi_approx:.6f}, nu = {nu_approx:.6f}")
        # copula.plot_cond_distr_1(plot_type="contour")
        copula.plot_pdf(
            plot_type="contour", zlim=(0, 5 * np.sqrt(b)), levels=900, grid_size=900
        )
