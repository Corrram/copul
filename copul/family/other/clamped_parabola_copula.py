import sympy as sp
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import types

from copul.family.core.biv_copula import BivCopula
from copul.family.other.biv_independence_copula import BivIndependenceCopula
from copul.family.other.upper_frechet import UpperFrechet

# Suppress potential integration warnings in normal use
import warnings
from scipy.integrate import IntegrationWarning

warnings.filterwarnings("ignore", category=IntegrationWarning)


class ClampedParabolaCopula(BivCopula):
    r"""
    Clamped-parabola copula that maximizes Blest's ν for a given Chatterjee's ξ.

    This copula family arises from a KKT analysis of the variational problem of
    maximizing ν(C) for a fixed ξ(C). The partial derivative h(t,v) = ∂₁C(t,v)
    takes the form of a clamped, decreasing, convex parabola:
    \[
      h_v(t) \;=\; \mathrm{clamp}\!\left(\frac{(1-t)^2 - q(v)}{\mu},\,0,\,1\right)
    \]
    The function q(v) is determined implicitly by the marginal constraint
    ∫₀¹ hᵥ(t) dt = v, and must be found numerically.
    """

    # Symbolic parameter & admissible interval
    mu = sp.symbols("mu", positive=True)
    params = [mu]
    intervals = {"mu": sp.Interval.open(0, sp.oo)}
    special_cases = {0: UpperFrechet, sp.oo: BivIndependenceCopula}

    # Convenience symbols
    u, v = sp.symbols("u v", positive=True)

    def __init__(self, *args, **kwargs):
        if args and len(args) == 1:
            kwargs["mu"] = args[0]
        super().__init__(**kwargs)
        self._q_cache = {}

    # ===================================================================
    # START: Core numerical methods for solving the implicit function q(v)
    # ===================================================================

    @staticmethod
    def _marginal_integral_residual(q, v_target, mu):
        """Calculates the residual F(q) = (∫ hᵥ(t) dt) - v for a given q."""
        if q > 1 or q < -mu:
            return 1e6

        s_v = 1.0 if q < 0 else 1.0 - np.sqrt(q)
        a_v = max(0, 1 - np.sqrt(q + mu))

        integral = a_v
        val_at_s = -((1 - s_v) ** 3) / 3 - q * s_v
        val_at_a = -((1 - a_v) ** 3) / 3 - q * a_v
        integral += (val_at_s - val_at_a) / mu
        return integral - v_target

    def _get_q_v(self, v_val, mu_val):
        """Numerically solves for q(v) for a single SCALAR value v."""
        if not (0 < v_val < 1):
            return v_val

        cache_key = (v_val, mu_val)
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]

        try:
            q_val = brentq(
                self._marginal_integral_residual, -mu_val, 1, args=(v_val, mu_val)
            )
            self._q_cache[cache_key] = q_val
            return q_val
        except ValueError:
            resid_at_lower = self._marginal_integral_residual(-mu_val, v_val, mu_val)
            if np.isclose(resid_at_lower, 0):
                return -mu_val
            resid_at_upper = self._marginal_integral_residual(1, v_val, mu_val)
            if np.isclose(resid_at_upper, 0):
                return 1.0
            raise RuntimeError(
                f"Failed to find root q for v={v_val}, mu={mu_val}. "
                f"Residuals at bounds F(-mu)={resid_at_lower:.3g}, F(1)={resid_at_upper:.3g}"
            )

    def _get_q_v_vec(self, v_arr, mu_val):
        """Vectorized wrapper for _get_q_v that handles any array shape."""
        v_arr = np.asarray(v_arr)
        original_shape = v_arr.shape
        v_flat = v_arr.flatten()
        q_flat = np.array([self._get_q_v(v, mu_val) for v in v_flat])
        return q_flat.reshape(original_shape)

    # ===================================================================
    # START: Rich plotting capabilities
    # ===================================================================

    def _plot3d(self, func, title, zlabel, zlim=None, **kwargs):
        """Overrides base 3D plot to use the numerical solver."""
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            f = func
        else:
            if isinstance(func, types.MethodType):
                func = func()

            def q_func(v_val):
                return self._get_q_v_vec(v_val, float(self.mu))

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
        self, func, title, zlabel, *, levels=50, zlim=None, log_z=False, **kwargs
    ):
        """Overrides base contour plot to use the numerical solver."""
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            f = func
        else:
            if isinstance(func, types.MethodType):
                func = func()

            def q_func(v_val):
                return self._get_q_v_vec(v_val, float(self.mu))

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
            norm = mcolors.LogNorm(vmin=np.ma.masked_less(Z, 1e-9).min(), vmax=Z.max())
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis", norm=norm)
        else:
            cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")

        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(zlabel)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(title)
        plt.show()
        return fig

    def _plot_functions(self, func, title, zlabel, xlabel="u", **kwargs):
        """Overrides base function plot to use the numerical solver."""
        if hasattr(func, "__name__") and func.__name__ in (
            "cdf_vectorized",
            "pdf_vectorized",
        ):
            f = func
        else:
            if isinstance(func, types.MethodType):
                func = func()

            def q_func(v_val):
                return self._get_q_v_vec(v_val, float(self.mu))

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
        """Overrides base method to use numerical cdf_vectorized."""
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
        """Overrides base method to use numerical pdf_vectorized."""
        title = kwargs.pop("title", "Probability Density Function")
        zlabel = kwargs.pop("zlabel", "PDF")

        if plot_type == "3d":
            return self._plot3d(self.pdf_vectorized, title, zlabel, **kwargs)
        elif plot_type == "contour":
            return self._plot_contour(
                self.pdf_vectorized, title, zlabel, log_z=log_z, **kwargs
            )
        else:
            raise ValueError(f"plot_type must be '3d' or 'contour', not {plot_type}")

    def plot_cond_distr_2(self, *, plot_type="3d", log_z=False, **kwargs):
        """Overrides base method to indicate it is not available."""
        raise NotImplementedError(
            "cond_distr_2 is not available due to the implicit function q(v)."
        )

    # ===================================================================
    # START: Vectorized CDF and PDF implementations
    # ===================================================================

    @property
    def cdf(self):
        return self._cdf_expr

    def cdf_vectorized(self, u, v):
        """Vectorized implementation of the cumulative distribution function."""
        u, v = np.asarray(u), np.asarray(v)
        mu = float(self.mu)

        q = self._get_q_v_vec(v, mu)
        s = np.where(q < 0, 1.0, 1.0 - np.sqrt(q))
        a = np.maximum(0, 1 - np.sqrt(q + mu))

        val_at_u = -((1 - u) ** 3) / 3 - q * u
        val_at_a = -((1 - a) ** 3) / 3 - q * a
        middle = a + (val_at_u - val_at_a) / mu

        return np.select([u <= a, u <= s], [u, middle], default=v)

    def pdf_vectorized(self, u, v):
        """Vectorized implementation of the probability density function."""
        u, v = np.atleast_1d(u), np.atleast_1d(v)
        pdf_vals = np.zeros_like(u, dtype=float)
        mu = float(self.mu)

        for i in np.ndindex(u.shape):
            eps = 1e-7
            v_i = v[i]
            q_v = self._get_q_v(v_i, mu)
            h_v = np.clip(((1 - u[i]) ** 2 - q_v) / mu, 0, 1)
            q_v_eps = self._get_q_v(min(v_i + eps, 1.0), mu)
            h_v_eps = np.clip(((1 - u[i]) ** 2 - q_v_eps) / mu, 0, 1)
            pdf_vals[i] = (h_v_eps - h_v) / eps

        return pdf_vals.reshape(np.asarray(u).shape)

    # ===================================================================
    # START: SymPy expressions and correlation measures
    # ===================================================================

    def cond_distr_1(self):
        """Symbolic expression for h(u,v) = ∂C/∂u."""
        q = sp.Function("q")(self.v)
        return sp.Min(sp.Max(0, ((1 - self.u) ** 2 - q) / self.mu), 1)

    @property
    def _cdf_expr(self):
        """Returns the integral form of the CDF for symbolic operations."""
        return sp.Integral(self.cond_distr_1(), (self.u, 0, self.u))

    def _pdf_expr(self):
        """Symbolic PDF is not available."""
        raise NotImplementedError(
            "Symbolic PDF is not available. Use `pdf_vectorized` instead."
        )

    @classmethod
    def from_xi(cls, x_target):
        """Instantiates the copula from a target value for Chatterjee's xi."""
        if not (0 < x_target < 1):
            raise ValueError("Target xi must be in (0, 1).")

        def xi_of_mu(mu):
            return cls(mu=mu).chatterjees_xi() - x_target

        mu_val = brentq(xi_of_mu, 1e-6, 1000)
        return cls(mu=mu_val)

    def chatterjees_xi(self):
        """Calculates Chatterjee's ξ via numerical integration."""
        mu = float(self.mu)

        def h_squared(t, v):
            return np.clip(((1 - t) ** 2 - self._get_q_v(v, mu)) / mu, 0, 1) ** 2

        def inner_int(v):
            return quad(h_squared, 0, 1, args=(v,))[0]

        return 6 * quad(inner_int, 0, 1)[0] - 2

    def nu(self):
        """Calculates Blest's ν via numerical integration."""
        mu = float(self.mu)

        def nu_integrand(t, v):
            return (1 - t) ** 2 * np.clip(
                ((1 - t) ** 2 - self._get_q_v(v, mu)) / mu, 0, 1
            )

        def inner_int(v):
            return quad(nu_integrand, 0, 1, args=(v,))[0]

        return 12 * quad(inner_int, 0, 1)[0] - 2


if __name__ == "__main__":
    mu_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for mu in mu_values:
        # 1. Create a copula with a specific mu
        copula = ClampedParabolaCopula(mu=mu)
        print(f"--- Copula with mu = {copula.mu} ---")

        # 2. Demonstrate the rich plotting capabilities
        print("Generating 3D CDF plot...")
        # copula.plot_cdf()

        print("Generating PDF contour plot (log scale)...")
        # copula.plot_pdf(plot_type="contour", log_z=True)

        print("Generating conditional distribution function plot (slices)...")
        copula.plot_cond_distr_1(plot_type="contour")

        # 3. Demonstrate error for unsupported plot
        # try:
        #     copula.plot_cond_distr_2()
        # except NotImplementedError as e:
        #     print(f"\nSuccessfully caught expected error: {e}")
