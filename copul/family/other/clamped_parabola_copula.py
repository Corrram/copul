import sympy as sp
import numpy as np
from scipy.optimize import brentq
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
    Clamped–parabola copula that maximizes Blest's :math:`\nu` for a given
    Chatterjee's :math:`\xi`.

    This family arises from a KKT analysis of the variational problem
    maximizing :math:`\nu(C)` subject to a fixed :math:`\xi(C)`. The
    partial derivative :math:`h(t,v)=\partial_1 C(t,v)` has the form of a
    clamped, decreasing, convex parabola:

    .. math::

       h_v(t) \;=\; \mathrm{clamp}\!\left(\frac{(1-t)^2 - q(v)}{\mu},\,0,\,1\right).

    The function :math:`q(v)` is determined implicitly by the marginal
    constraint :math:`\int_0^1 h_v(t)\,dt = v` and is found numerically.
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
        r"""
        Residual :math:`F(q) = \left(\int_0^1 h_v(t)\,dt\right) - v`
        for a given :math:`q`.

        Parameters
        ----------
        q : float
            Candidate value for :math:`q(v)`.
        v_target : float
            Target :math:`v \in (0,1)`.
        mu : float
            Model parameter :math:`\mu>0`.

        Returns
        -------
        float
            Residual value.
        """
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
        r"""
        Solve for :math:`q(v)` at a single scalar :math:`v`.

        Parameters
        ----------
        v_val : float
            Point :math:`v \in [0,1]`.
        mu_val : float
            Parameter :math:`\mu>0`.

        Returns
        -------
        float
            The root :math:`q(v)`.
        """
        # --- CORRECTED CODE START ---
        # Handle boundary cases v=0 and v=1 analytically, which is required
        # for the copula properties to hold.
        if v_val == 0.0:
            return 1.0
        if v_val == 1.0:
            return -mu_val
        # --- CORRECTED CODE END ---

        cache_key = (v_val, mu_val)
        if cache_key in self._q_cache:
            return self._q_cache[cache_key]

        try:
            # Search on the open interval for v in (0, 1)
            q_val = brentq(
                self._marginal_integral_residual, -mu_val, 1.0, args=(v_val, mu_val)
            )
            self._q_cache[cache_key] = q_val
            return q_val
        except ValueError:
            # Check if root is extremely close to a boundary
            resid_at_lower = self._marginal_integral_residual(-mu_val, v_val, mu_val)
            if np.isclose(resid_at_lower, 0):
                return -mu_val
            resid_at_upper = self._marginal_integral_residual(1, v_val, mu_val)
            if np.isclose(resid_at_upper, 0):
                return 1.0
            # If not, raise a more informative error
            raise RuntimeError(
                f"Failed to find root q for v={v_val}, mu={mu_val}. "
                f"Residuals at bounds F(-mu)={resid_at_lower:.3g}, F(1)={resid_at_upper:.3g}"
            )

    def _get_q_v_vec(self, v_arr, mu_val):
        r"""
        Vectorized wrapper around :meth:`_get_q_v` that accepts arrays.

        Parameters
        ----------
        v_arr : array_like
            Array of :math:`v` values.
        mu_val : float
            Parameter :math:`\mu>0`.

        Returns
        -------
        numpy.ndarray
            Array of :math:`q(v)`.
        """
        v_arr = np.asarray(v_arr)
        original_shape = v_arr.shape
        v_flat = v_arr.flatten()
        q_flat = np.array([self._get_q_v(v, mu_val) for v in v_flat])
        return q_flat.reshape(original_shape)

    # ===================================================================
    # START: Rich plotting capabilities
    # ===================================================================

    def _plot3d(self, func, title, zlabel, zlim=None, **kwargs):
        r"""
        Internal 3D surface plot using the numerical solver for :math:`q(v)`.
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
        """
        Internal contour plot using the numerical solver for :math:`q(v)`.
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
        """
        Internal line plots (slices) using the numerical solver for :math:`q(v)`.
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

    def plot_cond_distr_1(self, *, plot_type="3d", log_z=False, **kwargs):
        """
        Plot the conditional distribution h(u,v) = ∂_u C(u,v).

        This function represents the clamped-parabola shape for different slices of v.

        Parameters
        ----------
        plot_type : str, optional
            Type of plot to generate. One of '3d', 'contour', or 'slices'.
            Default is '3d'.
        log_z : bool, optional
            Whether to use a logarithmic scale for the z-axis in contour plots.
            Default is False.
        **kwargs
            Additional keyword arguments passed to the internal plotting functions.
        """
        title = kwargs.pop("title", "Conditional Distribution h(u,v)")
        zlabel = kwargs.pop("zlabel", "h(u,v)")
        # Get the symbolic expression for h(u,v)
        func = self.cond_distr_1()

        if plot_type == "3d":
            return self._plot3d(func, title, zlabel, zlim=(0, 1), **kwargs)
        elif plot_type == "contour":
            return self._plot_contour(
                func, title, zlabel, zlim=(0, 1), log_z=log_z, **kwargs
            )
        elif plot_type == "slices":
            return self._plot_functions(func, title, zlabel, **kwargs)
        else:
            raise ValueError(
                f"plot_type must be '3d', 'contour', or 'slices', not {plot_type}"
            )

    def plot_cond_distr_2(self, *, plot_type="3d", log_z=False, **kwargs):
        """Not available: :math:`q(v)` is implicit and prevents a closed form."""
        raise NotImplementedError(
            "cond_distr_2 is not available due to the implicit function q(v)."
        )

    # ===================================================================
    # START: Vectorized CDF and PDF implementations
    # ===================================================================

    @property
    def cdf(self):
        """Symbolic CDF (returned as a SymPy integral); for numerics use :meth:`cdf_vectorized`."""
        return self._cdf_expr

    def cdf_vectorized(self, u, v):
        """
        Vectorized cumulative distribution function.

        Parameters
        ----------
        u, v : array_like
            Points in :math:`[0,1]`.

        Returns
        -------
        numpy.ndarray
            Values :math:`C(u,v)`.
        """
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
        """
        Vectorized probability density function computed via a finite-difference
        approximation in :math:`v`.

        Parameters
        ----------
        u, v : array_like
            Points in :math:`[0,1]`.

        Returns
        -------
        numpy.ndarray
            Values of the PDF.
        """
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
        r"""
        Symbolic expression for :math:`h(u,v)=\partial_1 C(u,v)`.

        Returns
        -------
        sympy.Expr
            Piecewise-clamped parabola in symbolic form.
        """
        q = sp.Function("q")(self.v)
        return sp.Min(sp.Max(0, ((1 - self.u) ** 2 - q) / self.mu), 1)

    @property
    def _cdf_expr(self):
        r"""
        Symbolic integral representation of the CDF,

        .. math::

           C(u,v) \;=\; \int_0^u h(t,v)\,dt .
        """
        return sp.Integral(self.cond_distr_1(), (self.u, 0, self.u))

    def _pdf_expr(self):
        """No closed-form symbolic PDF; use :meth:`pdf_vectorized` instead."""
        raise NotImplementedError(
            "Symbolic PDF is not available. Use `pdf_vectorized` instead."
        )

    @classmethod
    def from_xi(cls, x_target):
        r"""
        Construct a copula with target Chatterjee's :math:`\xi`.

        Uses the corrected closed-form for :math:`\xi(\mu)`, which is strictly
        decreasing in :math:`\mu \in (0,\infty)`, with limits
        :math:`\lim_{\mu\downarrow 0}\xi=1` and :math:`\lim_{\mu\uparrow\infty}\xi=0`.

        Parameters
        ----------
        x_target : float
            Desired :math:`\xi \in (0,1)`.

        Returns
        -------
        ClampedParabolaCopula
            Instance with parameter :math:`\mu` chosen to match :math:`\xi`.
        """
        if not (0.0 < x_target < 1.0):
            raise ValueError("Target xi must be in (0, 1).")

        # Helper that uses the corrected closed-form xi
        def xi_of_mu(mu):
            return cls(mu=mu).chatterjees_xi()

        # xi(mu) is strictly decreasing. xi(1) = 32/105 ≈ 0.3047619
        xi_at_1 = xi_of_mu(1.0)

        # We want f(mu) := xi(mu) - x_target to change sign on [lo, hi]
        if x_target < xi_at_1:
            # Solution lies in mu >= 1. Start bracketing upward from 1.
            lo = 1.0
            hi = 2.0
            while xi_of_mu(hi) > x_target:
                hi *= 2.0
                if hi > 1e12:
                    raise RuntimeError(
                        "from_xi bracketing failed (upper bound exploded)."
                    )
        else:
            # Solution lies in mu <= 1. Start bracketing downward from 1.
            hi = 1.0
            lo = 0.5
            while xi_of_mu(lo) < x_target:
                lo *= 0.5
                if lo < 1e-14:
                    # Extremely close to xi = 1 -> effectively mu ~ 0
                    return cls(mu=lo)

        # Root-find with a robust bracket
        def f(mu):
            return xi_of_mu(mu) - x_target

        mu_val = brentq(f, lo, hi, maxiter=200, xtol=1e-14, rtol=1e-12)
        return cls(mu=mu_val)

    def chatterjees_xi(self):
        """
        Chatterjee's xi — corrected closed-form.

        For 0 < mu < 1:
            xi(mu) = [-105 s^8 A + 183 s^6 t - 38 s^4 t - 88 s^2 t + 112 s^2 + 48 t - 48] / (210 s^6),
            where s = sqrt(mu), t = sqrt(1 - mu), A = asinh(t/s).

        For mu >= 1:
            xi(mu) = 8(7 mu - 3) / (105 mu^3).
        """
        import numpy as _np

        mu = float(self.mu)
        if mu <= 0.0:
            raise ValueError("mu must be > 0.")

        s = _np.sqrt(mu)

        if mu < 1.0:
            t = _np.sqrt(1.0 - mu)
            A = _np.asinh(t / s)  # = acosh(1/s)
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
            # Purely algebraic branch; continuous at mu = 1
            return 8.0 * (7.0 * mu - 3.0) / (105.0 * mu**3)

    def blests_nu(self):
        """
        Blest's nu — corrected closed-form.

        For 0 < mu < 1:
            nu(mu) = [-105 s^8 A + 87 s^6 t + 250 s^4 t - 376 s^2 t + 448 s^2 + 144 t - 144] / (420 s^4),
            where s = sqrt(mu), t = sqrt(1 - mu), A = asinh(t/s).

        For mu >= 1:
            nu(mu) = 4(28 mu - 9) / (105 mu^2).
        """
        import numpy as _np

        mu = float(self.mu)
        if mu <= 0.0:
            raise ValueError("mu must be > 0.")

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
            return num / den
        else:
            # Purely algebraic branch; continuous at mu = 1
            return 4.0 * (28.0 * mu - 9.0) / (105.0 * mu**2)


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
