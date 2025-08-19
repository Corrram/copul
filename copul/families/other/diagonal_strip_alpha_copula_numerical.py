# file: copul/families/valid_diagonal_hole.py
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d

from copul.families.core.biv_copula import BivCopula
from copul.families.other.biv_independence_copula import BivIndependenceCopula


def _setup_numerical_functions_hole(alpha):
    """
    Pre-computes the numerical functions needed for the valid copula with a diagonal hole.
    """
    r = np.sqrt(alpha)
    psi_func = np.vectorize(lambda u: min(max(u - r / 2, 0), 1 - r))

    def L_strip_scalar(v):
        def integrand(u):
            return 1.0 if (psi_func(u) <= v <= psi_func(u) + r) else 0.0

        return quad(integrand, 0, 1, limit=100)[0]

    L_strip_vec = np.vectorize(L_strip_scalar)

    v_grid = np.linspace(0, 1, 201)
    L_strip_vals = L_strip_vec(v_grid)

    f_V_vals = (1 - L_strip_vals) / (1 - r + 1e-12)
    f_V_func = interp1d(
        v_grid,
        f_V_vals,
        kind="linear",
        bounds_error=False,
        fill_value=(f_V_vals[0], f_V_vals[-1]),
    )

    F_V_grid = np.cumsum(f_V_vals) * (v_grid[1] - v_grid[0])
    # Ensure the CDF grid is monotonic for interpolation
    F_V_grid = np.maximum.accumulate(F_V_grid)
    F_V_inv = interp1d(
        F_V_grid, v_grid, kind="linear", bounds_error=False, fill_value=(0, 1)
    )

    return {
        "r": r,
        "psi_func": psi_func,
        "f_V_func": f_V_func,
        "F_V_inv": F_V_inv,
        "L_strip_vec": L_strip_vec,
    }


class ValidDiagonalHoleCopula(BivCopula):
    r"""
    A valid, numerically defined copula with a diagonal "hole" of zero density.
    The density is positive on the complement of a diagonal strip.
    """

    params = ["alpha"]
    intervals = {"alpha": (0, 0.5)}

    def __new__(cls, *args, **kwargs):
        if args:
            kwargs["alpha"] = args[0]
        if "alpha" in kwargs and kwargs["alpha"] == 0:
            del kwargs["alpha"]
            return BivIndependenceCopula()
        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        if args:
            kwargs["alpha"] = args[0]
        alpha = kwargs.get("alpha", 0.1)

        init_kwargs = kwargs.copy()
        if "alpha" in init_kwargs:
            del init_kwargs["alpha"]

        super().__init__(**init_kwargs)

        if not 0 <= alpha <= 0.5:
            raise ValueError("Parameter alpha must be in [0, 0.5].")

        self.alpha = alpha

        helpers = _setup_numerical_functions_hole(alpha)
        self.r = helpers["r"]
        self._psi = helpers["psi_func"]
        self._f_V = helpers["f_V_func"]
        self._F_V_inv = helpers["F_V_inv"]
        self._L_strip = helpers["L_strip_vec"]

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def is_symmetric(self) -> bool:
        return False

    def pdf(self, u, w):
        u, w = np.asarray(u), np.asarray(w)
        v = self._F_V_inv(w)
        c_old = 1 / (1 - self.r + 1e-12)
        psi_vals = self._psi(u)
        is_outside_strip = (v < psi_vals) | (v > psi_vals + self.r)
        f_V_vals = self._f_V(v)
        density = np.divide(c_old, f_V_vals + 1e-12) * is_outside_strip
        return density

    def cdf(self, u, w):
        if isinstance(u, (list, np.ndarray)):
            u, w = np.asarray(u), np.asarray(w)
            results = np.zeros_like(u, dtype=float)
            for index in np.ndindex(u.shape):
                u_val, w_val = u[index], w[index]
                if u_val > 0 and w_val > 0:
                    val, err = dblquad(
                        lambda y, x: self.pdf(x, y),
                        0,
                        u_val,
                        lambda x: 0,
                        lambda x: w_val,
                    )
                    results[index] = val
            return results
        else:
            if u > 0 and w > 0:
                val, err = dblquad(
                    lambda y, x: self.pdf(x, y), 0, u, lambda x: 0, lambda x: w
                )
                return val
            return 0.0

    def cond_distr_1(self, u, w):
        u, w = np.asarray(u), np.asarray(w)
        v = self._F_V_inv(w)

        def get_strip_x_interval(v_s):
            r_s = self.r
            if 0 <= v_s < r_s:
                return 0, v_s + r_s / 2
            if r_s <= v_s <= 1 - r_s:
                return v_s - r_s / 2, v_s + r_s / 2
            if 1 - r_s < v_s <= 1:
                return v_s - r_s / 2, 1
            return 0, 0

        def C_old_u_given_v_scalar(u_s, v_s):
            x_start, x_end = get_strip_x_interval(v_s)
            len_intersection = max(0, min(u_s, x_end) - x_start)
            num = u_s - len_intersection
            den = 1 - self._L_strip(v_s)
            return num / (den + 1e-12)

        return np.vectorize(C_old_u_given_v_scalar)(u, v)

    def cond_distr_2(self, u, w):
        u, w = np.asarray(u), np.asarray(w)
        v = self._F_V_inv(w)
        psi_val = self._psi(u)
        num = np.minimum(v, psi_val) + np.maximum(0, v - psi_val - self.r)
        den = 1 - self.r
        return num / (den + 1e-12)

    # -------- Dependence Measures -------- #
    def psi(self):
        r"""
        Numerically computes Spearman's Footrule using an efficient double integral.
        \[ \psi(C) = 6 \int_0^1 \int_u^1 C(w|u) dw du - 2 \]
        This is much faster than the triple integral derived from the CDF.
        """

        # The integrand is the conditional distribution C(w|u) = cond_distr_2(u, w)
        # Note: dblquad expects func(y,x), so our integrand is func(w, u)
        def integrand_psi(w, u):
            return self.cond_distr_2(u, w)

        # We integrate 6 * C(w|u) over the region 0 <= u <= w <= 1
        # Outer integral for u from 0 to 1
        # Inner integral for w from u to 1
        integral_val, err = dblquad(
            integrand_psi,
            0,
            1,  # u integration limits
            lambda u: u,  # w lower integration limit
            lambda u: 1,  # w upper integration limit
        )

        return 6 * integral_val - 2

    def xi(self):
        r"""
        Numerically computes Chatterjee's Xi.
        \[ \xi(C) = 6 \int_0^1 \int_0^1 (C(w|u))^2 du dw - 2 \]
        """

        def integrand_xi(w, u):
            return self.cond_distr_2(u, w) ** 2

        integral_val, err = dblquad(integrand_xi, 0, 1, lambda u: 0, lambda u: 1)
        return 6 * integral_val - 2


if __name__ == "__main__":
    copula = ValidDiagonalHoleCopula(alpha=0.49)
    copula = copula.to_checkerboard(20)
    copula.plot_pdf(plot_type="contour")
    print(f"Instantiated ValidDiagonalHoleCopula with alpha = {copula.alpha}")

    print("\nCalculating dependence measures...")
    xi_val = copula.xi()
    print(f"Chatterjee's Xi (ξ): {xi_val:.4f}")

    print("Calculating Spearman's Footrule (ψ)... (this should be fast)")
    psi_val = copula.psi()
    print(f"Spearman's Footrule (ψ): {psi_val:.4f}")
