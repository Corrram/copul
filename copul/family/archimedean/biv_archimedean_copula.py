import logging
from abc import ABC
from functools import cached_property

import numpy as np
import sympy
import matplotlib.pyplot as plt

from copul.family.archimedean.archimedean_copula import ArchimedeanCopula
from copul.family.core.biv_core_copula import BivCoreCopula
from copul.family.helpers import concrete_expand_log, get_simplified_solution
from copul.family.copula_graphs import CopulaGraphs
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper
from scipy import optimize

log = logging.getLogger(__name__)


class BivArchimedeanCopula(ArchimedeanCopula, BivCoreCopula, ABC):
    """
    Bivariate Archimedean Copula implementation.

    This class extends the general ArchimedeanCopula for the bivariate case,
    providing specific methods for 2D dependence measures, visualization, and analysis.

    The bivariate Archimedean copula has the form: C(u,v) = φ⁻¹(φ(u) + φ(v))
    """

    def __init__(self, *args, **kwargs):
        ArchimedeanCopula.__init__(self, *args, **kwargs)
        BivCoreCopula.__init__(self)

    @property
    def dim(self) -> int:
        """
        Return the dimension of the copula.

        Returns
        -------
        int
            Always 2 for bivariate copulas
        """
        return 2

    @dim.setter
    def dim(self, value):
        pass

    @property
    def _raw_generator(self):
        raise NotImplementedError("Subclasses must implement this property")

    @property
    def _cdf_expr(self):
        """
        Cumulative distribution function of the bivariate copula.

        Returns
        -------
        SymPyFuncWrapper
            The CDF function C(u,v)
        """
        # Handle special case for the independence copula
        if type(self).__name__ == "IndependenceCopula":
            return SymPyFuncWrapper(self.u * self.v)

        # Get the generator values at u and v
        inv_gen_at_u = self.generator.subs(self.t, self.u)
        inv_gen_at_v = self.generator.subs(self.t, self.v)

        # Sum of generator values
        sum_gen = inv_gen_at_u.func + inv_gen_at_v.func

        # Apply inverse generator with proper handling of edge cases
        # Define special cases using Piecewise
        cdf = sympy.Piecewise(
            (
                sympy.Min(self.u, self.v),
                sum_gen == 0,
            ),  # When sum is 0, take minimum of u and v
            (self.inv_generator.subs(self.y, sum_gen).func, True),  # Regular case
        )

        return get_simplified_solution(cdf)

    def cdf_vectorized(self, u, v):
        """
        Vectorized implementation of the cumulative distribution function.
        This simplified version leverages NumPy broadcasting for cleaner and more efficient code.
        """
        # 1. Standard boilerplate: validation and conversion
        u = np.asarray(u)
        v = np.asarray(v)
        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Marginals must be in [0, 1]")

        if type(self).__name__ == "IndependenceCopula":
            return u * v

        # 2. Get the numeric functions (ideally, these could also be cached)
        generator_func = self.generator.numpy_func()
        inv_generator_func = self.inv_generator.numpy_func()

        # 3. Create the output array. `np.broadcast` creates the correct shape
        #    to handle all input combinations (scalar-scalar, array-scalar, etc.)
        result = np.zeros(np.broadcast(u, v).shape, dtype=float)

        # 4. Identify where computation is needed (i.e., not on the u=0 or v=0 axes)
        #    This mask works for all input shapes thanks to broadcasting.
        compute_mask = (u > 0) & (v > 0)
        if not np.any(compute_mask):
            return result  # Return all zeros if no computation is needed

        # 5. Apply the core Archimedean formula in a single vectorized step
        u_comp, v_comp = u[compute_mask], v[compute_mask]
        gen_sum = generator_func(u_comp) + generator_func(v_comp)

        # Patch the inverse generator for edge cases (0 and inf)
        # We can do this more cleanly with np.where
        inv_gen_vals = inv_generator_func(gen_sum)
        final_vals = np.where(np.isclose(gen_sum, 0), 1.0, inv_gen_vals)
        final_vals = np.where(np.isinf(gen_sum), 0.0, final_vals)

        result[compute_mask] = final_vals
        return result

    @cached_property
    def pdf(self):
        """
        Probability density function of the bivariate copula.

        Returns
        -------
        sympy expression
            The PDF function c(u,v)
        """
        first_diff = self.cdf().diff(self.u)
        return first_diff.diff(self.v)

    @cached_property
    def first_deriv_of_inv_gen(self):
        """
        First derivative of the inverse generator function.

        Returns
        -------
        sympy expression
            The derivative φ⁻¹'(y)
        """
        diff = sympy.diff(self.inv_generator.func, self.y)
        return sympy.simplify(diff)

    @property
    def second_deriv_of_inv_gen(self):
        """
        Second derivative of the inverse generator function.

        Returns
        -------
        sympy expression
            The second derivative φ⁻¹''(y)
        """
        first_diff = self.first_deriv_of_inv_gen
        second_diff = sympy.diff(first_diff, self.y)
        return sympy.simplify(second_diff)

    def kendalls_tau(self, *args, **kwargs):
        """
        Calculate Kendall's tau for the bivariate Archimedean copula.

        Kendall's tau is a measure of concordance. For Archimedean copulas,
        it can be calculated using the generator function.

        Returns
        -------
        float or sympy expression
            Kendall's tau value
        """
        self._set_params(args, kwargs)
        inv_gen = self.generator.func
        log.debug("inv gen: ", inv_gen)
        log.debug("inv gen latex: ", sympy.latex(inv_gen))
        inv_gen_diff = sympy.diff(inv_gen, self.t)
        log.debug("inv gen diff: ", inv_gen_diff)
        log.debug("inv gen diff latex: ", sympy.latex(inv_gen_diff))
        frac = inv_gen / inv_gen_diff
        log.debug("frac: ", frac)
        log.debug("frac latex: ", sympy.latex(frac))
        integral = sympy.integrate(frac, (self.t, 0, 1))
        log.debug("integral: ", integral)
        log.debug("integral latex: ", sympy.latex(integral))
        tau = 1 + 4 * integral
        log.debug("tau: ", tau)
        log.debug("tau latex: ", sympy.latex(tau))
        return tau

    def ltd_char(self):
        """
        Calculate the LTD (left-tail decreasing) characteristic.

        Returns
        -------
        sympy expression
            The LTD characteristic
        """
        return sympy.simplify(sympy.log(self.inv_generator.func))

    def diff2_ltd_char(self):
        """
        Calculate the second derivative of the LTD characteristic.

        Returns
        -------
        sympy expression
            The second derivative of the LTD characteristic
        """
        beauty_func = self.ltd_char()
        diff2 = sympy.diff(beauty_func, self.y, 2)
        return sympy.simplify(diff2)

    @property
    def ci_char(self):
        """
        Calculate the CI (conditional independence) characteristic.

        Returns
        -------
        SymPyFuncWrapper
            The CI characteristic
        """
        minus_gen_deriv = -self.first_deriv_of_inv_gen
        beauty_deriv = concrete_expand_log(sympy.simplify(sympy.log(minus_gen_deriv)))
        return SymPyFuncWrapper(beauty_deriv)

    def first_deriv_of_ci_char(self):
        """
        Calculate the first derivative of the CI characteristic.

        Returns
        -------
        sympy expression
            The first derivative of the CI characteristic
        """
        chi_char_func = self.ci_char()
        return sympy.simplify(sympy.diff(chi_char_func, self.y))

    def second_deriv_of_ci_char(self):
        """
        Calculate the second derivative of the CI characteristic.

        Returns
        -------
        sympy expression
            The second derivative of the CI characteristic
        """
        chi_char_func_deriv = self.first_deriv_of_ci_char()
        return sympy.simplify(sympy.diff(chi_char_func_deriv, self.y))

    def tp2_char(self, u, v):
        """
        Calculate the TP2 (totally positive of order 2) characteristic.

        Parameters
        ----------
        u, v : float or sympy symbol
            The arguments for the TP2 characteristic

        Returns
        -------
        SymPyFuncWrapper
            The TP2 characteristic
        """
        second_deriv = self.second_deriv_of_inv_gen.subs([(self.u, u), (self.v, v)])
        beauty_2deriv = concrete_expand_log(sympy.simplify(sympy.log(second_deriv)))
        print(sympy.latex(second_deriv))
        return SymPyFuncWrapper(beauty_2deriv)

    def first_deriv_of_tp2_char(self):
        """
        Calculate the first derivative of the TP2 characteristic.

        Returns
        -------
        sympy expression
            The first derivative of the TP2 characteristic
        """
        mtp2_char = self.tp2_char(self.u, self.v)
        return sympy.simplify(sympy.diff(mtp2_char.func, self.y))

    def second_deriv_of_tp2_char(self):
        """
        Calculate the second derivative of the TP2 characteristic.

        Returns
        -------
        sympy expression
            The second derivative of the TP2 characteristic
        """
        return sympy.simplify(sympy.diff(self.tp2_char(self.u, self.v).func, self.y, 2))

    @property
    def log_der(self):
        """
        Calculate the logarithmic derivative.

        Returns
        -------
        tuple
            A tuple containing the log derivative and related values
        """
        minus_log_derivative = self.ci_char()
        first_deriv = self.first_deriv_of_ci_char()
        second_deriv = self.second_deriv_of_ci_char()
        return self._compute_log2_der_of(
            first_deriv, minus_log_derivative, second_deriv
        )

    @property
    def log2_der(self):
        """
        Calculate the second logarithmic derivative.

        Returns
        -------
        tuple
            A tuple containing the second log derivative and related values
        """
        log_second_derivative = self.tp2_char(self.u, self.v)
        first_deriv = self.first_deriv_of_tp2_char()
        second_deriv = self.second_deriv_of_tp2_char()
        return self._compute_log2_der_of(
            first_deriv, log_second_derivative, second_deriv
        )

    def _compute_log2_der_of(self, first_deriv, log_second_derivative, second_deriv):
        """
        Helper method to compute logarithmic derivatives.

        Parameters
        ----------
        first_deriv : sympy expression
            The first derivative
        log_second_derivative : SymPyFuncWrapper
            The logarithm of the second derivative
        second_deriv : sympy expression
            The second derivative

        Returns
        -------
        tuple
            A tuple containing the results
        """
        log_der_lambda = sympy.lambdify([(self.y, self.theta)], second_deriv)
        bounds = [(self._t_min, self._t_max), (self.theta_min, self.theta_max)]
        starting_point = np.array(
            [
                min(self._t_min + 0.5, self._t_max),
                min(self.theta_min + 0.5, self.theta_max),
            ]
        )
        min_val = optimize.minimize(log_der_lambda, starting_point, bounds=bounds)
        return (
            log_second_derivative,
            first_deriv,
            second_deriv,
            [round(val, 2) for val in min_val.x],
            round(log_der_lambda(min_val.x), 2),
        )

    def lambda_L(self):
        """
        Calculate the lower tail dependence coefficient.

        Returns
        -------
        float or sympy expression
            The lower tail dependence coefficient
        """
        expr = self.inv_generator(y=2 * self.y).func / self.inv_generator(y=self.y).func
        return sympy.limit(expr, self.y, sympy.oo, dir="-")

    def lambda_U(self):
        """
        Calculate the upper tail dependence coefficient.

        Returns
        -------
        float or sympy expression
            The upper tail dependence coefficient
        """
        expr = (1 - self.inv_generator(y=2 * self.y).func) / (
            1 - self.inv_generator(y=self.y).func
        )
        return sympy.simplify(2 - sympy.limit(expr, self.y, 0, dir="+"))

    def blomqvists_beta(self, *args, **kwargs):
        r"""Blomqvist's :math:`\beta` for Archimedean copulas.

        Uses the generator-based formula:

        .. math::

           \beta = 4\,\varphi^{[-1]}\!\bigl(2\,\varphi(\tfrac12)\bigr) - 1

        where :math:`\varphi` is the generator and
        :math:`\varphi^{[-1]}` is the (pseudo-)inverse generator.

        Returns
        -------
        float or sympy.Expr
        """
        self._set_params(args, kwargs)
        # φ(1/2)
        gen_half = self.generator(t=sympy.Rational(1, 2)).func
        # φ^{-1}(2·φ(1/2))
        c_half = self.inv_generator(y=2 * gen_half).func
        result = 4 * c_half - 1
        try:
            return float(result)
        except (TypeError, ValueError):
            return sympy.simplify(result)

    def tail_order(self):
        r"""Tail order for Archimedean copulas.

        Lower tail order:

        .. math::

           \kappa_L = \lim_{s\to\infty}
                      \frac{\log\,\varphi^{[-1]}(2s)}{\log\,\varphi^{[-1]}(s)}

        Upper tail order is determined via the survival copula.

        Computed numerically from the generator.

        Returns
        -------
        dict
            ``{"lower": kappa_L, "upper": kappa_U}``
        """
        import numpy as _np

        try:
            gen_np = sympy.lambdify(self.t, self.generator.func, "numpy")
            inv_np = sympy.lambdify(self.y, self.inv_generator.func, "numpy")
        except Exception:
            # Fall back to base class numerical approach
            return super().tail_order()

        # Lower tail order: kappa_L via log(φ^{-1}(2s)) / log(φ^{-1}(s))
        try:
            ss = _np.array([10.0, 50.0, 100.0, 500.0, 1000.0])
            inv_s = _np.array([float(inv_np(si)) for si in ss])
            inv_2s = _np.array([float(inv_np(2.0 * si)) for si in ss])
            pos = (inv_s > 0) & (inv_2s > 0)
            if _np.sum(pos) >= 2:
                ratios = _np.log(inv_2s[pos]) / _np.log(inv_s[pos])
                kappa_L = float(_np.median(ratios))
            else:
                kappa_L = float("inf")
        except Exception:
            kappa_L = float("inf")

        # Upper tail order: kappa_U via log(1 - φ^{-1}(2s)) / log(1 - φ^{-1}(s))
        try:
            ss_u = _np.array([0.001, 0.005, 0.01, 0.05, 0.1])
            inv_s_u = _np.array([float(inv_np(si)) for si in ss_u])
            inv_2s_u = _np.array([float(inv_np(2.0 * si)) for si in ss_u])
            surv_s = 1.0 - inv_s_u
            surv_2s = 1.0 - inv_2s_u
            pos_u = (surv_s > 0) & (surv_2s > 0)
            if _np.sum(pos_u) >= 2:
                ratios_u = _np.log(surv_2s[pos_u]) / _np.log(surv_s[pos_u])
                kappa_U = float(_np.median(ratios_u))
            else:
                kappa_U = float("inf")
        except Exception:
            kappa_U = float("inf")

        return {"lower": kappa_L, "upper": kappa_U}

    def plot_generator(self, start=0, stop=1):
        """
        Plot the generator and inverse generator functions.

        Parameters
        ----------
        start : float, optional
            Start value for the x-axis
        stop : float, optional
            End value for the x-axis
        """
        generator = sympy.lambdify(self.t, self.generator.func)
        inv_generator = sympy.lambdify(self.y, self.inv_generator.func)
        x = np.linspace(start, stop, 1000)
        y = [generator(i) for i in x]
        z = [inv_generator(i) for i in x]
        plt.plot(x, y, label="Generator $\\varphi$")
        plt.plot(x, z, label="Inverse generator $\\psi$")
        title = CopulaGraphs(self).get_copula_title()
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
        return
