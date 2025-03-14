import logging
from abc import ABC

import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy import optimize

from copul.families.helpers import concrete_expand_log, get_simplified_solution
from copul.families.bivcopula import BivCopula
from copul.families.copula_graphs import CopulaGraphs
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

log = logging.getLogger(__name__)


class ArchimedeanCopula(BivCopula, ABC):
    _t_min = 0
    _t_max = 1
    y, t = sympy.symbols("y t", nonnegative=True)
    theta = sympy.symbols("theta")
    theta_interval = None
    params = [theta]
    _generator = None
    # Dictionary mapping parameter values to special case classes
    special_cases = {}  # To be overridden by subclasses
    # Set of parameter values that are invalid (will raise ValueError)
    invalid_params = set()  # To be overridden by subclasses

    @classmethod
    def create(cls, *args, **kwargs):
        """Factory method to create the appropriate copula instance based on parameters."""
        # Handle positional arguments
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]

        # Check for invalid parameters
        if "theta" in kwargs and kwargs["theta"] in cls.invalid_params:
            raise ValueError(f"Parameter theta cannot be {kwargs['theta']}")

        # Check for special cases
        if "theta" in kwargs and kwargs["theta"] in cls.special_cases:
            special_case_cls = cls.special_cases[kwargs["theta"]]
            del kwargs["theta"]  # Remove theta before creating special case
            return special_case_cls()

        # Otherwise create a normal instance
        return cls(**kwargs)

    def __new__(cls, *args, **kwargs):
        """Override __new__ to handle special cases."""
        # Handle positional arguments
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]

        # Check for invalid parameters
        if "theta" in kwargs and kwargs["theta"] in cls.invalid_params:
            raise ValueError(f"Parameter theta cannot be {kwargs['theta']}")

        # Check for special cases
        if "theta" in kwargs and kwargs["theta"] in cls.special_cases:
            special_case_cls = cls.special_cases[kwargs["theta"]]
            del kwargs["theta"]  # Remove theta before creating special case
            return special_case_cls()

        # Standard creation for normal cases
        return super().__new__(cls)

    def __call__(self, *args, **kwargs):
        """Handle special cases when calling the instance."""
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]

        # Check for invalid parameters
        if "theta" in kwargs and kwargs["theta"] in self.__class__.invalid_params:
            raise ValueError(f"Parameter theta cannot be {kwargs['theta']}")

        # Check for special cases
        if "theta" in kwargs and kwargs["theta"] in self.__class__.special_cases:
            special_case_cls = self.__class__.special_cases[kwargs["theta"]]
            del kwargs["theta"]  # Remove theta before creating special case
            return special_case_cls()

        # Create a new instance with updated parameters
        # Merge existing parameters with new ones
        new_kwargs = {**self._free_symbols}
        new_kwargs.update(kwargs)
        return self.__class__(**new_kwargs)

    # Rest of the class implementation remains the same...
    def __init__(self, *args, **kwargs):
        """Initialize an Archimedean copula with parameter validation."""
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]

        # Validate theta parameter against theta_interval if defined
        if "theta" in kwargs and self.theta_interval is not None:
            theta_val = kwargs["theta"]

            # Extract bounds from the interval
            lower_bound = float(self.theta_interval.start)
            upper_bound = float(self.theta_interval.end)
            left_open = self.theta_interval.left_open
            right_open = self.theta_interval.right_open

            # Check lower bound
            if left_open and theta_val <= lower_bound:
                raise ValueError(
                    f"Parameter theta must be > {lower_bound}, got {theta_val}"
                )
            elif not left_open and theta_val < lower_bound:
                raise ValueError(
                    f"Parameter theta must be >= {lower_bound}, got {theta_val}"
                )

            # Check upper bound
            if right_open and theta_val >= upper_bound:
                raise ValueError(
                    f"Parameter theta must be < {upper_bound}, got {theta_val}"
                )
            elif not right_open and theta_val > upper_bound:
                raise ValueError(
                    f"Parameter theta must be <= {upper_bound}, got {theta_val}"
                )

        super().__init__(**kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}({self.theta})"

    @classmethod
    def from_generator(cls, generator, params=None):
        sp_generator = sympy.sympify(generator)
        func_vars, params = cls._segregate_symbols(sp_generator, "t", params)
        obj = cls._from_string(params)
        obj._generator = sp_generator.subs(func_vars[0], cls.t)
        return obj

    @property
    def is_absolutely_continuous(self) -> bool:
        raise NotImplementedError("This method should be implemented in the subclass")

    @property
    def is_symmetric(self) -> bool:
        return True

    @property
    def intervals(self):
        """
        Return the parameter intervals for the copula.

        Returns
        -------
        dict
            A dictionary mapping parameter names to their corresponding intervals.
            For example, if ``self.theta_interval`` is defined, returns
            ``{"theta": self.theta_interval}``; otherwise, returns an empty dictionary.
        """
        return {"theta": self.theta_interval} if self.theta_interval is not None else {}

    @intervals.setter
    def intervals(self, value):
        self.theta_interval = value["theta"] if "theta" in value else None

    @property
    def generator(self):
        expr = self._generator
        for key, value in self._free_symbols.items():
            expr = expr.subs(value, getattr(self, key))
        return SymPyFuncWrapper(expr)

    @generator.setter
    def generator(self, value):
        self._generator = value

    @property
    def theta_max(self):
        return self.theta_interval.closure.end

    @property
    def theta_min(self):
        return self.theta_interval.closure.inf

    @property
    def cdf(self):
        """Cumulative distribution function of the copula"""
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
            (0, sympy.Or(self.u == 0, self.v == 0)),  # If either u or v is 0, CDF is 0
            (sympy.Min(self.u, self.v), sum_gen == 0),  # When sum is 0, take minimum of u and v
            (self.inv_generator.subs(self.y, sum_gen).func, True)  # Regular case
        )
        
        return SymPyFuncWrapper(get_simplified_solution(cdf))

    def cdf_vectorized(self, u, v):
        """
        Vectorized implementation of the cumulative distribution function.
        This method evaluates the CDF at multiple points simultaneously,
        which is more efficient than calling the scalar CDF function repeatedly.
        
        Parameters
        ----------
        u : array_like
            First uniform marginal, should be in [0, 1].
        v : array_like
            Second uniform marginal, should be in [0, 1].
            
        Returns
        -------
        numpy.ndarray
            The CDF values at the specified points.
            
        Notes
        -----
        This implementation uses numpy for vectorized operations, which
        provides significant performance improvements for large inputs.
        """
        # Convert inputs to numpy arrays if they aren't already
        u = np.asarray(u)
        v = np.asarray(v)
        
        # Ensure inputs are within [0, 1]
        if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
            raise ValueError("Marginals must be in [0, 1]")
        
        # Special case for the independence copula
        if type(self).__name__ == "IndependenceCopula":
            return u * v
        
        # Get vectorized functions for the generator and inverse generator
        generator_func = self.generator.numpy_func()
        inv_generator_func = self.inv_generator.numpy_func()
        
        # Create a properly patched inverse generator function that handles edge cases
        def inv_generator_func_patched(x):
            # Use numpy functions for vectorized operations
            result = np.zeros_like(x, dtype=float)
            
            # Handle edge cases
            zero_mask = np.isclose(x, 0)
            inf_mask = np.isinf(x)
            regular_mask = ~(zero_mask | inf_mask)
            
            # Apply appropriate values for each case
            result[zero_mask] = 1.0  # inv_generator(0) = 1
            result[inf_mask] = 0.0   # inv_generator(inf) = 0
            
            # Only compute the regular case where needed
            if np.any(regular_mask):
                result[regular_mask] = inv_generator_func(x[regular_mask])
            
            return result
        
        # Handle edge cases where u=0 or v=0
        zero_mask = (u == 0) | (v == 0)
        result = np.zeros_like(u, dtype=float)
        
        # Only compute non-zero cases
        if not np.all(zero_mask):
            non_zero_mask = ~zero_mask
            u_nz = u[non_zero_mask]
            v_nz = v[non_zero_mask]
            
            # Apply the generator to each marginal
            gen_u = generator_func(u_nz)
            gen_v = generator_func(v_nz)
            
            # Sum the generator values and apply the inverse generator
            gen_sum = gen_u + gen_v
            result[non_zero_mask] = inv_generator_func_patched(gen_sum)
        
        return result

    @property
    def pdf(self):
        """Probability density function of the copula"""
        first_diff = sympy.diff(self.cdf, self.u)
        return sympy.diff(first_diff, self.v)

    @property
    def inv_generator(self) -> SymPyFuncWrapper:
        """
        Finds the inverse of the generator function for a given value of y,
        considering the condition that y > 0.

        Returns:
            The inverse value(s) of t that satisfy the condition.
        """
        # Create the equation y = generator
        equation = sympy.Eq(self.y, self.generator.func)

        # Define the conditions: the equation and y > 0
        conditions = [equation, self.y >= 0]

        # Solve the equation under the given conditions for t
        solutions = sympy.solve(conditions, self.t)

        # If the solution is a dictionary, extract the value for t
        if isinstance(solutions, dict):
            my_sol = solutions[self.t]
        elif isinstance(solutions, list):
            my_sol = solutions[0]
        else:
            my_sol = solutions

        my_simplified_sol = get_simplified_solution(my_sol)

        return SymPyFuncWrapper(my_simplified_sol)

    @property
    def first_deriv_of_inv_gen(self):
        diff = sympy.diff(self.inv_generator.func, self.y)
        return sympy.simplify(diff)

    def kendalls_tau(self, *args, **kwargs):
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

    @property
    def second_deriv_of_inv_gen(self):
        first_diff = self.first_deriv_of_inv_gen
        second_diff = sympy.diff(first_diff, self.y)
        return sympy.simplify(second_diff)

    def ltd_char(self):
        return sympy.simplify(sympy.log(self.inv_generator.func))

    def diff2_ltd_char(self):
        beauty_func = self.ltd_char()
        diff2 = sympy.diff(beauty_func, self.y, 2)
        return sympy.simplify(diff2)

    @property
    def ci_char(self):
        minus_gen_deriv = -self.first_deriv_of_inv_gen
        beauty_deriv = concrete_expand_log(sympy.simplify(sympy.log(minus_gen_deriv)))
        return SymPyFuncWrapper(beauty_deriv)

    def first_deriv_of_ci_char(self):
        chi_char_func = self.ci_char()
        return sympy.simplify(sympy.diff(chi_char_func, self.y))

    def second_deriv_of_ci_char(self):
        chi_char_func_deriv = self.first_deriv_of_ci_char()
        return sympy.simplify(sympy.diff(chi_char_func_deriv, self.y))

    def tp2_char(self, u, v):
        second_deriv = self.second_deriv_of_inv_gen.subs([(self.u, u), (self.v, v)])
        beauty_2deriv = concrete_expand_log(sympy.simplify(sympy.log(second_deriv)))
        print(sympy.latex(second_deriv))
        return SymPyFuncWrapper(beauty_2deriv)

    def first_deriv_of_tp2_char(self):
        mtp2_char = self.tp2_char(self.u, self.v)
        return sympy.simplify(sympy.diff(mtp2_char.func, self.y))

    def second_deriv_of_tp2_char(self):
        return sympy.simplify(sympy.diff(self.tp2_char(self.u, self.v).func, self.y, 2))

    @property
    def log_der(self):
        minus_log_derivative = self.ci_char()
        first_deriv = self.first_deriv_of_ci_char()
        second_deriv = self.second_deriv_of_ci_char()
        return self._compute_log2_der_of(
            first_deriv, minus_log_derivative, second_deriv
        )

    @property
    def log2_der(self):
        log_second_derivative = self.tp2_char(self.u, self.v)
        first_deriv = self.first_deriv_of_tp2_char()
        second_deriv = self.second_deriv_of_tp2_char()
        return self._compute_log2_der_of(
            first_deriv, log_second_derivative, second_deriv
        )

    def _compute_log2_der_of(self, first_deriv, log_second_derivative, second_deriv):
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

    def compute_gen_max(self):
        try:
            limit = sympy.limit(self._generator, self.t, 0)
        except TypeError:
            limit = sympy.limit(
                self._generator.subs(self.theta, (self.theta_max - self.theta_min) / 2),
                self.t,
                0,
            )
        return sympy.simplify(limit)

    def lambda_L(self):
        expr = self.inv_generator(y=2 * self.y).func / self.inv_generator(y=self.y).func
        return sympy.limit(expr, self.y, sympy.oo, dir="-")

    def lambda_U(self):
        expr = (1 - self.inv_generator(y=2 * self.y).func) / (
            1 - self.inv_generator(y=self.y).func
        )
        return sympy.simplify(2 - sympy.limit(expr, self.y, 0, dir="+"))

    def plot_generator(self, start=0, stop=1):
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
