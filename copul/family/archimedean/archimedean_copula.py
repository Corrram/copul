import logging
from abc import ABC, abstractmethod

import sympy

from copul.family.archimedean._theory_mixins import ArchimedeanGeneratorMixin
from copul.family.core.copula import Copula
from copul.wrapper.inv_gen_wrapper import InvGenWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

log = logging.getLogger(__name__)


class ArchimedeanCopula(ArchimedeanGeneratorMixin, Copula, ABC):
    """
    General Archimedean Copula base class.

    This class provides the foundation for Archimedean copulas of any dimension.
    It handles generator functions, parameter validation, and special cases.

    Archimedean copulas are defined by a generator function φ and have the form:
    C(u₁, u₂, ..., uₙ) = φ⁻¹(φ(u₁) + φ(u₂) + ... + φ(uₙ))
    """

    _t_min = 0
    _t_max = 1
    t = sympy.symbols("t", nonnegative=True)
    y = sympy.symbols("y", nonnegative=True)
    theta = sympy.symbols("theta")
    theta_interval = None
    params = [theta]
    _generator = None
    _generator_at_0 = sympy.oo
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
                raise ValueError(f"Parameter theta must be > {lower_bound}, got {theta_val}")
            elif not left_open and theta_val < lower_bound:
                raise ValueError(f"Parameter theta must be >= {lower_bound}, got {theta_val}")

            # Check upper bound
            if right_open and theta_val >= upper_bound:
                raise ValueError(f"Parameter theta must be < {upper_bound}, got {theta_val}")
            elif not right_open and theta_val > upper_bound:
                raise ValueError(f"Parameter theta must be <= {upper_bound}, got {theta_val}")

        if "dimension" not in kwargs:
            kwargs["dimension"] = self.dim
        super().__init__(**kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}({self.theta})"

    @classmethod
    def from_generator(cls, generator, params=None):
        """
        Create an Archimedean copula from a generator function.

        Parameters
        ----------
        generator : str or sympy expression
            The generator function φ
        params : list or None
            List of parameters if needed

        Returns
        -------
        ArchimedeanCopula
            A new copula instance using the provided generator
        """
        return from_generator(generator, params=params)

    @abstractmethod
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        Returns
        -------
        bool
            True if the copula is absolutely continuous, False otherwise
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric.

        Returns
        -------
        bool
            True if the copula is symmetric, False otherwise
        """
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
        """
        Set the parameter intervals for the copula.

        Parameters
        ----------
        value : dict
            A dictionary mapping parameter names to their intervals
        """
        self.theta_interval = value.get("theta", None)

    @property
    def generator(self):
        """
        The generator function with proper edge case handling.
        Subclasses should implement _raw_generator instead of _generator.

        Returns
        -------
        SymPyFuncWrapper
            The generator function φ
        """
        # Get the raw generator from the subclass
        raw_generator = self._raw_generator

        # Create a piecewise function to handle edge cases properly
        expr = sympy.Piecewise(
            (raw_generator, self.t > 0),  # Regular case for valid t
            (self._generator_at_0, True),  # Default case for invalid values
        )

        # Substitute parameter values
        for key, value in self._free_symbols.items():
            expr = expr.subs(value, getattr(self, key))

        return SymPyFuncWrapper(expr)

    @generator.setter
    def generator(self, value):
        """
        Set the generator function.

        Parameters
        ----------
        value : sympy expression
            The generator function φ
        """
        self._raw_generator = value

    @property
    @abstractmethod
    def _raw_generator(self):
        """
        Raw generator function without edge case handling.
        This should be implemented by subclasses.

        Returns
        -------
        sympy expression
            The raw generator function φ
        """
        raise NotImplementedError("Subclasses must implement _raw_generator")

    @property
    def inv_generator(self):
        """
        The inverse generator function with proper edge case handling.
        Uses _raw_inv_generator from subclasses.

        Returns
        -------
        InvGenWrapper
            The inverse generator function φ⁻¹
        """
        # Get the raw inverse generator or compute it if not provided
        if hasattr(self, "_raw_inv_generator"):
            raw_inv = self._raw_inv_generator
        else:
            # Default implementation: compute inverse from equation
            equation = sympy.Eq(self.y, self._raw_generator)
            solutions = sympy.solve(equation, self.t)

            # Extract solution
            if isinstance(solutions, dict):
                raw_inv = solutions[self.t]
            elif isinstance(solutions, list):
                raw_inv = solutions[0]
            else:
                raw_inv = solutions

        # Return the wrapper with properly handled edge cases
        return InvGenWrapper(raw_inv, self.y, self)

    @property
    def theta_max(self):
        """
        Maximum value of the parameter theta.

        Returns
        -------
        float or sympy expression
            The maximum value of theta
        """
        return self.theta_interval.closure.end

    @property
    def theta_min(self):
        """
        Minimum value of the parameter theta.

        Returns
        -------
        float or sympy expression
            The minimum value of theta
        """
        return self.theta_interval.closure.inf

    def _archimedean_generator(self):
        r"""Vectorized numeric generator data (see
        :func:`copul.theory.archimedean.archimedean_generator`).

        Returns
        -------
        copul.family.archimedean.numeric_archimedean.ArchimedeanGenerator
        """
        from copul.family.archimedean.numeric_archimedean import generator_from_sympy

        return generator_from_sympy(self)

    def compute_gen_max(self):
        """
        Compute the maximum value of the generator function.

        Returns
        -------
        float or sympy expression
            The maximum value of the generator
        """
        try:
            limit = sympy.limit(self._generator, self.t, 0)
        except TypeError:
            limit = sympy.limit(
                self._generator.subs(self.theta, (self.theta_max - self.theta_min) / 2),
                self.t,
                0,
            )
        return sympy.simplify(limit)


def from_generator(generator, params=None, theta_interval=None):
    r"""
    Create a bivariate Archimedean copula (family) from a generator function.

    The copula is :math:`C(u, v) = \varphi^{[-1]}(\varphi(u) + \varphi(v))`
    with the pseudo-inverse :math:`\varphi^{[-1]}` of the generator
    :math:`\varphi`. The returned object is an ordinary
    :class:`~copul.family.archimedean.biv_archimedean_copula.BivArchimedeanCopula`:
    cdf, conditional distributions, dependence measures, sampling and
    plotting are available.

    Parameters
    ----------
    generator : str or sympy expression
        The generator :math:`\varphi(t)` in the variable ``t``, e.g.
        ``"(t**(-theta) - 1)/theta"``. At most one parameter is supported; it
        is exposed as ``theta`` (a parameter with another name is renamed to
        ``theta``).
    params : list, str or None
        Name(s) of the parameter symbol(s) if they cannot be inferred, i.e.
        if the variable is not called ``t``.
    theta_interval : sympy.Interval or None
        Admissible parameter values; defaults to the whole real line. The
        caller is responsible for :math:`\varphi` being a valid generator
        (continuous, strictly decreasing and convex with
        :math:`\varphi(1) = 0`) for the chosen parameters.

    Returns
    -------
    BivArchimedeanCopula
        A copula family (free ``theta``) or a copula (no parameter).

    Examples
    --------
    >>> import copul as cp
    >>> family = cp.from_generator("(t**(-theta) - 1)/theta")  # Clayton
    >>> C = family(theta=2)
    >>> bool(abs(C.cdf(0.3, 0.7) - cp.Clayton(2).cdf(0.3, 0.7)) < 1e-12)
    True
    >>> round(C.kendalls_tau(), 10)
    0.5
    """
    # local imports: the bivariate classes import this module
    from copul.family.archimedean.biv_archimedean_copula import BivArchimedeanCopula

    sp_generator = sympy.sympify(generator)
    symbols = sorted(sp_generator.free_symbols, key=str)
    if params is not None:
        names = {str(p) for p in (params if isinstance(params, (list, tuple)) else [params])}
        param_syms = [s for s in symbols if str(s) in names]
        func_vars = [s for s in symbols if str(s) not in names]
    else:
        func_vars = [s for s in symbols if str(s) == "t"]
        param_syms = [s for s in symbols if str(s) != "t"]
    if len(func_vars) != 1:
        raise ValueError(
            "The generator must depend on exactly one variable (called 't', or "
            f"pass the parameter names via params=); found {sorted(map(str, func_vars))}."
        )
    if len(param_syms) > 1:
        raise ValueError(
            "from_generator supports generators with at most one parameter; "
            f"found {sorted(map(str, param_syms))}."
        )

    t = BivArchimedeanCopula.t
    theta = ArchimedeanCopula.theta
    expr = sp_generator.subs(func_vars[0], t)
    if param_syms:
        expr = expr.subs(param_syms[0], theta)
    interval = (
        theta_interval
        if theta_interval is not None
        else sympy.Interval(-sympy.oo, sympy.oo, left_open=True, right_open=True)
    )

    class ArchimedeanFromGenerator(BivArchimedeanCopula):
        """Bivariate Archimedean copula defined by a user-supplied generator."""

        theta_interval = interval
        params = [theta] if param_syms else []
        _user_generator = expr

        @property
        def _raw_generator(self):
            return self._user_generator.subs(ArchimedeanCopula.theta, self.theta)

        @property
        def _generator_at_0(self):
            try:
                value = sympy.limit(self._raw_generator, self.t, 0, "+")
            except (NotImplementedError, TypeError, ValueError):
                return sympy.oo
            return value if value.is_finite or value == sympy.oo else sympy.oo

        @property
        def is_absolutely_continuous(self) -> bool:
            # strict generators (phi(0) = oo) give absolutely continuous
            # copulas; non-strict ones put mass on the zero curve
            return self._generator_at_0 == sympy.oo

    return ArchimedeanFromGenerator()
