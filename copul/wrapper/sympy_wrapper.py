import numpy as np
import sympy
from typing import Union, Dict, Any, Optional, Tuple, Set


class SymPyFuncWrapper:
    """
    A wrapper class for SymPy expressions that provides additional functionality
    and a more intuitive interface for working with symbolic mathematics.

    This class allows for easier function substitution, differentiation, and
    arithmetic operations with proper type handling.
    """

    def __init__(self, sympy_func: Union["SymPyFuncWrapper", sympy.Expr, float, int]):
        """
        Initialize a SymPyFuncWrapper with a SymPy expression or numeric value.

        Args:
            sympy_func: A SymPy expression, another SymPyFuncWrapper, or a numeric value.

        Raises:
            AssertionError: If the input is not a valid SymPy expression or numeric value.
        """
        if isinstance(sympy_func, SymPyFuncWrapper):
            sympy_func = sympy_func.func

        type_ = type(sympy_func)
        allowed = (sympy.Expr, float, int)
        assert isinstance(sympy_func, allowed), (
            f"Function must be from sympy, but is {type_}"
        )

        # Convert numeric types to SymPy Number
        if isinstance(sympy_func, (float, int)):
            self._func = sympy.Number(sympy_func)
        else:
            self._func = sympy_func

    def __str__(self) -> str:
        """Return a string representation of the wrapped expression."""
        return str(self._func)

    def __float__(self) -> float:
        """
        Convert the expression to a float if possible.

        Returns:
            The float value of the expression.

        Raises:
            TypeError: If the expression cannot be converted to a float.
        """
        if isinstance(self._func, (sympy.Number, float, int)):
            return float(self._func)

        result = self._func.evalf()
        if isinstance(result, sympy.Number):
            return float(result)

        raise TypeError(f"Cannot convert expression '{self._func}' to float")

    def __repr__(self) -> str:
        """Return a string representation of the wrapped expression.

        Note: For backward compatibility, this returns the representation
        of the underlying SymPy expression, not the wrapper itself.
        """
        return repr(self._func)

    def __call__(self, *args, **kwargs) -> "SymPyFuncWrapper":
        """
        Substitute values for symbols in the expression.

        Args:
            *args: Positional arguments matched to free symbols in alphabetical order.
            **kwargs: Keyword arguments matched to free symbol names.

        Returns:
            A new SymPyFuncWrapper with the substitutions applied.

        Raises:
            ValueError: If both args and kwargs are provided.
        """
        vars_, _ = self._prepare_call(args, kwargs)
        func = self._func.subs(vars_)
        return SymPyFuncWrapper(func)

    def _prepare_call(
        self, args: tuple, kwargs: dict
    ) -> Tuple[Dict[sympy.Symbol, Any], Dict[str, Any]]:
        """
        Prepare arguments for substitution in the expression.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            A tuple of (substitution_dict, processed_kwargs).

        Raises:
            ValueError: If both args and kwargs are provided.
        """
        # Get all free symbols in the expression
        symbols = list(self._func.free_symbols)
        free_symbols = sorted([str(s) for s in symbols])

        # Check if we have both args and kwargs
        non_none_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if args and non_none_kwargs:
            raise ValueError("Cannot provide both positional and keyword arguments")

        # Map positional args to symbols if appropriate
        if args and len(free_symbols) == len(args):
            kwargs = {free_sym: arg for free_sym, arg in zip(free_symbols, args)}

        # Filter out None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Create substitution dictionary
        vars_ = {}
        for sym in symbols:
            sym_str = str(sym)
            if sym_str in kwargs:
                value = kwargs[sym_str]
                # Handle nested SymPyFuncWrapper
                if isinstance(value, SymPyFuncWrapper):
                    value = value.func
                vars_[sym] = value

        return vars_, kwargs

    @property
    def func(self) -> sympy.Expr:
        """Return the underlying SymPy expression."""
        return self._func

    @property
    def free_symbols(self) -> Set[sympy.Symbol]:
        """Return the set of free symbols in the expression."""
        return self._func.free_symbols

    def subs(self, *args, **kwargs) -> "SymPyFuncWrapper":
        """
        Substitute values in the expression.

        This method directly uses SymPy's subs method.

        Returns:
            A new SymPyFuncWrapper with the substitutions applied.
        """
        result = self._func.subs(*args, **kwargs)
        return SymPyFuncWrapper(result)

    def diff(self, *args, **kwargs) -> "SymPyFuncWrapper":
        """
        Differentiate the expression.

        This method directly uses SymPy's diff method.

        Returns:
            A new SymPyFuncWrapper with the differentiated expression.
        """
        result = self._func.diff(*args, **kwargs)
        return SymPyFuncWrapper(result)

    def integrate(self, *args, **kwargs) -> "SymPyFuncWrapper":
        """
        Integrate the expression.

        This method directly uses SymPy's integrate method.

        Returns:
            A new SymPyFuncWrapper with the integrated expression.
        """
        result = sympy.integrate(self._func, *args, **kwargs)
        return SymPyFuncWrapper(result)

    def simplify(self) -> "SymPyFuncWrapper":
        """
        Simplify the expression.

        Returns:
            A new SymPyFuncWrapper with the simplified expression.
        """
        return SymPyFuncWrapper(sympy.simplify(self._func))

    def expand(self) -> "SymPyFuncWrapper":
        """
        Expand the expression.

        Returns:
            A new SymPyFuncWrapper with the expanded expression.
        """
        return SymPyFuncWrapper(self._func.expand())

    def factor(self) -> "SymPyFuncWrapper":
        """
        Factor the expression.

        Returns:
            A new SymPyFuncWrapper with the factored expression.
        """
        return SymPyFuncWrapper(sympy.factor(self._func))

    def to_latex(self) -> str:
        """
        Convert the expression to LaTeX.

        Returns:
            A LaTeX string representation of the expression.
        """
        return sympy.latex(self._func)

    def evalf(
        self, n: Optional[int] = None
    ) -> Union[float, sympy.Expr, "SymPyFuncWrapper"]:
        """
        Evaluate the expression numerically.

        Args:
            n: Optional number of significant digits.

        Returns:
            For backward compatibility:
            - For numeric expressions, returns a float
            - For expressions with symbols, returns the evalf'd SymPy expression
        """
        if n is not None:
            result = self._func.evalf(n=n)
        else:
            result = self._func.evalf()

        if isinstance(result, sympy.Number) and not self._func.free_symbols:
            return float(result)

        # For backward compatibility, return the raw SymPy expression
        # if the original expression has free symbols
        if self._func.free_symbols:
            return result

        return result

    def __eq__(self, other) -> bool:
        """
        Check if this expression equals another expression or value.

        Args:
            other: Another SymPyFuncWrapper, SymPy expression, or numeric value.

        Returns:
            True if expressions are mathematically equal, False otherwise.
        """
        try:
            if isinstance(other, SymPyFuncWrapper):
                other = other.func
            return sympy.simplify(self._func - other) == 0
        except (TypeError, ValueError):
            return False

    def __ne__(self, other) -> bool:
        """
        Check if this expression does not equal another expression or value.

        Args:
            other: Another SymPyFuncWrapper, SymPy expression, or numeric value.

        Returns:
            True if expressions are not mathematically equal, False otherwise.
        """
        return not self == other

    def isclose(self, other, tolerance: float = 1e-10) -> bool:
        """
        Check if this expression is numerically close to another expression or value.

        Args:
            other: Another SymPyFuncWrapper, SymPy expression, or numeric value.
            tolerance: Maximum absolute difference allowed for equality (default: 1e-10).

        Returns:
            True if expressions are numerically close, False otherwise.
        """
        try:
            self_val = float(self.evalf())

            if isinstance(other, SymPyFuncWrapper):
                other_val = float(other.evalf())
            else:
                try:
                    other_val = float(other)
                except (TypeError, ValueError):
                    return self == other

            return abs(self_val - other_val) < tolerance

        except (TypeError, ValueError):
            # Fall back to symbolic comparison if numeric comparison fails
            return self == other

    # Arithmetic operations with proper handling of left and right operations

    def __add__(self, other) -> "SymPyFuncWrapper":
        """Add another expression or value to this expression."""
        if isinstance(other, SymPyFuncWrapper):
            other = other.func
        return SymPyFuncWrapper(self.func + other)

    def __radd__(self, other) -> "SymPyFuncWrapper":
        """Add this expression to another value (right-side operation)."""
        return SymPyFuncWrapper(other + self.func)

    def __sub__(self, other) -> "SymPyFuncWrapper":
        """Subtract another expression or value from this expression."""
        if isinstance(other, SymPyFuncWrapper):
            other = other.func
        return SymPyFuncWrapper(self.func - other)

    def __rsub__(self, other) -> "SymPyFuncWrapper":
        """Subtract this expression from another value (right-side operation)."""
        return SymPyFuncWrapper(other - self.func)

    def __mul__(self, other) -> "SymPyFuncWrapper":
        """Multiply this expression by another expression or value."""
        if isinstance(other, SymPyFuncWrapper):
            other = other.func
        return SymPyFuncWrapper(self.func * other)

    def __rmul__(self, other) -> "SymPyFuncWrapper":
        """Multiply another value by this expression (right-side operation)."""
        return SymPyFuncWrapper(other * self.func)

    def __truediv__(self, other) -> "SymPyFuncWrapper":
        """Divide this expression by another expression or value."""
        if isinstance(other, SymPyFuncWrapper):
            other = other.func
        return SymPyFuncWrapper(self.func / other)

    def __rtruediv__(self, other) -> "SymPyFuncWrapper":
        """Divide another value by this expression (right-side operation)."""
        return SymPyFuncWrapper(other / self.func)

    def __pow__(self, other) -> "SymPyFuncWrapper":
        """Raise this expression to the power of another expression or value."""
        if isinstance(other, SymPyFuncWrapper):
            other = other.func
        return SymPyFuncWrapper(self.func**other)

    def __rpow__(self, other) -> "SymPyFuncWrapper":
        """Raise another value to the power of this expression (right-side operation)."""
        return SymPyFuncWrapper(other**self.func)

    def __neg__(self) -> "SymPyFuncWrapper":
        """Negate this expression."""
        return SymPyFuncWrapper(-self.func)

    def __abs__(self) -> "SymPyFuncWrapper":
        """Return the absolute value of this expression."""
        return SymPyFuncWrapper(sympy.Abs(self.func))

    def __hash__(self) -> int:
        """Return a hash of this expression for use in dictionaries and sets."""
        return hash(self.func)

    def numpy(self) -> np.ndarray:
        """
        Convert the expression to a numpy function.

        Returns:
            A numpy function that evaluates the expression with numpy inputs.

        Raises:
            ValueError: If the expression cannot be converted to a numpy function.
        """
        try:
            from sympy.utilities.lambdify import lambdify

            symbols = sorted(self.free_symbols, key=lambda s: str(s))
            if not symbols:  # Constant expression
                return np.array(float(self.evalf()))
            func = lambdify(symbols, self.func, "numpy")
            return func
        except Exception as e:
            raise ValueError(f"Failed to convert to numpy function: {e}")
