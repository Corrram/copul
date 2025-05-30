import copy
import numpy as np
import sympy

from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.cdi_wrapper import CDiWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CoreCopula:
    """
    A unified Copula class that combines functionality previously split between
    CoreCopula and Copula classes.
    """

    params = []
    intervals = {}
    log_cut_off = 4
    _cdf_expr_internal = (
        None  # Renamed from _cdf to avoid confusion with the new method
    )
    _free_symbols = {}

    @property
    def _cdf_expr(self):
        # If we have an internal expression set, use it as a template
        if self._cdf_expr_internal is not None:
            # Make a deep copy to avoid modifying the original
            expr = self._cdf_expr_internal

            # Apply any symbol updates if _free_symbols is available
            if hasattr(self, "_free_symbols") and self._free_symbols:
                current_values = {}
                for symbol_name, symbol_obj in self._free_symbols.items():
                    # Get the current value from the object
                    if hasattr(self, symbol_name):
                        current_values[symbol_obj] = getattr(self, symbol_name)

                # Only create a new expression if we have substitutions
                if current_values:
                    expr = expr.subs(current_values)

            return expr
        return None

    @_cdf_expr.setter
    def _cdf_expr(self, value):
        self._cdf_expr_internal = value

    def __str__(self):
        return self.__class__.__name__

    def __init__(self, dimension, *args, **kwargs):
        """
        Initialize a Copula.

        Parameters
        ----------
        dimension : int
            Dimension of the copula.
        *args : tuple
            Positional arguments for parameters.
        **kwargs : dict
            Keyword arguments for parameters.
        """
        self.u_symbols = sympy.symbols(f"u1:{dimension + 1}")
        self.dim = dimension
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(self, k, v)
        self.params = [param for param in self.params if str(param) not in kwargs]
        self.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }

    def __call__(self, *args, **kwargs):
        """
        Create a new Copula instance with updated parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments for parameters.
        **kwargs : dict
            Keyword arguments for parameters.

        Returns
        -------
        Copula
            A new Copula instance with the updated parameters.
        """
        new_copula = copy.copy(self)
        self._are_class_vars(kwargs)
        for i in range(len(args)):
            kwargs[str(self.params[i])] = args[i]
        for k, v in kwargs.items():
            if isinstance(v, str):
                v = getattr(self.__class__, v)
            setattr(new_copula, k, v)
        new_copula.params = [param for param in self.params if str(param) not in kwargs]
        new_copula.intervals = {
            k: v for k, v in self.intervals.items() if str(k) not in kwargs
        }
        return new_copula

    def _set_params(self, args, kwargs):
        """
        Set parameters from args and kwargs.

        Parameters
        ----------
        args : tuple
            Positional arguments for parameters.
        kwargs : dict
            Keyword arguments for parameters.
        """
        if args and len(args) <= len(self.params):
            for i in range(len(args)):
                kwargs[str(self.params[i])] = args[i]
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    @property
    def parameters(self):
        """
        Get the parameters of the copula.

        Returns
        -------
        dict
            Dictionary of parameter intervals.
        """
        return self.intervals

    @property
    def is_absolutely_continuous(self) -> bool:
        """
        Check if the copula is absolutely continuous.

        Returns
        -------
        bool
            True if the copula is absolutely continuous, False otherwise.
        """
        # Implementations should override this method
        raise NotImplementedError("This method must be implemented in a subclass")

    @property
    def is_symmetric(self) -> bool:
        """
        Check if the copula is symmetric.

        Returns
        -------
        bool
            True if the copula is symmetric, False otherwise.
        """
        # Implementations should override this method
        raise NotImplementedError("This method must be implemented in a subclass")

    def _are_class_vars(self, kwargs):
        """
        Check if keys in kwargs are class variables.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to check.

        Raises
        ------
        AssertionError
            If any key in kwargs is not a class variable.
        """
        class_vars = set(dir(self))
        assert set(kwargs).issubset(class_vars), (
            f"keys: {set(kwargs)}, free symbols: {class_vars}"
        )

    def slice_interval(self, param, interval_start=None, interval_end=None):
        """
        Slice the interval of a parameter.

        Parameters
        ----------
        param : str or sympy.Symbol
            The parameter to slice.
        interval_start : float, optional
            Start of the interval.
        interval_end : float, optional
            End of the interval.
        """
        if not isinstance(param, str):
            param = str(param)
        left_open = self.intervals[param].left_open
        right_open = self.intervals[param].right_open
        if interval_start is None:
            interval_start = self.intervals[param].inf
        else:
            left_open = False
        if interval_end is None:
            interval_end = self.intervals[param].sup
        else:
            right_open = False
        self.intervals[param] = sympy.Interval(
            interval_start, interval_end, left_open, right_open
        )

    def _get_cdf_expr(self):
        """
        Get the symbolic CDF expression with parameters substituted.

        Returns
        -------
        sympy expression
            The CDF expression with free symbols substituted.
        """
        return CDFWrapper(self._cdf_expr)

    def cdf(self, *args, **kwargs):
        """
        Compute the CDF at one or multiple points.

        This method handles both single-point and multi-point CDF evaluation
        in an efficient vectorized manner. It also supports variable substitution
        via keyword arguments.

        Parameters
        ----------
        *args : array-like or float
            Either:
            - Multiple separate coordinates (x, y, ...) of a single point
            - A single array-like object with coordinates of a single point
            - A 2D array where each row represents a separate point
        **kwargs : dict
            Keyword arguments for variable substitution. Supports various naming conventions:
            - Standard: u1=0.3, u2=0.7, etc.
            - Bivariate alternative: u=0.3, v=0.7
            - Original expression variables: x=0.3, y=0.7
            Partial substitution is allowed.

        Returns
        -------
        float or numpy.ndarray or CDFWrapper
            If a single point is provided, returns a float.
            If multiple points are provided, returns an array of shape (n_points,).
            If only variable substitutions are provided, returns a partially evaluated CDFWrapper.
            If no arguments are provided, returns the full CDFWrapper.

        Examples
        --------
        # Single point as separate arguments
        value = copula.cdf(0.3, 0.7)

        # Single point as array
        value = copula.cdf([0.3, 0.7])

        # Multiple points as 2D array
        values = copula.cdf(np.array([[0.1, 0.2], [0.3, 0.4]]))

        # Variable substitution (standard)
        value = copula.cdf(u1=0.3, u2=0.7)

        # Variable substitution (bivariate alternative)
        value = copula.cdf(u=0.3, v=0.7)

        # Variable substitution (original expression)
        value = copula.cdf(x=0.3, y=0.7)

        # Partial variable substitution
        partial_cdf = copula.cdf(u1=0.3)  # Returns a function of u2
        value = partial_cdf(0.7)  # Evaluate at u2=0.7
        """
        cdf_expr = self._get_cdf_expr()
        # Apply substitutions
        cdf_expr = cdf_expr(**kwargs)

        # If args are also provided, evaluate the resulting expression
        if args:
            # Get the remaining symbols in the expression
            remaining_vars = [str(sym) for sym in self.u_symbols if cdf_expr.has(sym)]
            remaining_dim = len(remaining_vars)

            # Convert args to a point array for remaining variables
            if len(args) == 1:
                arg = args[0]
                if hasattr(arg, "ndim") and hasattr(arg, "shape"):
                    arr = np.asarray(arg, dtype=float)
                    if arr.ndim == 1:
                        if len(arr) != remaining_dim:
                            raise ValueError(
                                f"Expected {remaining_dim} remaining coordinates, got {len(arr)}"
                            )
                        point = arr
                    else:
                        raise ValueError(
                            "Cannot mix variable substitution with multi-point evaluation"
                        )
                elif hasattr(arg, "__len__"):
                    if len(arg) != remaining_dim:
                        raise ValueError(
                            f"Expected {remaining_dim} remaining coordinates, got {len(arg)}"
                        )
                    point = np.array(arg, dtype=float)
                else:
                    if remaining_dim != 1:
                        raise ValueError(
                            f"Expected {remaining_dim} remaining coordinates, got 1"
                        )
                    point = np.array([arg], dtype=float)
            else:
                if len(args) != remaining_dim:
                    raise ValueError(
                        f"Expected {remaining_dim} remaining coordinates, got {len(args)}"
                    )
                point = np.array(args, dtype=float)

            # Substitute the remaining values
            cdf_expr = cdf_expr(**{var: val for var, val in zip(remaining_vars, point)})
            # Evaluate the fully substituted expression
            return cdf_expr

        # Return a partially evaluated CDF
        return cdf_expr

    def _cdf_single_point(self, u):
        """
        Helper method to compute CDF for a single point.

        Parameters
        ----------
        u : numpy.ndarray
            1D array of length dim representing a single point.

        Returns
        -------
        float
            CDF value at the point.
        """
        # Get the CDF wrapper and evaluate
        cdf_wrapper = self._get_cdf_expr()
        return cdf_wrapper(*u)

    def cond_distr(self, i, *args, **kwargs):
        """
        Compute the conditional distribution for one or multiple points.

        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        *args : array-like or float
            Either:
            - Multiple separate coordinates (x, y, ...) of a single point
            - A single array-like object with coordinates of a single point
            - A 2D array where each row represents a separate point
        **kwargs : dict
            Keyword arguments for variable substitution. Supports various naming conventions:
            - Standard: u1=0.3, u2=0.7, etc.
            - Bivariate alternative: u=0.3, v=0.7
            - Original expression variables: x=0.3, y=0.7
            Partial substitution is allowed.

        Returns
        -------
        float or numpy.ndarray or SymPyFuncWrapper
            If a single point is provided, returns a float.
            If multiple points are provided, returns an array of shape (n_points,).
            If only variable substitutions are provided, returns a partially evaluated SymPyFuncWrapper.
            If no arguments are provided, returns the full SymPyFuncWrapper.

        Examples
        --------
        # Single point as separate arguments
        value = copula.cond_distr(1, 0.3, 0.7)

        # Single point as array
        value = copula.cond_distr(1, [0.3, 0.7])

        # Multiple points as 2D array
        values = copula.cond_distr(1, np.array([[0.1, 0.2], [0.3, 0.4]]))

        # Variable substitution (standard)
        value = copula.cond_distr(1, u1=0.3, u2=0.7)

        # Variable substitution (bivariate alternative)
        value = copula.cond_distr(1, u=0.3, v=0.7)

        # Variable substitution (original expression)
        value = copula.cond_distr(1, x=0.3, y=0.7)

        # Partial variable substitution
        partial_cond = copula.cond_distr(1, u1=0.3)  # Returns a function of u2
        value = partial_cond(0.7)  # Evaluate at u2=0.7
        """
        if i < 1 or i > self.dim:
            raise ValueError(f"Dimension {i} out of range 1..{self.dim}")

        # Get the conditional distribution expression
        cdf = self.cdf()
        cond_expr = cdf.diff(self.u_symbols[i - 1])
        cond_expr = CDiWrapper(cond_expr, i)(**kwargs)
        # If args are also provided, evaluate the resulting expression
        if args:
            remaining_vars = [str(sym) for sym in self.u_symbols if cond_expr.has(sym)]
            remaining_dim = len(remaining_vars)

            # Convert args to a point array for remaining variables
            if len(args) == 1:
                arg = args[0]
                if hasattr(arg, "ndim") and hasattr(arg, "shape"):
                    arr = np.asarray(arg, dtype=float)
                    if arr.ndim == 1:
                        if len(arr) != remaining_dim:
                            raise ValueError(
                                f"Expected {remaining_dim} remaining coordinates, got {len(arr)}"
                            )
                        point = arr
                    else:
                        raise ValueError(
                            "Cannot mix variable substitution with multi-point evaluation"
                        )
                elif hasattr(arg, "__len__"):
                    if len(arg) != remaining_dim:
                        raise ValueError(
                            f"Expected {remaining_dim} remaining coordinates, got {len(arg)}"
                        )
                    point = np.array(arg, dtype=float)
                else:
                    if remaining_dim != 1:
                        raise ValueError(
                            f"Expected {remaining_dim} remaining coordinates, got 1"
                        )
                    point = np.array([arg], dtype=float)
            else:
                if len(args) != remaining_dim:
                    raise ValueError(
                        f"Expected {remaining_dim} remaining coordinates, got {len(args)}"
                    )
                point = np.array(args, dtype=float)

            # Create a mapping from remaining variables to values
            sub_dict = {var: point[i] for i, var in enumerate(remaining_vars)}

            # Substitute the remaining values
            wrapper = CDiWrapper(cond_expr, i)
            return wrapper(**sub_dict)

        # Return a partially evaluated conditional distribution
        return CDiWrapper(cond_expr, i)

    def _cond_distr_single(self, i, u):
        """
        Helper method for conditional distribution of a single point.

        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        u : numpy.ndarray
            Single point as a 1D array of length dim.

        Returns
        -------
        float
            Conditional distribution value.
        """
        # Get the conditional distribution function
        cdf = self.cdf()
        derivative = sympy.diff(cdf, self.u_symbols[i - 1])
        cond_distr_func = SymPyFuncWrapper(derivative)

        # Evaluate at the point
        return cond_distr_func(*u)

    def _cond_distr_vectorized(self, i, points):
        """
        Vectorized implementation of conditional distribution for multiple points.

        Parameters
        ----------
        i : int
            Dimension index (1-based) to condition on.
        points : numpy.ndarray
            Multiple points as a 2D array of shape (n_points, dim).

        Returns
        -------
        numpy.ndarray
            Array of shape (n_points,) with conditional distribution values.
        """
        n_points = points.shape[0]
        results = np.zeros(n_points)

        # Get the conditional distribution function
        cond_distr_func = SymPyFuncWrapper(
            sympy.diff(self._get_cdf_expr(), self.u_symbols[i - 1])
        )

        # Evaluate for each point
        for j, point in enumerate(points):
            results[j] = cond_distr_func(*point)

        return results

    def cond_distr_1(self, *args, **kwargs):
        """F_{U_{-1}|U_1}(u_{-1} | u_1).

        Parameters
        ----------
        *args : array-like or float
            Either:
            - Multiple separate coordinates (x, y, ...) of a single point
            - A single array-like object with coordinates of a single point
            - A 2D array where each row represents a separate point
        **kwargs : dict
            Keyword arguments for variable substitution, like u1=0.3, u2=0.7.
            Partial substitution is allowed.
        """
        return self.cond_distr(1, *args, **kwargs)

    def cond_distr_2(self, *args, **kwargs):
        """F_{U_{-2}|U_2}(u_{-2} | u_2).

        Parameters
        ----------
        *args : array-like or float
            Either:
            - Multiple separate coordinates (x, y, ...) of a single point
            - A single array-like object with coordinates of a single point
            - A 2D array where each row represents a separate point
        **kwargs : dict
            Keyword arguments for variable substitution, like u1=0.3, u2=0.7.
            Partial substitution is allowed.
        """
        return self.cond_distr(2, *args, **kwargs)

    def pdf(self, *args, **kwargs):
        """
        Evaluate the PDF at one or multiple points.

        Parameters
        ----------
        *args : array-like or float
            Either:
            - Multiple separate coordinates (x, y, ...) of a single point
            - A single array-like object with coordinates of a single point
            - A 2D array where each row represents a separate point
        **kwargs : dict
            Keyword arguments for variable substitution. Supports various naming conventions:
            - Standard: u1=0.3, u2=0.7, etc.
            - Bivariate alternative: u=0.3, v=0.7
            - Original expression variables: x=0.3, y=0.7
            Partial substitution is allowed.

        Returns
        -------
        float or numpy.ndarray or SymPyFuncWrapper
            If a single point is provided, returns a float.
            If multiple points are provided, returns an array of shape (n_points,).
            If only variable substitutions are provided, returns a partially evaluated SymPyFuncWrapper.
            If no arguments are provided, returns the full SymPyFuncWrapper.

        Examples
        --------
        # Single point as separate arguments
        value = copula.pdf(0.3, 0.7)

        # Single point as array
        value = copula.pdf([0.3, 0.7])

        # Multiple points as 2D array
        values = copula.pdf(np.array([[0.1, 0.2], [0.3, 0.4]]))

        # Variable substitution (standard)
        value = copula.pdf(u1=0.3, u2=0.7)

        # Variable substitution (bivariate alternative)
        value = copula.pdf(u=0.3, v=0.7)

        # Variable substitution (original expression)
        value = copula.pdf(x=0.3, y=0.7)

        # Partial variable substitution
        partial_pdf = copula.pdf(u1=0.3)  # Returns a function of u2
        value = partial_pdf(0.7)  # Evaluate at u2=0.7
        """
        # Get the PDF expression
        pdf_expr = self._get_cdf_expr()
        for u_symbol in self.u_symbols:
            pdf_expr = pdf_expr.diff(u_symbol)

        pdf_expr = pdf_expr(**kwargs)

        # If args are also provided, evaluate the resulting expression
        if args:
            remaining_vars = [str(sym) for sym in self.u_symbols if pdf_expr.has(sym)]
            remaining_dim = len(remaining_vars)

            # Convert args to a point array for remaining variables
            if len(args) == 1:
                arg = args[0]
                if hasattr(arg, "ndim") and hasattr(arg, "shape"):
                    arr = np.asarray(arg, dtype=float)
                    if arr.ndim == 1:
                        if len(arr) != remaining_dim:
                            raise ValueError(
                                f"Expected {remaining_dim} remaining coordinates, got {len(arr)}"
                            )
                        point = arr
                    else:
                        raise ValueError(
                            "Cannot mix variable substitution with multi-point evaluation"
                        )
                elif hasattr(arg, "__len__"):
                    if len(arg) != remaining_dim:
                        raise ValueError(
                            f"Expected {remaining_dim} remaining coordinates, got {len(arg)}"
                        )
                    point = np.array(arg, dtype=float)
                else:
                    if remaining_dim != 1:
                        raise ValueError(
                            f"Expected {remaining_dim} remaining coordinates, got 1"
                        )
                    point = np.array([arg], dtype=float)
            else:
                if len(args) != remaining_dim:
                    raise ValueError(
                        f"Expected {remaining_dim} remaining coordinates, got {len(args)}"
                    )
                point = np.array(args, dtype=float)

            # Create a mapping from remaining variables to values
            sub_dict = {var: point[i] for i, var in enumerate(remaining_vars)}

            # Substitute the remaining values
            return pdf_expr(**sub_dict)

        # Return a partially evaluated PDF
        return SymPyFuncWrapper(pdf_expr)

    def _pdf_single_point(self, u):
        """
        Helper method to compute PDF for a single point.

        Parameters
        ----------
        u : numpy.ndarray
            1D array of length dim representing a single point.

        Returns
        -------
        float
            PDF value at the point.
        """
        # Compute the PDF
        term = self._get_cdf_expr()
        for u_symbol in self.u_symbols:
            term = sympy.diff(term, u_symbol)
        pdf_func = SymPyFuncWrapper(term)

        # Evaluate at the point
        return pdf_func(*u)

    def _pdf_vectorized(self, points):
        """
        Vectorized implementation of PDF for multiple points.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape (n_points, dim) where each row is a point.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_points,) with PDF values.
        """
        n_points = points.shape[0]
        results = np.zeros(n_points)

        # Compute the PDF function
        term = self._get_cdf_expr()
        for u_symbol in self.u_symbols:
            term = sympy.diff(term, u_symbol)
        pdf_func = SymPyFuncWrapper(term)

        # Evaluate for each point
        for i, point in enumerate(points):
            results[i] = pdf_func(*point)

        return results

    # ------------------------------------------------------------------
    # Copula transforms
    # ------------------------------------------------------------------
    def survival_copula(self):
        """
        Return the survival (upper–tail) copula 𝑪̂ corresponding to *self*.

        In d dimensions the survival copula is given by the inclusion–
        exclusion formula  
            𝑪̂(u) = Σ_{J⊆{1,…,d}} (−1)^{|J|} C(u_J^c),  
        where *u_J^c* replaces u_j by 1 for j∈J.

        Returns
        -------
        CoreCopula
            A new copula object whose CDF expression is the survival copula
            of the current one.
        """
        from itertools import combinations

        if self._cdf_expr is None:
            raise ValueError("CDF expression is not set for this copula.")

        expr = 0
        # Inclusion–exclusion over all coordinate subsets
        for k in range(self.dim + 1):
            for J in combinations(range(self.dim), k):
                subs = {self.u_symbols[j]: 1 for j in J}
                expr += (-1) ** k * self._cdf_expr.subs(subs)

        new_copula = copy.copy(self)
        new_copula._cdf_expr = sympy.simplify(expr)
        return new_copula

    def vertical_reflection(self, margin: int = 2):
        """
        Return the vertical reflection C^{∨} of *self* with respect to one
        margin.

        By default (margin=2) and for the bivariate case this is  
            C^{∨}(u,v) = u − C(u, 1−v).

        For arbitrary `margin = j` (1 ≤ j ≤ dim) the definition is  
            C^{∨}(u) = u_j − C(u_1,…,u_{j−1}, 1−u_j, u_{j+1},…,u_d).

        Parameters
        ----------
        margin : int, optional (default=2)
            The coordinate index (1-based) that is reflected.

        Returns
        -------
        CoreCopula
            A new copula object whose CDF expression is the vertical
            reflection of the current one.
        """
        if not (1 <= margin <= self.dim):
            raise ValueError(f"margin must be in 1..{self.dim}")

        if self._cdf_expr is None:
            raise ValueError("CDF expression is not set for this copula.")

        uj = self.u_symbols[margin - 1]
        reflected_expr = sympy.simplify(uj - self._cdf_expr.subs({uj: 1 - uj}))

        new_copula = copy.copy(self)
        new_copula._cdf_expr = reflected_expr
        return new_copula
