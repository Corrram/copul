import copy
import numpy as np
import sympy

from copul.wrapper.cdf_wrapper import CDFWrapper
from copul.wrapper.cdi_wrapper import CDiWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CoreCopula:
    r"""
    Unified core for copula classes.

    This class consolidates functionality that was previously split between
    ``CoreCopula`` and ``Copula``. It manages symbolic CDF/PDF expressions,
    parameters and their admissible intervals, and provides flexible evaluation
    utilities (single point, vectorized, and partially substituted forms).
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
        r"""
        Initialize a copula.

        Parameters
        ----------
        dimension : int
            Dimension :math:`d` of the copula.
        *args : tuple
            Positional arguments mapped onto remaining symbolic parameters in order.
        **kwargs : dict
            Explicit parameter assignments (e.g. ``rho=0.5``) or other attributes.

        Notes
        -----
        Any provided parameters are removed from ``self.params`` and their intervals
        from ``self.intervals``; the remaining ones stay symbolic.
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
        r"""
        Create a **new** copula instance with updated parameters.

        Parameters
        ----------
        *args : tuple
            Positional arguments mapped onto the instance’s **remaining** parameters.
        **kwargs : dict
            Explicit parameter assignments.

        Returns
        -------
        CoreCopula
            A (shallow) copy of ``self`` with the provided parameter updates applied.
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
        r"""
        Populate parameters from ``args``/``kwargs`` on the current instance.

        Parameters
        ----------
        args : tuple
            Positional arguments mapped onto the remaining parameters in order.
        kwargs : dict
            Explicit parameter assignments (overrides values from ``args``).
        """

        if args and len(args) <= len(self.params):
            for i in range(len(args)):
                kwargs[str(self.params[i])] = args[i]
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    @property
    def parameters(self):
        r"""
        Parameter intervals of the copula.

        Returns
        -------
        dict
            Mapping ``name -> sympy.Interval`` for the **remaining** (not yet fixed)
            parameters.
        """

        return self.intervals

    @property
    def is_absolutely_continuous(self) -> bool:
        r"""
        Whether the copula is absolutely continuous.

        Returns
        -------
        bool
            ``True`` if the copula has a density a.e. on :math:`[0,1]^d`,
            otherwise ``False``.

        Notes
        -----
        Subclasses must override this property.
        """

        # Implementations should override this method
        raise NotImplementedError("This method must be implemented in a subclass")

    @property
    def is_symmetric(self) -> bool:
        r"""
        Whether the copula is exchangeable (symmetric under coordinate permutations).

        Returns
        -------
        bool
            ``True`` if :math:`C(u_{\pi(1)},\ldots,u_{\pi(d)}) = C(u_1,\ldots,u_d)`
            for all permutations :math:`\pi`, otherwise ``False``.

        Notes
        -----
        Subclasses must override this property.
        """

        # Implementations should override this method
        raise NotImplementedError("This method must be implemented in a subclass")

    def _are_class_vars(self, kwargs):
        r"""
        Validate that all keys in ``kwargs`` are attributes of the instance.

        Parameters
        ----------
        kwargs : dict
            Candidate attribute assignments.

        Raises
        ------
        AssertionError
            If a key is not found among the instance attributes.
        """

        class_vars = set(dir(self))
        assert set(kwargs).issubset(class_vars), (
            f"keys: {set(kwargs)}, free symbols: {class_vars}"
        )

    def slice_interval(self, param, interval_start=None, interval_end=None):
        r"""
        Restrict the admissible interval of a parameter.

        Parameters
        ----------
        param : str or sympy.Symbol
            Parameter to restrict.
        interval_start : float, optional
            New lower bound (inclusive).
        interval_end : float, optional
            New upper bound (inclusive).

        Notes
        -----
        If a bound is not provided, the corresponding bound from the current
        interval is kept. Open bounds are closed when explicitly set.
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
        r"""
        Return the symbolic CDF wrapper with current parameter substitutions.

        Returns
        -------
        CDFWrapper
            Wrapper around the (possibly partially substituted) CDF expression.
        """

        return CDFWrapper(self._cdf_expr)

    def cdf(self, *args, **kwargs):
        r"""
        Evaluate (or partially evaluate) the CDF.

        This method supports single-point evaluation, partial variable substitution,
        and returning a callable wrapper if not all variables are specified.

        Parameters
        ----------
        *args : array-like or float
            Either
            - coordinates of a **single** point (as separate scalars or a 1D array), or
            - a **single** 1D array of remaining coordinates after substitution.
            (Multi-point arrays are not supported here.)
        **kwargs : dict
            Variable substitutions. Supported naming schemes (mixable):
            - Standard: ``u1=0.3, u2=0.7, ...``
            - Bivariate alias: ``u=0.3, v=0.7``
            - Original expression variables (e.g. ``x=..., y=...``)

        Returns
        -------
        float or CDFWrapper
            - A scalar if all variables are provided,
            - otherwise a partially evaluated ``CDFWrapper`` callable.

        Examples
        --------
        Single point as separate args::

            value = copula.cdf(0.3, 0.7)

        Single point as array::

            value = copula.cdf([0.3, 0.7])

        Variable substitution (bivariate alias)::

            value = copula.cdf(u=0.3, v=0.7)

        Partial substitution (returns a function of the remaining variable)::

            f = copula.cdf(u1=0.3)
            value = f(0.7)
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
        r"""
        Helper: CDF at a **single** point.

        Parameters
        ----------
        u : numpy.ndarray
            1D array of length ``dim``.

        Returns
        -------
        float
            :math:`C(u)`.
        """

        # Get the CDF wrapper and evaluate
        cdf_wrapper = self._get_cdf_expr()
        return cdf_wrapper(*u)

    def cond_distr(self, i, *args, **kwargs):
        r"""
        Evaluate (or partially evaluate) the conditional distribution

        .. math::
           F_{U_{-i}\mid U_i}(u_{-i}\mid u_i)
           \;=\; \frac{\partial}{\partial u_i} \, C(u_1,\ldots,u_d).

        Parameters
        ----------
        i : int
            Index (1-based) of the conditioning coordinate.
        *args : array-like or float
            Coordinates of a **single** point for the remaining variables
            (same rules as :meth:`cdf`).
        **kwargs : dict
            Variable substitutions (same schemes as :meth:`cdf`).

        Returns
        -------
        float or SymPyFuncWrapper
            - Scalar if all variables are provided,
            - otherwise a partially evaluated ``SymPyFuncWrapper`` callable.
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
        r"""
        Helper: conditional distribution at a **single** point.

        Parameters
        ----------
        i : int
            Conditioning index (1-based).
        u : numpy.ndarray
            1D array of length ``dim``.

        Returns
        -------
        float
            Value of :math:`F_{U_{-i}\mid U_i}(u_{-i}\mid u_i)`.
        """

        # Get the conditional distribution function
        cdf = self.cdf()
        derivative = sympy.diff(cdf, self.u_symbols[i - 1])
        cond_distr_func = SymPyFuncWrapper(derivative)

        # Evaluate at the point
        return cond_distr_func(*u)

    def _cond_distr_vectorized(self, i, points):
        r"""
        Helper: conditional distribution for **multiple** points.

        Parameters
        ----------
        i : int
            Conditioning index (1-based).
        points : numpy.ndarray
            Array of shape ``(n_points, dim)``.

        Returns
        -------
        numpy.ndarray
            Values of :math:`F_{U_{-i}\mid U_i}` for each row.
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
        r"""
        :math:`F_{U_{-1}\mid U_1}(u_{-1}\mid u_1)`.

        Parameters
        ----------
        *args, **kwargs
            See :meth:`cond_distr`.
        """

        return self.cond_distr(1, *args, **kwargs)

    def cond_distr_2(self, *args, **kwargs):
        r"""
        :math:`F_{U_{-2}\mid U_2}(u_{-2}\mid u_2)`.

        Parameters
        ----------
        *args, **kwargs
            See :meth:`cond_distr`.
        """

        return self.cond_distr(2, *args, **kwargs)

    def pdf(self, *args, **kwargs):
        r"""
        Evaluate (or partially evaluate) the PDF.

        This differentiates the CDF once w.r.t. **each** coordinate and then applies
        the same substitution/evaluation logic as :meth:`cdf`.

        Parameters
        ----------
        *args : array-like or float
            Coordinates of a **single** point (same rules as :meth:`cdf`).
        **kwargs : dict
            Variable substitutions (same schemes as :meth:`cdf`).

        Returns
        -------
        float or numpy.ndarray or SymPyFuncWrapper
            - Scalar if all variables are provided,
            - otherwise a partially evaluated ``SymPyFuncWrapper`` callable.

        Examples
        --------
        Single point as separate args::

            value = copula.pdf(0.3, 0.7)

        Variable substitution (bivariate alias)::

            value = copula.pdf(u=0.3, v=0.7)

        Partial substitution::

            f = copula.pdf(u1=0.3)
            value = f(0.7)
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
        r"""
        Helper: PDF at a **single** point.

        Parameters
        ----------
        u : numpy.ndarray
            1D array of length ``dim``.

        Returns
        -------
        float
            :math:`c(u)` (the copula density at ``u``).
        """

        # Compute the PDF
        term = self._get_cdf_expr()
        for u_symbol in self.u_symbols:
            term = sympy.diff(term, u_symbol)
        pdf_func = SymPyFuncWrapper(term)

        # Evaluate at the point
        return pdf_func(*u)

    def _pdf_vectorized(self, points):
        r"""
        Vectorized PDF for **multiple** points.

        Parameters
        ----------
        points : numpy.ndarray
            Array of shape ``(n_points, dim)`` where each row is a point.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_points,)`` with values of :math:`c(u)`.
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
        r"""
        Return the survival (upper-tail) copula :math:`\widehat C` corresponding to *self*.

        In :math:`d` dimensions, the survival copula is given by the inclusion–exclusion formula

        .. math::

           \widehat C(u)
           \;=\;
           \sum_{J\subseteq\{1,\dots,d\}} (-1)^{|J|}
           \, C\!\big(u^{(J)}\big),

        where :math:`u^{(J)}` denotes the vector obtained from :math:`u` by replacing
        :math:`u_j` with :math:`1` for all :math:`j\in J`.

        Returns
        -------
        CoreCopula
            A new copula object whose CDF expression is the survival copula of the current one.
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
        r"""
        Vertical reflection :math:`C^{\vee}` of *self* with respect to one margin.

        By default (``margin=2``) and in the bivariate case,
        \[
        C^{\vee}(u,v) \;=\; u \;-\; C\bigl(u,\,1-v\bigr).
        \]
        For general ``margin=j`` (``1 \le j \le \mathrm{dim}``),
        \[
        C^{\vee}(u) \;=\; u_j \;-\;
        C\bigl(u_1,\dots,u_{j-1},\,1-u_j,\,u_{j+1},\dots,u_d\bigr).
        \]

        Parameters
        ----------
        margin : int, optional
            1-based index of the reflected coordinate (default ``2``).

        Returns
        -------
        CoreCopula
            A new copula object whose CDF expression is the vertical reflection of the current one.
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
