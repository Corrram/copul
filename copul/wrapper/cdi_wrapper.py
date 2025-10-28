import sympy
from copul.wrapper.conditional_wrapper import ConditionalWrapper
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper


class CDiWrapper(ConditionalWrapper):
    """
    General wrapper for the conditional distribution when conditioning on the i-th variable.
    Handles multivariate case (u1, u2, ..., un).

    Boundary conditions:
    - If ui = 0, returns 0
    - If ui = 1, returns 1 (representing the unconditional distribution of the remaining variables)
    - If any other variable is 0 or 1, handles according to conditional distribution rules
    """

    def __init__(self, func, i):
        """
        Initialize with a sympy expression and the index for conditioning.

        Parameters
        ----------
        func : sympy.Expr
            The symbolic expression representing the conditional distribution.
        i : int
            The index of the variable being conditioned on (1-based indexing).
        """
        self._index = i  # Store the index separately for creating new instances
        super().__init__(func, condition_index=i)

    def _check_boundary_conditions(self, u_symbols, vars_dict, kwargs):
        """
        Check boundary conditions for conditioning on the i-th variable.

        Parameters
        ----------
        u_symbols : dict
            Dictionary of u-symbols in the expression.
        vars_dict : dict
            Dictionary of variable substitutions.
        kwargs : dict
            Keyword arguments for substitution.

        Returns
        -------
        SymPyFuncWrapper or None
            A wrapper with the boundary value, or None if no boundary condition applies.
        """
        i = self.condition_index

        # Check all other variables first (priority for other variables being 0 or 1)
        # get max_dim from u_symbols
        dimensions = [
            int(k[1:]) for k in u_symbols.keys() if k.startswith("u") and len(k) > 1
        ]
        max_dim = max(dimensions, default=0)
        for j in range(1, max_dim):  # Assume reasonable max dimension of 10
            if j == i:
                continue  # Skip the conditioning variable

            target_sym = f"u{j}"
            if target_sym in u_symbols or target_sym in kwargs:
                uj_sym = u_symbols.get(target_sym, sympy.symbols(target_sym))
                uj_val = None

                if target_sym in kwargs:
                    uj_val = kwargs[target_sym]
                elif uj_sym in vars_dict:
                    uj_val = vars_dict[uj_sym]

                if uj_val == 0:
                    return SymPyFuncWrapper(sympy.S.Zero)
                elif uj_val == 1:
                    # When another variable is 1, the conditional depends only on remaining variables
                    # For simplicity, we'll return the conditioning variable itself or 1
                    target_cond_sym = f"u{i}"
                    if target_cond_sym in u_symbols:
                        ui_sym = u_symbols[target_cond_sym]
                        if ui_sym in vars_dict:
                            return SymPyFuncWrapper(vars_dict[ui_sym])
                        elif target_cond_sym in kwargs:
                            return SymPyFuncWrapper(kwargs[target_cond_sym])
                    return SymPyFuncWrapper(sympy.symbols(f"u{i}"))

        # Now check the conditioning variable
        target_sym = f"u{i}"
        if target_sym in u_symbols or target_sym in kwargs:
            ui_sym = u_symbols.get(target_sym, sympy.symbols(target_sym))
            ui_val = None

            if target_sym in kwargs:
                ui_val = kwargs[target_sym]
            elif ui_sym in vars_dict:
                ui_val = vars_dict[ui_sym]

            if ui_val == 0:
                return SymPyFuncWrapper(sympy.S.Zero)
            elif ui_val == 1:
                return SymPyFuncWrapper(sympy.S.One)

        # Handle the bivariate special case
        if i == 2 and "u" in u_symbols and "v" in u_symbols:
            # Check u (first variable)
            u_sym = u_symbols["u"]
            u_val = None

            if "u" in kwargs:
                u_val = kwargs["u"]
            elif u_sym in vars_dict:
                u_val = vars_dict[u_sym]

            if u_val == 0:
                return SymPyFuncWrapper(sympy.S.Zero)
            elif u_val == 1:
                return SymPyFuncWrapper(sympy.S.One)

        elif i == 1 and "u" in u_symbols and "v" in u_symbols:
            # Check v (second variable)
            v_sym = u_symbols["v"]
            v_val = None

            if "v" in kwargs:
                v_val = kwargs["v"]
            elif v_sym in vars_dict:
                v_val = vars_dict[v_sym]

            if v_val == 0:
                return SymPyFuncWrapper(sympy.S.Zero)
            elif v_val == 1:
                return SymPyFuncWrapper(sympy.S.One)

        return None

    def __call__(self, *args, **kwargs):
        """
        Evaluate the conditional distribution with the given arguments.

        Parameters
        ----------
        *args, **kwargs
            Arguments to substitute into the expression.

        Returns
        -------
        CDiWrapper or SymPyFuncWrapper
            A new wrapper with the substituted expression.
        """
        # Get all u-symbols in the expression (names present in the sympy expr)
        u_symbols = self._get_u_symbols()

        # --- NEW: eager positional mapping for the common bivariate signature (u, v)
        # If user passed 2 positional args and no kwargs, interpret as (u, v)
        # regardless of how many free symbols are in the underlying expression.
        if args and not kwargs:
            if len(args) == 2:
                # Standardize bivariate call signature
                kwargs = {"u": args[0], "v": args[1]}
                args = ()
            elif len(args) == 1:
                # If expression only uses one of {'u','v'}, map the single arg to that one.
                # Prefer 'u' if present, else 'v'. Fall back to 'u1' if neither present.
                if "u" in u_symbols:
                    kwargs = {"u": args[0]}
                elif "v" in u_symbols:
                    kwargs = {"v": args[0]}
                else:
                    # Multivariate fallback: map to the conditioning variable ui
                    i = self.condition_index
                    kwargs = {f"u{i}": args[0]}
                args = ()

        # Process arguments to create variable substitutions
        vars_, kwargs = self._prepare_call(args, kwargs)

        # Check for boundary conditions
        boundary_result = self._check_boundary_conditions(u_symbols, vars_, kwargs)
        if boundary_result is not None:
            return boundary_result

        # Apply substitutions
        func = self._func.subs(vars_)

        # Return a new wrapper of the same type with the same index
        return CDiWrapper(func, self._index)
