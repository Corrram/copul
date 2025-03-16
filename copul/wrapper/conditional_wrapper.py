import sympy
import re
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

class ConditionalWrapper(SymPyFuncWrapper):
    """
    Base class for conditional distribution wrappers.
    Handles multivariate cases with symbols u1, u2, ..., un.
    """
    def __init__(self, func, condition_index=None):
        """
        Initialize with a sympy expression and the index for conditioning.
        
        Parameters
        ----------
        func : sympy.Expr
            The symbolic expression representing the conditional distribution.
        condition_index : int or None
            The index of the variable being conditioned on (1-based indexing).
            If None, will be determined based on the wrapper class.
        """
        super().__init__(func)
        self.condition_index = condition_index
        
    def _get_u_symbols(self):
        """
        Extract u1, u2, ..., un symbols from the function.
        
        Returns
        -------
        dict
            Dictionary mapping symbol names to symbol objects.
        """
        u_pattern = re.compile(r'u(\d+)')
        u_symbols = {}
        
        for sym in self._func.free_symbols:
            sym_str = str(sym)
            match = u_pattern.match(sym_str)
            if match:
                u_symbols[sym_str] = sym
                
        # Also check for simple u and v symbols
        for sym in self._func.free_symbols:
            sym_str = str(sym)
            if sym_str == "u" or sym_str == "v":
                u_symbols[sym_str] = sym
                
        return u_symbols
    
    def _check_boundary_conditions(self, u_symbols, vars_dict, kwargs):
        """
        Check boundary conditions for a multivariate conditional distribution.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def __call__(self, *args, **kwargs):
        """
        Evaluate the conditional distribution with the given arguments.
        
        Parameters
        ----------
        *args, **kwargs
            Arguments to substitute into the expression.
            
        Returns
        -------
        SymPyFuncWrapper
            A wrapper around the result of the substitution.
        """
        # Get all u-symbols in the expression
        u_symbols = self._get_u_symbols()
        
        # Process arguments to create variable substitutions
        vars_, kwargs = self._prepare_call(args, kwargs)
        
        # Check for boundary conditions
        boundary_result = self._check_boundary_conditions(u_symbols, vars_, kwargs)
        if boundary_result is not None:
            return boundary_result
            
        # Apply substitutions
        func = self._func.subs(vars_)
        
        # Return a new wrapper of the same type
        return self.__class__(func, self.condition_index)
