import sympy


def get_simplified_solution(sol):
    """
    Simplify a sympy expression and extract the primary result.

    This function attempts to simplify the given expression using
    sympy.simplify. If the result is a Tuple (e.g. multiple solutions),
    the first element is returned. If simplification fails due to a TypeError,
    the original expression is returned.

    Parameters
    ----------
    sol : sympy expression
        The sympy expression to simplify.

    Returns
    -------
    sympy expression
        The simplified expression, or the first element if the result is a Tuple.
        If a TypeError occurs during simplification, returns the original input.
    """
    try:
        simplified_sol = sympy.simplify(sol)
    except TypeError:
        return sol
    if isinstance(simplified_sol, sympy.core.containers.Tuple):
        return simplified_sol[0]
    else:
        return simplified_sol


def concrete_expand_log(expr, first_call=True):
    """
    Recursively expand logarithms in a sympy expression in a concrete manner.

    On the first call, the function forces an expansion of logarithms using
    sympy.expand_log. It then recursively traverses the expression. If an expression
    is a logarithm of a product (as a concrete Product), it is converted into a Sum
    representation. The recursion continues until all parts of the expression are
    processed.

    Parameters
    ----------
    expr : sympy expression
        The sympy expression in which to expand logarithms.
    first_call : bool, optional
        If True (default), forces an initial expansion of logarithms using expand_log.

    Returns
    -------
    sympy expression
        The expression with logarithms expanded into a sum form where applicable.
    """
    import sympy as sp

    if first_call:
        expr = sp.expand_log(expr, force=True)
    func = expr.func
    args = expr.args
    if args == ():
        return expr
    if func == sp.log and args[0].func == sp.concrete.products.Product:
        prod = args[0]
        term = prod.args[0]
        indices = prod.args[1:]
        return sp.Sum(sp.log(term), *indices)
    return func(*map(lambda x: concrete_expand_log(x, False), args))
