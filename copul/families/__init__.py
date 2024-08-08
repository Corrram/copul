import sympy


def get_simplified_solution(sol):
    try:
        simplified_sol = sympy.simplify(sol)
    except TypeError:
        return sol
    if isinstance(simplified_sol, sympy.core.containers.Tuple):
        return simplified_sol[0]
    else:
        return simplified_sol


def round_expression(expr, n=2):
    expr = sympy.simplify(expr)
    for a in sympy.preorder_traversal(expr):
        if isinstance(a, sympy.Float):
            expr = expr.subs(a, round(a, n))
    return expr


def concrete_expand_log(expr, first_call=True):
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
