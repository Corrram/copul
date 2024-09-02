from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper

import sympy as sp


def test_persistence_of_orig_func():
    x = sp.symbols("x")
    func = x**2
    wrapped_func = SymPyFunctionWrapper(func)
    assert wrapped_func(2) == 4
    assert wrapped_func(1) == 1
