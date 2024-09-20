import numpy as np

from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

import sympy as sp


def test_persistence_of_orig_func():
    x = sp.symbols("x")
    func = x**2
    wrapped_func = SymPyFuncWrapper(func)
    assert wrapped_func(2) == 4
    assert wrapped_func(1) == 1


def test_evalf():
    x = sp.symbols("x")
    func = x**2
    wrapped_func = SymPyFuncWrapper(func)
    assert np.isclose(wrapped_func(2).evalf(), 4)
