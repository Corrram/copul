from abc import ABC

import sympy


class AbstractMultivarCopula(ABC):
    params = None
    intervals = None
    log_cut_off = 4

    def __init__(self, n):
        self.x_symbols = sympy.symbols(f"x0:{n}")
