import numpy as np
import sympy

from copul.families.extreme_value.extreme_value_copula import ExtremeValueCopula
from copul.sympy_wrapper import SymPyFunctionWrapper


class Galambos(ExtremeValueCopula):
    @property
    def is_symmetric(self) -> bool:
        return True

    delta = sympy.symbols("delta", positive=True)
    params = [delta]
    intervals = {"delta": sympy.Interval(0, np.inf, left_open=True, right_open=True)}

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    @property
    def pickand(self):
        t = self.t
        delta = self.delta
        return 1 - (t ** (-delta) + (1 - t) ** (-delta)) ** (-1 / delta)

    @property
    def cdf(self):
        u = self.u
        delta = self.delta
        v = self.v
        cdf = (
            u
            * v
            * sympy.exp(
                (sympy.log(1 / u) ** (-delta) + sympy.log(1 / v) ** (-delta)) ** (-1 / delta)
            )
        )
        return SymPyFunctionWrapper(cdf)

    @property
    def pdf(self):
        u = self.u
        v = self.v
        delta = self.delta
        return (
            (u * v)
            ** (
                (
                    (
                        (
                            ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                            + (sympy.log(v) / sympy.log(u * v)) ** delta
                        )
                        / (
                            ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                            * (sympy.log(v) / sympy.log(u * v)) ** delta
                        )
                    )
                    ** (1 / delta)
                    - 1
                )
                / (
                    (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    / (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        * (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                )
                ** (1 / delta)
            )
            * (
                (
                    (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    / (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        * (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                )
                ** (1 / delta)
                * (delta + 1)
                * (
                    ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                    * (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    * (sympy.log(v) - sympy.log(u * v)) ** 2
                    + (sympy.log(v) / sympy.log(u * v)) ** delta
                    * (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    * sympy.log(v) ** 2
                    - (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        * (sympy.log(v) - sympy.log(u * v))
                        + (sympy.log(v) / sympy.log(u * v)) ** delta * sympy.log(v)
                    )
                    ** 2
                )
                + (
                    ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                    * (sympy.log(v) - sympy.log(u * v))
                    + (sympy.log(v) / sympy.log(u * v)) ** delta * sympy.log(v)
                    + (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    * (
                        (
                            (
                                ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                                + (sympy.log(v) / sympy.log(u * v)) ** delta
                            )
                            / (
                                ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                                * (sympy.log(v) / sympy.log(u * v)) ** delta
                            )
                        )
                        ** (1 / delta)
                        - 1
                    )
                    * (sympy.log(v) - sympy.log(u * v))
                )
                * (
                    ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                    * (sympy.log(v) - sympy.log(u * v))
                    + (sympy.log(v) / sympy.log(u * v)) ** delta * sympy.log(v)
                    + (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    * (
                        (
                            (
                                ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                                + (sympy.log(v) / sympy.log(u * v)) ** delta
                            )
                            / (
                                ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                                * (sympy.log(v) / sympy.log(u * v)) ** delta
                            )
                        )
                        ** (1 / delta)
                        - 1
                    )
                    * sympy.log(v)
                )
                * sympy.log(u * v)
            )
            / (
                u
                * v
                * (
                    (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        + (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                    / (
                        ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                        * (sympy.log(v) / sympy.log(u * v)) ** delta
                    )
                )
                ** (2 / delta)
                * (
                    ((-sympy.log(v) + sympy.log(u * v)) / sympy.log(u * v)) ** delta
                    + (sympy.log(v) / sympy.log(u * v)) ** delta
                )
                ** 2
                * (sympy.log(v) - sympy.log(u * v))
                * sympy.log(v)
                * sympy.log(u * v)
            )
        )


B7 = Galambos
