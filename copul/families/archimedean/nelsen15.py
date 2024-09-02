import numpy as np
import sympy

from copul.families.archimedean.archimedean_copula import ArchimedeanCopula
from copul.families.other.lower_frechet import LowerFrechet
from copul.wrapper.sympy_wrapper import SymPyFunctionWrapper


class GenestGhoudi(ArchimedeanCopula):
    ac = ArchimedeanCopula
    theta = sympy.symbols("theta", positive=True)
    theta_interval = sympy.Interval(1, np.inf, left_open=False, right_open=True)

    def __call__(self, *args, **kwargs):
        if args is not None and len(args) > 0:
            kwargs["theta"] = args[0]
        if "theta" in kwargs and kwargs["theta"] == 1:
            del kwargs["theta"]
            return LowerFrechet()(**kwargs)
        return super().__call__(**kwargs)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    @property
    def _generator(self):
        return (1 - self.t ** (1 / self.theta)) ** self.theta

    @property
    def inv_generator(self):
        ind = sympy.Piecewise((1, self.y <= 1), (0, True))
        gen = (1 - self.y ** (1 / self.theta)) ** self.theta * ind
        return SymPyFunctionWrapper(gen)

    @property
    def cdf(self):
        cdf = (
            sympy.Max(
                1
                - (
                    (1 - self.u ** (1 / self.theta)) ** self.theta
                    + (1 - self.v ** (1 / self.theta)) ** self.theta
                )
                ** (1 / self.theta),
                0,
            )
            ** self.theta
        )
        return SymPyFunctionWrapper(cdf)

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 2 - 2 ** (1 / self.theta)


Nelsen15 = GenestGhoudi
