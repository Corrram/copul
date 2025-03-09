import numpy as np

from copul import basictools
from copul.checkerboard.check_pi import CheckPi
from copul.families.bivcopula import BivCopula


class BivCheckPi(CheckPi, BivCopula):
    params = []
    intervals = {}

    def __init__(self, matr, **kwargs):
        CheckPi.__init__(self, matr)
        BivCopula.__init__(self, **kwargs)
        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]

    def __str__(self):
        return f"BivCheckPi(m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def cond_distr_1(self, u=None, v=None):
        return self.cond_distr(1, (u, v))

    def cond_distr_2(self, u=None, v=None):
        return self.cond_distr(2, (u, v))

    def tau(self):
        result = basictools.monte_carlo_integral(
            lambda x, y: self.cdf(x, y) * self.pdf(x, y)
        )
        return 4 * result - 1

    def rho(self):
        result = basictools.monte_carlo_integral(lambda x, y: self.cdf(x, y))
        return 12 * result - 3

    def chatterjees_xi(self, n_samples=100_000, condition_on_y=False, *args, **kwargs):
        self._set_params(args, kwargs)
        i = 2 if condition_on_y else 1

        def f(x, y):
            return self.cond_distr(i, (x, y)) ** 2

        result = basictools.monte_carlo_integral(f, n_samples, vectorized_func=False)
        return 6 * result - 2


if __name__ == "__main__":
    matr = [[1, 5, 4], [5, 3, 2], [4, 2, 4]]
    copul = BivCheckPi(matr)
    copul.plot_cond_distr_1()
