import numpy as np

from copul import basictools
from copul.families.copula import Copula


class MultivarCheckerboardCopula(Copula):
    params = []
    intervals = {}

    def __init__(self, matr):
        self.matr = matr / matr.sum()
        self.dim = matr.shape
        super().__init__(dimension=len(self.dim))

    def __str__(self):
        return f"CheckerboardCopula({self.dim})"

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

    def cdf(self, *args):
        if len(args) != len(self.dim):
            msg = "Number of arguments must be equal to the dimension of the copula"
            raise ValueError(msg)
        indices = []
        overlaps = []
        for i in range(len(args)):
            arg = args[i]
            if arg <= 0:
                return 0
            shape = self.dim[i]
            index = int((arg * shape) // 1)
            indices.append(index)
            overlap = arg * shape - index
            overlaps.append(overlap)
        total_integral = self.matr[tuple(slice(i) for i in indices)].sum()
        # total_integral += overlap_x * self.matr[x, :y].sum()
        # total_integral += overlap_y * self.matr[:x, y].sum()
        # total_integral += overlap_x * overlap_y * self.matr[x, y]
        return total_integral

    def cond_distr_1(self, u, v):
        if isinstance(u, np.ndarray):
            return np.array([self.cond_distr_1(u_, v) for u_ in u])
        x = int((u * self.matr.shape[0]) // 1)
        y = int((v * self.matr.shape[1]) // 1)
        total_integral = self.matr[x, :y].sum()
        overlap_y = v * self.matr.shape[1] - y
        total_integral += overlap_y * self.matr[x, y]
        return total_integral / self.matr[x, :].sum()

    def cond_distr_2(self, u, v):
        if isinstance(v, np.ndarray):
            return np.array([self.cond_distr_2(u, v_) for v_ in v])
        x = int((u * self.matr.shape[0]) // 1)
        y = int((v * self.matr.shape[1]) // 1)
        total_integral = self.matr[:x, y].sum()
        overlap_x = u * self.matr.shape[0] - x
        total_integral += overlap_x * self.matr[x, y]
        return total_integral / self.matr[:, y].sum()

    def pdf(self, *args):
        box = []
        for i in range(len(args)):
            arg = args[i]
            if arg < 0 or arg > 1:
                return 0
            box.append(int((arg * self.dim[i]) // 1))
        return self.matr[tuple(box)]

    def tau(self):
        result = basictools.monte_carlo_integral(
            lambda x, y: self.cdf(x, y) * self.pdf(x, y)
        )
        return 4 * result - 1

    def rho(self):
        result = basictools.monte_carlo_integral(lambda x, y: self.cdf(x, y))
        return 12 * result - 3

    def xi(self):
        result = basictools.monte_carlo_integral(
            lambda x, y: self.cond_distr_1(x, y) ** 2
        )
        return 6 * result - 2
