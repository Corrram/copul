from typing import Union

import numpy as np

from copul.exceptions import PropertyUnavailableException
from copul.families.other.checkerboard_copula import CheckerboardCopula


class UpperCheckerboardCopula(CheckerboardCopula):

    def __str__(self):
        return f"CheckerboardCopula(m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False

    def cdf(self, u, v):
        x = int((u * self.matr.shape[0]) // 1)
        y = int((v * self.matr.shape[1]) // 1)
        overlap_x = u * self.m - x
        overlap_y = v * self.matr.shape[1] - y
        total_integral = self.matr[:x, :y].sum()
        if overlap_x > 0:
            total_integral += overlap_x * self.matr[x, :y].sum()
        if overlap_y > 0:
            total_integral += overlap_y * self.matr[:x, y].sum()
        if overlap_x > 0 and overlap_y > 0:
            total_integral += np.min([overlap_x, overlap_y]) * self.matr[x, y]
        return total_integral

    def cond_distr_1(self, u, v):
        if isinstance(u, np.ndarray):
            return np.array([self.cond_distr_1(u_, v) for u_ in u])
        if isinstance(v, np.ndarray):
            return np.array([self.cond_distr_1(u, v_) for v_ in v])
        x = int(u * self.matr.shape[0])
        y = int(v * self.matr.shape[1])
        integral = self.matr[x, :y].sum()
        overlap_y = v * self.matr.shape[1] - y
        adjusted_u = u + y / self.matr.shape[1]
        adjusted_v = v + x / self.matr.shape[0]
        if overlap_y > 0 and adjusted_u <= adjusted_v:
            integral += self.matr[x, y]
        result = integral * self.matr.shape[0]
        return result

    def cond_distr_2(self, u, v) -> Union[float, np.ndarray]:
        if isinstance(v, np.ndarray):
            return np.array([self.cond_distr_2(u, v_) for v_ in v])
        if isinstance(u, np.ndarray):
            return np.array([self.cond_distr_2(u_, v) for u_ in u])
        x = int(u * self.matr.shape[0])
        y = int(v * self.matr.shape[1])
        integral = self.matr[:x, y].sum()
        overlap_x = u * self.matr.shape[0] - x
        adjusted_v = v + x / self.matr.shape[0]
        adjusted_u = u + y / self.matr.shape[1]
        if overlap_x > 0 and adjusted_v <= adjusted_u:
            integral += self.matr[x, y]
        result = integral * self.matr.shape[1]
        return result

    @property
    def pdf(self):
        msg = "PDF does not exist for Upper Checkerboard Copula"
        raise PropertyUnavailableException(msg)

    def rvs(self, n=1):
        sel_ele, sel_idx = self._weighted_random_selection(self.matr, n)
        u = np.random.rand(n) / self.m
        v = u
        add_random = np.array([u, v]).T
        data_points = np.array([(idx[0] / self.m, idx[1] / self.n) for idx in sel_idx])
        data_points += add_random
        return data_points

    def lambda_L(self):
        return 1

    def lambda_U(self):
        return 1
