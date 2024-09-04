from typing import Union

import numpy as np
import sympy

from copul import basictools
from copul.families.bivcopula import BivCopula


class CheckerboardCopula(BivCopula):
    params = []
    intervals = {}

    def __init__(self, matr, mc_size=200_000, **kwargs):
        if isinstance(matr, (list, sympy.matrices.dense.Matrix)):
            matr = np.array(matr)
        self.matr = matr / matr.sum()
        self.m = matr.shape[0]
        self.n = matr.shape[1]
        self.n_samples = mc_size
        super().__init__(**kwargs)

    def __str__(self):
        return f"CheckerboardCopula(m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        return True

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
            total_integral += overlap_x * overlap_y * self.matr[x, y]
        return total_integral

    def cond_distr_1(self, u, v):
        if isinstance(u, np.ndarray):
            return np.array([self.cond_distr_1(u_, v) for u_ in u])
        if isinstance(v, np.ndarray):
            return np.array([self.cond_distr_1(u, v_) for v_ in v])
        nrow = self.matr.shape[0]
        x = min(int(u * nrow), nrow - 1)
        n_col = self.matr.shape[1]
        y = min(int(v * n_col), n_col - 1)
        total_integral = self.matr[x, :y].sum()
        overlap_y = v * n_col - y
        if overlap_y > 0:
            total_integral += overlap_y * self.matr[x, y]
        result = total_integral * nrow
        return result

    def cond_distr_2(self, u, v) -> Union[float, np.ndarray]:
        if isinstance(v, np.ndarray):
            return np.array([self.cond_distr_2(u, v_) for v_ in v])
        if isinstance(u, np.ndarray):
            return np.array([self.cond_distr_2(u_, v) for u_ in u])
        x = int(u * self.matr.shape[0])
        y = int(v * self.matr.shape[1])
        total_integral = self.matr[:x, y].sum()
        overlap_x = u * self.matr.shape[0] - x
        if overlap_x > 0:
            total_integral += overlap_x * self.matr[x, y]
        result = total_integral * self.matr.shape[1]
        return result

    def pdf(self, u, v):
        x = int(u * self.matr.shape[0])
        y = int(v * self.matr.shape[1])
        try:
            return self.matr[x, y]
        except IndexError as e:
            if u == 1:
                return self.matr[x - 1, y]
            if v == 1:
                return self.matr[x, y - 1]
            raise e

    def kendalls_tau(self, *args, **kwargs):
        self._set_params(args, kwargs)
        result = basictools.monte_carlo_integral(
            lambda x, y: self.cdf(x, y) * self.pdf(x, y), self.n_samples
        )
        return 4 * result - 1

    def spearmans_rho(self, *args, **kwargs):
        self._set_params(args, kwargs)
        result = basictools.monte_carlo_integral(
            lambda x, y: self.cdf(x, y), self.n_samples
        )
        return 12 * result - 3

    def chatterjees_xi(self, condition_on_y=False, *args, **kwargs):
        self._set_params(args, kwargs)
        method = self.cond_distr_2 if condition_on_y else self.cond_distr_1

        def f(x, y):
            return method(x, y) ** 2

        result = basictools.monte_carlo_integral(
            f, self.n_samples, vectorized_func=False
        )
        return 6 * result - 2

    def rvs(self, n=1):
        sel_ele, sel_idx = self._weighted_random_selection(self.matr, n)
        u = np.random.rand(n) / self.m
        v = np.random.rand(n) / self.n
        add_random = np.array([u, v]).T
        data_points = np.array([(idx[0] / self.m, idx[1] / self.n) for idx in sel_idx])
        data_points += add_random
        return data_points

    @staticmethod
    def _weighted_random_selection(matrix, num_samples):
        """
        Select elements from the matrix at random with likelihood proportional to their values.

        Parameters
        ----------
        matrix : numpy.ndarray
            2D array from which to select elements.
        num_samples : int
            Number of elements to select.

        Returns
        -------
        selected_elements : numpy.ndarray
            Array of selected elements.
        selected_indices : list of tuples
            List of indices of the selected elements in the original matrix.
        """
        # Flatten the matrix to a 1D array
        matrix = np.asarray(matrix, dtype=np.float64)
        flat_matrix = matrix.flatten()

        # Create the probability distribution proportional to the values
        probabilities = flat_matrix / np.sum(flat_matrix)

        # Select indices based on the probability distribution
        selected_indices_flat = np.random.choice(
            np.arange(flat_matrix.size), size=num_samples, p=probabilities
        )

        # Map the selected indices back to the original matrix
        selected_indices = [
            np.unravel_index(idx, matrix.shape) for idx in selected_indices_flat
        ]
        selected_elements = matrix[tuple(np.array(selected_indices).T)]

        return selected_elements, selected_indices

    def lambda_L(self):
        return 0

    def lambda_U(self):
        return 0
