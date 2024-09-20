import logging
from typing import Union

import numpy as np
import pandas as pd

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.check_pi import CheckPi

log = logging.getLogger(__name__)


class Checkerboarder:
    def __init__(self, n: Union[int, list] = None, dim=2):
        if n is None:
            n = 20
        if isinstance(n, int):
            n = [n] * dim
        self.n = n
        self.d = len(self.n)

    def compute_check_pi(self, copula):
        log.debug(
            "Computing checkerboard copula for d-dimensional case "
            "with different grid sizes..."
        )

        # Matrix to store the copula values
        cmatr = np.zeros(self.n)

        # Create grid indices for each dimension
        indices = np.ndindex(*self.n)

        for idx in indices:
            # Generate the edges of the hypercube for each dimension based on the index
            u_lower = [i / self.n[k] for k, i in enumerate(idx)]
            u_upper = [(i + 1) / self.n[k] for k, i in enumerate(idx)]

            # Initialize the CDF terms for inclusion-exclusion principle
            inclusion_exclusion_sum = 0

            # Compute the CDF for all corners of the hypercube using the inclusion-exclusion principle
            for corner in range(
                1 << self.d
            ):  # Iterate over 2^d corners of the hypercube
                corner_indices = [
                    (u_upper[k] if corner & (1 << k) else u_lower[k])
                    for k in range(self.d)
                ]
                sign = (-1) ** (
                    bin(corner).count("1") + 2
                )  # Use inclusion-exclusion principle
                cdf_value = copula.cdf(*corner_indices)
                inclusion_exclusion_sum += sign * cdf_value.evalf()

            # Assign the result to the copula matrix
            cmatr[idx] = inclusion_exclusion_sum
        return CheckPi(cmatr) if self.d > 2 else BivCheckPi(cmatr)

    def from_data(self, data: Union[pd.DataFrame, np.ndarray, list]):
        if isinstance(data, (list, np.ndarray)):
            data = pd.DataFrame(data)
        # transform each column to ranks
        for col in data.columns:
            data[col] = data[col].rank(pct=True)
        n_obs = len(data)
        data = data.sort_values(by=data.columns[0])
        check_pi_matr = np.ndarray(self.n)
        for i in range(self.n[0]):
            for j in range(self.n[1]):
                # count rows in the i-th and j-th quantile
                n_ij = len(
                    data[
                        (data[data.columns[0]] >= i / self.n[0])
                        & (data[data.columns[0]] < (i + 1) / self.n[0])
                        & (data[data.columns[1]] >= j / self.n[1])
                        & (data[data.columns[1]] < (j + 1) / self.n[1])
                    ]
                )
                check_pi_matr[i, j] = n_ij / n_obs
        return CheckPi(check_pi_matr) if self.d > 2 else BivCheckPi(check_pi_matr)


def from_data(data, checkerboard_size=None):
    return Checkerboarder(checkerboard_size, data.shape[1]).from_data(data)
