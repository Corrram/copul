from typing import Union

import numpy as np

from copul import BivCheckPi
from copul.checkerboard.check_min import CheckMin


class BivCheckMin(CheckMin, BivCheckPi):

    def __init__(self, matr, mc_size=200_000, **kwargs):
        CheckMin.__init__(self, matr)
        BivCheckPi.__init__(self, matr, **kwargs)
        self.m = self.matr.shape[0]
        self.n = self.matr.shape[1]
        self.n_samples = mc_size

    def __str__(self):
        return f"CheckMin(m={self.m}, n={self.n})"

    @property
    def is_symmetric(self) -> bool:
        if self.matr.shape[0] != self.matr.shape[1]:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def is_absolutely_continuous(self) -> bool:
        return False
