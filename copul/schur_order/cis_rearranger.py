import copy
import logging

import numpy as np
import sympy

from copul import BivCheckPi
from copul.schur_order.checkerboarder import Checkerboarder

log = logging.getLogger(__name__)


class CISRearranger:
    def __init__(self, checkerboard_size=None):
        self._checkerboard_size = checkerboard_size

    def __str__(self):
        return f"CISRearranger(checkerboard_size={self._checkerboard_size})"

    def rearrange_copula(self, copula):
        checkerboarder = Checkerboarder(self._checkerboard_size)
        if isinstance(copula, BivCheckPi):
            ccop = copula
        else:
            ccop = checkerboarder.compute_check_pi(copula)
        return self.rearrange_checkerboard(ccop)

    @staticmethod
    def rearrange_checkerboard(ccop):
        """
        Implements Algorithm 1 on p.8 from 2022 Strothmann, Dette, Siburg - Rearranged
        dependence measures. Computes the rearranged copula from the checkerboard,
        which is CIS (with respect to conditioning on the first variable, which
        corresponds to the matrix row entries).
        """
        log.info("Rearranging checkerboard...")
        if isinstance(ccop, BivCheckPi):
            matr = ccop.matr
        else:
            matr = ccop
        if isinstance(matr, list):
            matr = np.array(matr)
        matr = matr.shape[0] * matr / sum(matr)  # asserts 3.2
        matr = np.nan_to_num(matr)

        # step (1)
        B = sympy.Matrix.zeros(matr.shape[0], matr.shape[1])
        for k, l in np.ndindex(matr.shape):
            B[k, l] = sum(matr[k, j] for j in range(l + 1))
        B = B.col_insert(0, sympy.Matrix([0] * matr.shape[0]))

        # step (2)
        B_tilde = sympy.Matrix.zeros(B.shape[0], B.shape[1])
        for l_ in range(B.shape[1]):
            B_tilde.col_del(l_)
            sorted_col = sorted(B.col(l_), reverse=True)
            insert_val = sympy.Matrix(sorted_col)
            B_tilde = B_tilde.col_insert(l_, insert_val)

        # step (3)
        a_arrow = sympy.Matrix.zeros(matr.shape[0], matr.shape[1])
        for k, l in np.ndindex(matr.shape):
            a_arrow[k, l] = B_tilde[k, l + 1] - B_tilde[k, l]
        a_arrow_final = copy.copy(a_arrow)

        return a_arrow_final / (matr.shape[0] * matr.shape[1])
