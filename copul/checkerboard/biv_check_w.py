import logging
from typing import TypeAlias

import numpy as np

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.exceptions import PropertyUnavailableException

log = logging.getLogger(__name__)


class BivCheckW(BivCheckPi):
    """
    Bivariate checkerboard W-copula (2D only).

    Inside every cell the mass is placed on the falling cell diagonal, i.e.
    the cell-local distribution function is ``max(0, a + b - 1)`` with the
    cell-local coordinates ``a = m u - i``, ``b = n v - j``.

    All numerics are exact, see :mod:`copul.checkerboard._biv_engine`.
    Relative to :class:`BivCheckPi` with the same matrix ``Delta``::

        rho  -= 1 / (m n)
        tau  -= sum Delta_ij^2
        xi   += (m / n) sum Delta_ij^2
        nu   -= sum_i r_i (2m - 2i - 1) / (m^2 n),   r_i = row sums (0-based i)

    and, for square grids (``n x n``),
    ``footrule = footrule_Pi - tr(Delta) / (2n)`` and
    ``gini = gini_Pi - tr(Delta) / (3n) - 2 antitr(Delta) / (3n)``.
    """

    def __init__(self, matr, **kwargs):
        """
        Initialize the 2D W-copula with a matrix of nonnegative weights.

        :param matr: 2D array/list of nonnegative weights. Will be normalized to sum=1.
        """
        super().__init__(matr, **kwargs)

    def _kernel_signs(self):
        return -1

    def __str__(self):
        return f"BivCheckW(m={self.m}, n={self.n})"

    @property
    def is_absolutely_continuous(self):
        """Checkerboard W-copula is not absolutely continuous (singular cell diagonals)."""
        return False

    @property
    def is_symmetric(self):
        """Check if m = n and the matrix is symmetric about the diagonal."""
        if self.m != self.n:
            return False
        return np.allclose(self.matr, self.matr.T)

    @property
    def pdf(self):
        """PDF is not available for BivCheckW.

        Raises:
            PropertyUnavailableException: Always raised, since PDF does not exist for BivCheckW.
        """
        raise PropertyUnavailableException("PDF does not exist for BivCheckW.")


CheckW: TypeAlias = BivCheckW

if __name__ == "__main__":
    copula = BivCheckW([[1, 1]])
    print(f"Footrule: {copula.spearmans_footrule()}, Rho: {copula.spearmans_rho()}")
