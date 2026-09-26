"""
CISRearranger module for rearranging copulas to be conditionally increasing in sequence.

This module implements the rearrangement algorithm from:
Strothmann, Dette, Siburg (2022) - "Rearranged dependence measures"
"""

import logging
from typing import Any

import numpy as np
import sympy
from numpy.typing import NDArray

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.checkerboarder import Checkerboarder

# Set up logger
log = logging.getLogger(__name__)


class CISRearranger:
    """
    Class for rearranging copulas to be conditionally increasing in sequence (CIS).

    The rearrangement preserves the checkerboard approximation's margins while
    creating an ordering such that the conditional distribution functions are
    ordered decreasingly with respect to the conditioning value.

    Attributes:
        _checkerboard_size: Size of the checkerboard grid for approximating copulas
    """

    def __init__(self, checkerboard_size: int | None = None):
        """
        Initialize a CISRearranger.

        Args:
            checkerboard_size: Size of checkerboard grid for approximation.
                If None, uses the default size in Checkerboarder.
        """
        self._checkerboard_size = checkerboard_size

    def __str__(self) -> str:
        """Return string representation of the rearranger."""
        return f"CISRearranger(checkerboard_size={self._checkerboard_size})"

    def rearrange_copula(self, copula: Any) -> np.ndarray:
        """
        Rearrange a copula to be conditionally increasing in sequence.

        Args:
            copula: A copula object or copul.checkerboard.biv_check_pi.BivCheckPi object to rearrange

        Returns:
            np.ndarray: the rearranged checkerboard mass matrix (total mass 1)
        """
        # Create checkerboarder with specified grid size
        checkerboarder = Checkerboarder(self._checkerboard_size)

        # If input is already a checkerboard copula, use it directly
        if isinstance(copula, BivCheckPi):
            ccop = copula
        else:
            # Otherwise convert to checkerboard approximation
            log.debug(
                f"Converting copula to checkerboard approximation with grid size {self._checkerboard_size}"
            )
            ccop = checkerboarder.get_checkerboard_copula(copula)

        # Perform the rearrangement
        return self.rearrange_checkerboard(ccop)

    @staticmethod
    def rearrange_checkerboard(
        ccop: BivCheckPi | list[list[float]] | NDArray | sympy.Matrix | Any,
    ) -> np.ndarray:
        """
        Rearrange a checkerboard copula to be stochastically increasing (SI/CIS),
        implementing Algorithm 1 of Strothmann, Dette, Siburg (2022).

        Parameters
        ----------
        ccop : BivCheckPi, list, np.ndarray, sympy.Matrix or object with ``.matr``
            The checkerboard copula (or its mass matrix) to rearrange.

        Returns
        -------
        np.ndarray
            The mass matrix of the rearranged copula, shape ``(n_rows, n_cols)``,
            normalised to total mass one.  Wrap it in ``BivCheckPi`` (or use
            :meth:`BivCheckPi.rearrange_cis`) to obtain a copula object.
        """
        log.debug("Rearranging checkerboard...")

        # 1. Extract the matrix
        matr = getattr(ccop, "matr", ccop)
        if isinstance(matr, sympy.Matrix):
            matr = np.array(matr.tolist(), dtype=float)
        elif isinstance(matr, list):
            matr = np.array(matr, dtype=float)
        if not isinstance(matr, np.ndarray):
            raise TypeError(
                f"Expected a BivCheckPi, list, np.ndarray, or sympy.Matrix. Got: {type(matr)}"
            )
        matr = np.asarray(matr, dtype=float)
        if matr.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got {matr.ndim}D array.")

        n_rows, n_cols = matr.shape
        matr_sum = matr.sum()
        if matr_sum == 0:
            raise ValueError("Input matrix has sum zero; cannot rearrange.")

        # 2. Scale so that the total mass is n_rows (Condition 3.2)
        matr_scaled = (n_rows / matr_sum) * matr

        # 3. Row-wise partial sums with a leading zero column
        B = np.zeros((n_rows, n_cols + 1), dtype=float)
        B[:, 1:] = np.cumsum(matr_scaled, axis=1)

        # 4. Sort every column in descending order
        B_tilde = -np.sort(-B, axis=0)

        # 5. Differences between adjacent columns
        a_arrow = np.diff(B_tilde, axis=1)

        # 6. Normalise to total mass one (each row of a_arrow sums to its
        #    original scaled row mass, i.e. the total is n_rows)
        rearranged = np.clip(a_arrow, 0.0, None) / n_rows

        log.debug("Rearrangement complete.")
        return rearranged

    @staticmethod
    def verify_cis_property(matrix: np.ndarray | Any) -> bool:
        """
        Verify that a matrix has the conditionally increasing in sequence property.

        Args:
            matrix: The matrix to check

        Returns:
            bool: True if the matrix has the CIS property, False otherwise
        """
        # Convert sympy matrix to numpy array for easier processing
        if hasattr(matrix, "tolist") and not isinstance(matrix, np.ndarray):
            matrix_np = np.array(matrix.tolist(), dtype=float)
        else:
            matrix_np = matrix

        n_rows, n_cols = matrix_np.shape

        # Compute cumulative sums for each row
        cum_sums = np.zeros((n_rows, n_cols + 1))
        cum_sums[:, 1:] = np.cumsum(np.asarray(matrix_np, dtype=float), axis=1)

        # Check if each column is in decreasing order
        return bool(np.all(np.diff(cum_sums, axis=0) <= 0))


def apply_cis_rearrangement(copula: Any, grid_size: int | None = None) -> BivCheckPi:
    """
    Apply CIS rearrangement to a copula and return as a BivCheckPi object.

    This convenience function rearranges a copula and returns it as a
    BivCheckPi object for easy use in further computations.

    Args:
        copula: The copula to rearrange
        grid_size: Size of the checkerboard grid (optional)

    Returns:
        copul.checkerboard.biv_check_pi.BivCheckPi: A checkerboard copula with the CIS property
    """
    rearranger = CISRearranger(grid_size)
    rearranged_matrix = rearranger.rearrange_copula(copula)
    return BivCheckPi(np.asarray(rearranged_matrix, dtype=float))
