"""
CISRearranger module for rearranging copulas to be conditionally increasing in sequence.

This module implements the rearrangement algorithm from:
Strothmann, Dette, Siburg (2022) - "Rearranged dependence measures"
"""

import logging
from typing import Union, Optional, Any, List

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

    def __init__(self, checkerboard_size: Optional[int] = None):
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

    def rearrange_copula(self, copula: Any) -> sympy.Matrix:
        """
        Rearrange a copula to be conditionally increasing in sequence.

        Args:
            copula: A copula object or BivCheckPi object to rearrange

        Returns:
            A sympy Matrix representing the rearranged copula's density
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
        ccop: Union[BivCheckPi, List[List[float]], NDArray, sympy.Matrix, Any],
    ) -> sympy.Matrix:
        """
        Rearrange a checkerboard copula to be conditionally increasing in sequence (CIS),
        using numeric (NumPy) operations for speed. Implements Algorithm 1 from
        Strothmann, Dette, Siburg (2022).

        Parameters
        ----------
        ccop : Union[BivCheckPi, list, np.ndarray, sympy.Matrix, Any]
            The checkerboard copula to rearrange. Can be:
              - a BivCheckPi instance,
              - a 2D list of floats,
              - a 2D numpy.ndarray,
              - a sympy.Matrix,
              - or any object with a `.matr` attribute containing one of the above.

        Returns
        -------
        sympy.Matrix
            The density matrix of the rearranged copula, with shape (n_rows, n_cols).
            Each entry is a sympy-compatible expression (usually float).
        """
        log.debug("Rearranging checkerboard...")

        # ------------------------------------------------------
        # 1. Extract matrix from BivCheckPi or direct input
        # ------------------------------------------------------
        if isinstance(ccop, BivCheckPi):
            matr = ccop.matr
        else:
            matr = ccop  # assumed to be array-like or sympy.Matrix

        # Convert Python lists to np array
        if isinstance(matr, list):
            matr = np.array(matr, dtype=float)
        elif isinstance(matr, sympy.Matrix):
            # Convert sympy Matrix to np array for faster numeric ops
            matr = np.array(matr.tolist(), dtype=float)

        # Ensure matr is now a NumPy 2D array
        if not isinstance(matr, np.ndarray):
            raise TypeError(
                f"Expected a BivCheckPi, list, np.ndarray, or sympy.Matrix. Got: {type(matr)}"
            )
        if matr.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got {matr.ndim}D array.")

        n_rows, n_cols = matr.shape
        matr_sum = matr.sum()
        if matr_sum == 0:
            raise ValueError("Input matrix has sum zero; cannot rearrange.")

        # ------------------------------------------------------
        # 2. Scale the matrix (Condition 3.2 in Strothmann et al.)
        #    => multiply by n_rows / matr_sum
        # ------------------------------------------------------
        matr_scaled = (n_rows / matr_sum) * matr  # shape: (n_rows, n_cols)

        # ------------------------------------------------------
        # 3. Step 1 of the algorithm:
        #    Build partial sums for each row => matrix B
        #    B[k, i] = sum_{j=0..i} matr_scaled[k, j]
        #
        # We'll add a left "zero" column to B => shape is (n_rows, n_cols+1)
        #   B[:, 1:] = row-wise cumsum of matr_scaled
        #   B[:, 0]  = 0
        # ------------------------------------------------------
        partial_sums = np.cumsum(matr_scaled, axis=1)  # shape (n_rows, n_cols)
        B = np.zeros((n_rows, n_cols + 1), dtype=float)
        B[:, 1:] = partial_sums

        # ------------------------------------------------------
        # 4. Step 2: Sort each column of B in descending order => B_tilde
        # ------------------------------------------------------
        B_tilde = np.zeros_like(B)
        for col_idx in range(n_cols + 1):
            col_vals = B[:, col_idx]
            sorted_col = np.sort(col_vals)[::-1]  # descending
            B_tilde[:, col_idx] = sorted_col

        # ------------------------------------------------------
        # 5. Step 3: Compute differences between adjacent columns
        #    => a_arrow = B_tilde[:, 1:] - B_tilde[:, :-1]
        # ------------------------------------------------------
        a_arrow = B_tilde[:, 1:] - B_tilde[:, :-1]  # shape (n_rows, n_cols)

        # ------------------------------------------------------
        # 6. Normalize by (n_rows * n_cols)
        # ------------------------------------------------------
        rearranged_np = a_arrow / (n_rows * n_cols)

        # ------------------------------------------------------
        # 7. Convert to a Sympy Matrix for final return
        # ------------------------------------------------------
        rearranged_sp = sympy.Matrix(rearranged_np)

        log.debug("Rearrangement complete.")
        return rearranged_sp

    @staticmethod
    def verify_cis_property(matrix: Union[np.ndarray, Any]) -> bool:
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
        for k in range(n_rows):
            for i in range(n_cols):
                cum_sums[k, i + 1] = cum_sums[k, i] + matrix_np[k, i]

        # Check if each column is in decreasing order
        for i in range(cum_sums.shape[1]):
            col = cum_sums[:, i]
            if not all(col[j] >= col[j + 1] for j in range(len(col) - 1)):
                return False

        return True


def apply_cis_rearrangement(copula: Any, grid_size: Optional[int] = None) -> BivCheckPi:
    """
    Apply CIS rearrangement to a copula and return as a BivCheckPi object.

    This convenience function rearranges a copula and returns it as a
    BivCheckPi object for easy use in further computations.

    Args:
        copula: The copula to rearrange
        grid_size: Size of the checkerboard grid (optional)

    Returns:
        BivCheckPi: A checkerboard copula with the CIS property
    """
    rearranger = CISRearranger(grid_size)
    rearranged_matrix = rearranger.rearrange_copula(copula)

    # Convert sympy matrix to numpy array
    if hasattr(rearranged_matrix, "tolist") and not isinstance(
        rearranged_matrix, np.ndarray
    ):
        rearranged_np = np.array(rearranged_matrix.tolist(), dtype=float)
    else:
        rearranged_np = rearranged_matrix

    # Create BivCheckPi from rearranged matrix
    rearranged_copula = BivCheckPi(rearranged_np)
    return rearranged_copula
