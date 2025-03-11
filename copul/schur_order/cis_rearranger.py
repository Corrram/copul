"""
CISRearranger module for rearranging copulas to be conditionally increasing in sequence.

This module implements the rearrangement algorithm from:
Strothmann, Dette, Siburg (2022) - "Rearranged dependence measures"
"""

import copy
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
            ccop = checkerboarder.compute_check_pi(copula)

        # Perform the rearrangement
        return self.rearrange_checkerboard(ccop)

    @staticmethod
    def rearrange_checkerboard(
        ccop: Union[BivCheckPi, List[List[float]], NDArray, Any],
    ) -> Any:
        """
        Rearrange a checkerboard copula to be conditionally increasing in sequence (CIS).

        Implements Algorithm 1 from Strothmann, Dette, Siburg (2022), computing the
        rearranged copula which is CIS with respect to conditioning on the first variable
        (corresponding to matrix row entries).

        Args:
            ccop: Either a BivCheckPi object or a matrix (as list, numpy array, or sympy Matrix)
                 representing the checkerboard copula density

        Returns:
            sympy.Matrix: The density matrix of the rearranged copula

        Note:
            The algorithm preserves the margins of the original copula while
            ensuring that the conditional distributions are ordered by the
            conditioning value.
        """
        log.info("Rearranging checkerboard...")

        # Extract the matrix from the input
        if isinstance(ccop, BivCheckPi):
            matr = ccop.matr
        else:
            matr = ccop

        # Convert list to numpy array if needed
        if isinstance(matr, list):
            matr = np.array(matr)

        # Validate input matrix
        if not isinstance(matr, (np.ndarray, list)) and not hasattr(matr, "shape"):
            raise TypeError(
                f"Expected numpy array, sympy Matrix, or list, got {type(matr)}"
            )

        if isinstance(matr, np.ndarray) and matr.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got {matr.ndim}D")

        # Normalize matrix
        # Handle sympy Matrix
        if hasattr(matr, "shape") and not isinstance(matr, np.ndarray):
            matr_sum = sum(matr)
        else:
            # Handle numpy array
            matr_sum = matr.sum()
        if matr_sum == 0:
            raise ValueError("Input matrix sum is zero")

        # Scale to satisfy condition 3.2 from the paper
        n_rows, n_cols = matr.shape
        matr = n_rows * matr / matr_sum

        # Handle potential NaN values from division
        if isinstance(matr, np.ndarray):
            matr = np.nan_to_num(matr)

        # Step 1: Compute cumulative sums for each row
        B = sympy.Matrix.zeros(n_rows, n_cols)
        for k in range(n_rows):
            for i in range(n_cols):
                # Sum from j=0 to i for each row k
                B[k, i] = sum(matr[k, j] for j in range(i + 1))

        # Insert zero column at the beginning (representing F(u,0))
        zero_col = sympy.Matrix([0] * n_rows)
        B = B.col_insert(0, zero_col)

        # Step 2: Rearrange each column in decreasing order
        B_tilde = sympy.Matrix.zeros(B.shape[0], B.shape[1])
        for i in range(B.shape[1]):
            # Remove the current column (to later insert sorted column)
            B_tilde.col_del(i)

            # Sort column values in decreasing order
            col_values = B.col(i)
            sorted_col_values = sorted(col_values, reverse=True)

            # Insert sorted column
            B_tilde = B_tilde.col_insert(i, sympy.Matrix(sorted_col_values))

        # Step 3: Compute differences between consecutive columns to get density
        a_arrow = sympy.Matrix.zeros(n_rows, n_cols)
        for k in range(n_rows):
            for i in range(n_cols):
                # Density is the difference between consecutive CDF values
                a_arrow[k, i] = B_tilde[k, i + 1] - B_tilde[k, i]

        # Create a copy to avoid potential reference issues
        rearranged_density = copy.copy(a_arrow)

        # Normalize to get proper density (dividing by grid size)
        normalized_density = rearranged_density / (n_rows * n_cols)

        log.info("Rearrangement complete")
        return normalized_density

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
