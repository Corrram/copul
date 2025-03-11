"""
Module for verifying Schur ordering of copula families.

This module provides tools to check if a copula family satisfies the Schur ordering
property as the parameter varies.
"""

import itertools
import logging
import numpy as np
import sympy
from copul.checkerboard.checkerboarder import Checkerboarder
from copul.schur_order.cis_rearranger import CISRearranger

# Configure logging
logger = logging.getLogger(__name__)


class SchurOrderVerifier:
    """
    A class to verify if a copula family satisfies the Schur ordering property.

    Schur ordering is a property related to the dependence structure of copulas.
    This class provides methods to check if a copula family is positively or
    negatively Schur ordered with respect to its parameter.

    Parameters
    ----------
    copula : Copula object
        The copula family to verify.
    n_theta : int, optional
        Number of parameter values to check. Default is 40.
    chess_board_size : int, optional
        Size of the chess board for discretization. Default is 10.
    tolerance : float, optional
        Numerical tolerance for comparisons. Default is 1e-10.
    """

    def __init__(self, copula, n_theta=40, chess_board_size=10, tolerance=1e-10):
        self.copula = copula
        self._n_theta = n_theta
        self._chess_board_size = chess_board_size
        self._tolerance = tolerance

    def verify(self, range_min=None, range_max=None, verbose=True):
        """
        Verify if the copula family is Schur ordered.

        Parameters
        ----------
        range_min : float, optional
            Minimum value of the parameter range. If None, uses the lower bound
            of the copula parameter interval.
        range_max : float, optional
            Maximum value of the parameter range. If None, uses the upper bound
            of the copula parameter interval.
        verbose : bool, optional
            Whether to print progress and result messages. Default is True.

        Returns
        -------
        bool
            True if the copula is Schur ordered (either positively or negatively),
            False otherwise.

        Notes
        -----
        The function checks for positive Schur ordering first. If that fails,
        it checks for negative Schur ordering. A copula is Schur ordered if
        either of these checks passes.
        """
        # Initialize parameter range
        range_min = -10 if range_min is None else range_min
        interval = self.copula.intervals[str(self.copula.params[0])]
        range_min = float(max(interval.inf, range_min)) + 0.01
        range_max = 10 if range_max is None else range_max
        range_max = float(min(interval.end, range_max)) - 0.01

        # Validate parameter range
        if range_min >= range_max:
            msg = f"Invalid parameter range: [{range_min}, {range_max}]"
            logger.error(msg)
            if verbose:
                print(msg)
            return False

        # Generate checkerboard approximation
        try:
            checkerboarder = Checkerboarder(self._chess_board_size)
            checkerboarder.compute_check_pi(self.copula)
        except Exception as e:
            msg = f"Error creating checkerboard approximation: {e}"
            logger.error(msg)
            if verbose:
                print(msg)
            return False

        # Generate parameter values to test
        thetas = np.linspace(range_min, range_max, self._n_theta)
        if len(thetas) < 2:
            msg = f"Need at least 2 parameter values, but got {len(thetas)}"
            logger.error(msg)
            if verbose:
                print(msg)
            return False

        # Generate conditional distributions for different parameter values
        cond_distributions = []
        for theta in thetas:
            try:
                # Get copy of the copula with this parameter value
                copula_with_theta = self._get_copula_with_param(theta)

                # Generate checkerboard for this specific parameter value
                theta_ccop = checkerboarder.compute_check_pi(copula_with_theta)
                rearranged_ccop = CISRearranger().rearrange_checkerboard(theta_ccop)

                # Calculate conditional distributions
                cond_dens = sympy.Matrix.zeros(
                    rearranged_ccop.shape[0], rearranged_ccop.shape[1]
                )
                for k, l_ in np.ndindex(rearranged_ccop.shape):
                    cond_dens[k, l_] = sum(rearranged_ccop[i, l_] for i in range(k + 1))

                cond_distr = sympy.Matrix.zeros(cond_dens.shape[0], cond_dens.shape[1])
                for k, l_ in np.ndindex(cond_dens.shape):
                    cond_distr[k, l_] = sum(cond_dens[k, j] for j in range(l_ + 1))

                cond_distributions.append(cond_distr)
            except Exception as e:
                msg = f"Error processing theta={theta}: {e}"
                logger.error(msg)
                if verbose:
                    print(msg)
                return False

        # Check for Schur ordering
        return self._check_schur_ordering(cond_distributions, thetas, verbose)

    def _get_copula_with_param(self, theta_value):
        """
        Get a copy of the copula with a specific parameter value.

        Parameters
        ----------
        theta_value : float
            The parameter value to set.

        Returns
        -------
        Copula
            A new copula instance with the specified parameter value.
        """
        # Get the copula class
        copula_class = self.copula.__class__

        # Create a new instance
        new_copula = copula_class()

        # Set the parameter value
        param_name = str(self.copula.params[0])
        setattr(new_copula, param_name, theta_value)

        return new_copula

    def _check_schur_ordering(self, cond_distributions, thetas, verbose=True):
        """
        Check if the conditional distributions are Schur ordered.

        Parameters
        ----------
        cond_distributions : list of sympy.Matrix
            List of conditional distribution matrices for different parameter values.
        thetas : ndarray
            Array of parameter values corresponding to the conditional distributions.
        verbose : bool
            Whether to print progress and result messages.

        Returns
        -------
        bool
            True if the copula is Schur ordered (either positively or negatively),
            False otherwise.
        """
        positively_ordered = True

        # Check for positive Schur ordering
        for i in range(len(cond_distributions) - 1):
            smaller_cop = cond_distributions[i]
            larger_cop = cond_distributions[i + 1]

            if positively_ordered and not self._is_pointwise_lower_equal(
                smaller_cop, larger_cop
            ):
                msg = f"Not positively Schur ordered at {thetas[i]} / {thetas[i + 1]}."
                logger.info(msg)
                if verbose:
                    print(msg)

                counterexample = [
                    f"{i} {j}, diff: {larger_cop[i, j] - smaller_cop[i, j]}"
                    for i, j in itertools.product(
                        range(smaller_cop.rows), range(smaller_cop.cols)
                    )
                    if not smaller_cop[i, j] <= larger_cop[i, j] + self._tolerance
                ]

                if verbose and counterexample:
                    print(
                        f"Found {len(counterexample)} violations. Example: {counterexample[:3]}"
                    )

                positively_ordered = False

        # If not positively ordered, check for negative ordering
        if not positively_ordered:
            negatively_ordered = True
            for i in range(len(cond_distributions) - 1):
                smaller_cop = cond_distributions[i + 1]  # Reverse the comparison
                larger_cop = cond_distributions[i]

                if not self._is_pointwise_lower_equal(smaller_cop, larger_cop):
                    msg = f"Not negatively Schur ordered at {thetas[i]} / {thetas[i + 1]}."
                    logger.info(msg)
                    if verbose:
                        print(msg)

                    counterexample = [
                        f"{i} {j}, diff: {larger_cop[i, j] - smaller_cop[i, j]}"
                        for i, j in itertools.product(
                            range(larger_cop.rows), range(larger_cop.cols)
                        )
                        if not smaller_cop[i, j] <= larger_cop[i, j] + self._tolerance
                    ]

                    if verbose and counterexample:
                        print(
                            f"Found {len(counterexample)} violations. Example: {counterexample[:3]}"
                        )

                    negatively_ordered = False
                    break

            if negatively_ordered:
                msg = "Negatively Schur ordered."
                logger.info(msg)
                if verbose:
                    print(msg)
                return True
            else:
                return False
        else:
            msg = "Positively Schur ordered."
            logger.info(msg)
            if verbose:
                print(msg)
            return True

    @staticmethod
    def _is_pointwise_lower_equal(cdf1, cdf2, tolerance=1e-10):
        """
        Check if cdf1 is pointwise less than or equal to cdf2 (with tolerance).

        Parameters
        ----------
        cdf1 : sympy.Matrix
            First matrix to compare.
        cdf2 : sympy.Matrix
            Second matrix to compare.
        tolerance : float, optional
            Tolerance for the comparison. Default is 1e-10.

        Returns
        -------
        bool
            True if cdf1[i,j] <= cdf2[i,j] + tolerance for all i,j, False otherwise.

        Raises
        ------
        ValueError
            If the matrices have different shapes.
        """
        if cdf1.shape != cdf2.shape:
            raise ValueError("Matrices must have the same shape.")

        return all(
            cdf1[i, j] <= cdf2[i, j] + tolerance
            for i, j in itertools.product(range(cdf1.rows), range(cdf1.cols))
        )


if __name__ == "__main__":
    from copul.families import archimedean

    for i in [2, 8, 15, 18, 21]:
        print(f"Nelsen{i}")
        copula = getattr(archimedean, f"Nelsen{i}")()
        SchurOrderVerifier(copula).verify(1)
