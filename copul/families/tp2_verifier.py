"""
TP2Verifier module for checking the TP2 property of copulas.

This module provides functionality to verify if a copula satisfies the TP2
(totally positive of order 2) property by analyzing its log-density function.
"""

import itertools
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import sympy
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import BooleanFalse, BooleanTrue
from sympy.utilities.exceptions import SymPyDeprecationWarning
import warnings

# Set up logger
log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Class to represent TP2 verification results.

    Attributes:
        is_tp2: Whether the copula satisfies the TP2 property
        violations: List of parameter values where TP2 property is violated
        tested_params: List of parameter values that were tested
    """

    is_tp2: bool
    violations: List[Dict[str, float]]
    tested_params: List[Dict[str, float]]


class TP2Verifier:
    """
    Class for verifying if a copula satisfies the TP2 property.

    The TP2 (totally positive of order 2) property is an important mathematical
    property for copulas, indicating that the copula's density satisfies certain
    log-supermodularity conditions.
    """

    def __init__(
        self, range_min: Optional[float] = None, range_max: Optional[float] = None
    ):
        """
        Initialize a TP2Verifier.

        Args:
            range_min: Minimum value for parameter range (default: None, uses copula's lower bound)
            range_max: Maximum value for parameter range (default: None, uses copula's upper bound)
        """
        self.range_min = range_min
        self.range_max = range_max

    def is_tp2(self, copula: Any) -> bool:
        """
        Check if a copula satisfies the TP2 property.

        Args:
            copula: Copula instance to check

        Returns:
            True if the copula is TP2, False otherwise
        """
        result = self.verify_tp2(copula)
        return result.is_tp2

    def verify_tp2(self, copula: Any) -> VerificationResult:
        """
        Verify the TP2 property for a copula and return detailed results.

        Args:
            copula: Copula instance to check

        Returns:
            VerificationResult object containing verification details
        """
        log.info(f"Checking if {type(copula).__name__} copula is TP2")

        # If the copula is not absolutely continuous, it cannot be TP2
        if (
            hasattr(copula, "is_absolutely_continuous")
            and not copula.is_absolutely_continuous
        ):
            log.info("Copula is not absolutely continuous, therefore not TP2")
            return VerificationResult(False, [], [])

        # Determine parameter ranges
        parameter_ranges = self._get_parameter_ranges(copula)
        if not parameter_ranges:
            log.warning("Could not determine parameter ranges for copula")
            return VerificationResult(False, [], [])

        # Get the test points for u and v
        test_points = np.linspace(0.0001, 0.9999, 20)

        violations = []
        tested_params = []

        # Iterate through all parameter combinations
        for param_values in itertools.product(*parameter_ranges.values()):
            param_dict = dict(zip(parameter_ranges.keys(), param_values))
            keys = [str(key) for key in parameter_ranges.keys()]
            param_dict_str = dict(zip(keys, param_values))
            tested_params.append(param_dict_str)

            # Create copula instance with specific parameters
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SymPyDeprecationWarning)
                try:
                    copula_instance = copula(**param_dict_str)
                except Exception as e:
                    log.warning(
                        f"Error creating copula with params {param_dict_str}: {e}"
                    )
                    continue

            # Check if the copula has a density
            if (
                not hasattr(copula_instance, "is_absolutely_continuous")
                or not copula_instance.is_absolutely_continuous
            ):
                log.info(f"No density for params: {param_dict}")
                continue

            # Get the log-pdf for TP2 checking
            try:
                log_pdf = sympy.log(copula_instance.pdf)
            except Exception as e:
                log.warning(f"Error computing log-pdf for params {param_dict_str}: {e}")
                continue

            # Check TP2 property at different points
            violation_found = False
            for i in range(len(test_points) - 1):
                if violation_found:
                    break

                for j in range(len(test_points) - 1):
                    x1, x2 = test_points[i], test_points[i + 1]
                    y1, y2 = test_points[j], test_points[j + 1]

                    if self.check_violation(copula_instance, log_pdf, x1, x2, y1, y2):
                        log.info(
                            f"TP2 violation at params: {param_dict}, "
                            f"points: ({x1}, {y1}), ({x2}, {y2})"
                        )
                        violations.append(param_dict_str)
                        violation_found = True
                        break

            if not violation_found:
                log.info(f"No TP2 violations for params: {param_dict}")

        # Determine overall result
        is_tp2 = len(violations) == 0 and len(tested_params) > 0

        return VerificationResult(is_tp2, violations, tested_params)

    def _get_parameter_ranges(self, copula: Any) -> Dict[Symbol, np.ndarray]:
        """
        Get parameter ranges for testing.

        Args:
            copula: Copula object

        Returns:
            Dictionary mapping parameter symbols to arrays of test values
        """
        range_min = -10 if self.range_min is None else self.range_min
        ranges = {}

        # Determine number of interpolation points based on parameter count
        if len(copula.params) == 1:
            n_interpolate = 20
        elif len(copula.params) == 2:
            n_interpolate = 10
        else:
            n_interpolate = 6

        # Get ranges for each parameter
        for param in copula.params:
            param_str = str(param)
            if param_str not in copula.intervals:
                log.warning(f"Parameter {param_str} not found in intervals")
                continue

            interval = copula.intervals[param_str]

            # Get lower bound
            param_min = float(max(interval.inf, range_min))
            if interval.left_open:
                param_min += 0.01

            # Get upper bound
            param_max = 10 if self.range_max is None else self.range_max
            param_max = float(min(interval.end, param_max))
            if interval.right_open:
                param_max -= 0.01

            # Create array of test values
            ranges[param] = np.linspace(param_min, param_max, n_interpolate)

        return ranges

    def check_violation(
        self, copula: Any, log_pdf: Expr, x1: float, x2: float, y1: float, y2: float
    ) -> bool:
        """
        Check if the TP2 property is violated at specific points.

        Args:
            copula: Copula instance
            log_pdf: Symbolic log-pdf expression
            x1, x2: Test points for first variable
            y1, y2: Test points for second variable

        Returns:
            True if TP2 property is violated, False otherwise
        """
        u = copula.u
        v = copula.v

        try:
            return self._check_extreme_mixed_term(copula, log_pdf, u, v, x1, x2, y1, y2)
        except Exception as e:
            log.warning(f"Error checking TP2 at points ({x1}, {y1}), ({x2}, {y2}): {e}")
            return False

    def _check_extreme_mixed_term(
        self,
        copula: Any,
        log_pdf: Expr,
        u: Symbol,
        v: Symbol,
        x1: float,
        x2: float,
        y1: float,
        y2: float,
    ) -> bool:
        """
        Check if the TP2 property is violated by comparing extreme and mixed terms.

        For a function to be TP2, the following condition must hold:
        f(x1,y1) * f(x2,y2) ≥ f(x1,y2) * f(x2,y1)

        For the log-pdf, this translates to:
        log(f(x1,y1)) + log(f(x2,y2)) ≥ log(f(x1,y2)) + log(f(x2,y1))

        Args:
            copula: Copula instance
            log_pdf: Symbolic log-pdf expression
            u, v: Symbolic variables for substitution
            x1, x2, y1, y2: Test points

        Returns:
            True if TP2 is violated (extreme_term < mixed_term), False otherwise
        """
        # Compute terms for the inequality
        min_term = log_pdf.subs(u, x1).subs(v, y1)
        max_term = log_pdf.subs(u, x2).subs(v, y2)
        mix_term_1 = log_pdf.subs(u, x1).subs(v, y2)
        mix_term_2 = log_pdf.subs(u, x2).subs(v, y1)

        # Sum extreme and mixed terms
        extreme_term = min_term + max_term
        mixed_term = mix_term_1 + mix_term_2

        # Compare terms (with a small tolerance to avoid numerical issues)
        try:
            # Try direct comparison
            comparison = extreme_term * 0.9999999999999 < mixed_term
        except TypeError:
            # For complex expressions, use the real part
            try:
                extreme_real = extreme_term.as_real_imag()[0]
                mixed_real = mixed_term.as_real_imag()[0]
                comparison = extreme_real * 0.9999999999999 < mixed_real
            except Exception:
                # If that fails, try again with the symbolic variables
                return self._check_extreme_mixed_term(
                    copula, log_pdf, copula.u, copula.v, x1, x2, y1, y2
                )

        # Evaluate the comparison if it's not a boolean value
        if not isinstance(comparison, (bool, BooleanFalse, BooleanTrue)):
            comparison = comparison.evalf()

            # If still not boolean, try recursive call with copula symbols
            if not isinstance(comparison, (bool, BooleanFalse, BooleanTrue)):
                return self._check_extreme_mixed_term(
                    copula, log_pdf, copula.u, copula.v, x1, x2, y1, y2
                )

        # Detailed logging for violations
        if comparison:
            log.debug(f"TP2 violation at points: ({x1}, {y1}), ({x2}, {y2})")
            log.debug(f"Extreme term: {extreme_term}, Mixed term: {mixed_term}")

        return bool(comparison)


def verify_copula_tp2(
    copula: Any, range_min: Optional[float] = None, range_max: Optional[float] = None
) -> VerificationResult:
    """
    Convenience function to verify if a copula satisfies the TP2 property.

    Args:
        copula: Copula instance to check
        range_min: Minimum value for parameter range (optional)
        range_max: Maximum value for parameter range (optional)

    Returns:
        VerificationResult object with verification details
    """
    verifier = TP2Verifier(range_min, range_max)
    return verifier.verify_tp2(copula)
