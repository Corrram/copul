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
            copula: Copula instance or class to check

        Returns:
            True if the copula is TP2, False otherwise
        """
        result = self.verify_tp2(copula)
        return result.is_tp2

    def verify_tp2(self, copula: Any) -> VerificationResult:
        """
        Verify the TP2 property for a copula and return detailed results.

        Args:
            copula: Copula class or factory to check

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
            log.debug("No parameter ranges detected—treating as a single 'unique' copula")

        # Grid of evaluation points
        test_points = np.linspace(0.0001, 0.9999, 20)
        violations: List[Dict[str, float]] = []
        tested_params: List[Dict[str, float]] = []

        # Iterate through all parameter combinations (empty dict → one iteration)
        for param_values in itertools.product(*parameter_ranges.values()):
            # Build a simple dict of { 'theta': 0.5, ... }
            keys = [str(k) for k in parameter_ranges.keys()]
            param_dict = dict(zip(keys, param_values))
            tested_params.append(param_dict)

            # Instantiate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SymPyDeprecationWarning)
                try:
                    copula_instance = copula(**param_dict)
                except Exception as e:
                    log.warning(f"Error creating copula with params {param_dict}: {e}")
                    continue

            # Skip if no density
            if (
                not hasattr(copula_instance, "is_absolutely_continuous")
                or not copula_instance.is_absolutely_continuous
            ):
                log.info(f"No density for params: {param_dict}")
                continue

            # Symbolic log-pdf
            try:
                log_pdf = sympy.log(copula_instance.pdf)
            except Exception as e:
                log.warning(f"Error computing log-pdf for params {param_dict}: {e}")
                continue

            # Check TP2 on the grid
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
                        violations.append(param_dict)
                        violation_found = True
                        break

            if not violation_found:
                log.info(f"No TP2 violations for params: {param_dict}")

        # Final verdict
        is_tp2 = len(violations) == 0 and len(tested_params) > 0
        return VerificationResult(is_tp2, violations, tested_params)

    def _get_parameter_ranges(self, copula: Any) -> Dict[Symbol, np.ndarray]:
        """
        Get parameter ranges for testing.

        Args:
            copula: Copula class or instance with `.params` and `.intervals`

        Returns:
            Dictionary mapping parameter symbols to arrays of test values
        """
        # Defaults if none provided
        range_min = -10 if self.range_min is None else self.range_min
        range_max = 10 if self.range_max is None else self.range_max

        ranges: Dict[Symbol, np.ndarray] = {}
        num_params = len(getattr(copula, "params", []))

        # Choose how many grid points
        if num_params == 1:
            n_interpolate = 20
        elif num_params == 2:
            n_interpolate = 10
        else:
            n_interpolate = 6

        for param in getattr(copula, "params", []):
            param_str = str(param)
            if param_str not in copula.intervals:
                log.warning(f"Parameter {param_str} not found in intervals")
                continue

            interval = copula.intervals[param_str]
            pmin = float(max(interval.inf, range_min))
            if interval.left_open:
                pmin += 0.01

            # **Use interval.end here (not .sup)**
            pmax = float(min(interval.end, range_max))
            if interval.right_open:
                pmax -= 0.01

            ranges[param] = np.linspace(pmin, pmax, n_interpolate)

        return ranges

    def check_violation(
        self, copula: Any, log_pdf: Expr, x1: float, x2: float, y1: float, y2: float
    ) -> bool:
        """
        Check if the TP2 property is violated at specific points.
        """
        u, v = copula.u, copula.v
        try:
            return self._check_extreme_mixed_term(copula, log_pdf, u, v, x1, x2, y1, y2)
        except Exception as e:
            log.warning(f"Error checking TP2 at points ({x1},{y1}),({x2},{y2}): {e}")
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
        Check TP2 inequality on the log-pdf:
        log f(x1,y1) + log f(x2,y2) ≥ log f(x1,y2) + log f(x2,y1)
        """
        # Substitutions
        min_term = log_pdf.subs(u, x1).subs(v, y1)
        max_term = log_pdf.subs(u, x2).subs(v, y2)
        mix1 = log_pdf.subs(u, x1).subs(v, y2)
        mix2 = log_pdf.subs(u, x2).subs(v, y1)

        extreme = min_term + max_term
        mixed = mix1 + mix2

        # Direct numeric/symbolic comparison
        try:
            comp = extreme * 0.9999999999999 < mixed
        except TypeError:
            # Fallback to real parts if complex
            try:
                extreme = extreme.as_real_imag()[0]
                mixed = mixed.as_real_imag()[0]
                comp = extreme * 0.9999999999999 < mixed
            except Exception:
                # Last resort: retry with fresh symbols
                return self._check_extreme_mixed_term(
                    copula, log_pdf, copula.u, copula.v, x1, x2, y1, y2
                )

        # Ensure a bool
        if not isinstance(comp, (bool, BooleanFalse, BooleanTrue)):
            comp = comp.evalf()
            if not isinstance(comp, (bool, BooleanFalse, BooleanTrue)):
                return self._check_extreme_mixed_term(
                    copula, log_pdf, copula.u, copula.v, x1, x2, y1, y2
                )

        if comp:
            log.debug(
                f"TP2 violation at ({x1},{y1}),({x2},{y2}): extreme={extreme}, mixed={mixed}"
            )
        return bool(comp)


def verify_copula_tp2(
    copula: Any, range_min: Optional[float] = None, range_max: Optional[float] = None
) -> VerificationResult:
    """
    Convenience function to verify if a copula satisfies the TP2 property.
    """
    verifier = TP2Verifier(range_min, range_max)
    return verifier.verify_tp2(copula)
