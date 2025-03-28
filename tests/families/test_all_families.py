import logging

import numpy as np
import pytest

import copul
from copul.exceptions import PropertyUnavailableException
from tests.family_representatives import (
    archimedean_representatives,
    family_representatives,
)

log = logging.getLogger(__name__)


import pytest
import numpy as np
import copul

@pytest.mark.parametrize(
    "point, expected", 
    [
        pytest.param((0, 0), 0, id="point=(0,0)"),
        pytest.param((0, 0.5), 0, id="point=(0,0.5)"), 
        pytest.param((1, 0.5), 0.5, id="point=(1,0.5)"),
        pytest.param((1, 1), 1, id="point=(1,1)")
    ]
)
@pytest.mark.parametrize(
    "copula_name", 
    [pytest.param(name, id=name) for name in family_representatives.keys()]
)
def test_cdf_edge_cases(point, expected, copula_name):
    """
    Test CDF values at edge cases for different copula families.
    
    This doubly-parameterized test creates a separate test case for each combination
    of point/expected value and copula family.
    
    Parameters
    ----------
    point : tuple
        The point (u, v) at which to evaluate the CDF
    expected : float
        The expected value of the CDF at this point
    copula_name : str
        Name of the copula family to test
    """
    # Get the parameter for this copula family
    param = family_representatives[copula_name]
    
    # Create the copula instance
    cop_class = getattr(copul, copula_name)
    if param is None:
        cop = cop_class()
    elif isinstance(param, tuple):
        cop = cop_class(*param)
    else:
        cop = cop_class(param)
    
    # Evaluate the CDF at the given point
    evaluated_cdf = cop.cdf(*point)
    actual = evaluated_cdf.evalf() if hasattr(evaluated_cdf, 'evalf') else float(evaluated_cdf)
    
    # Test with expected value
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), \
        f"{copula_name} CDF at {point} = {actual}, expected {expected}"



@pytest.mark.parametrize(
    "method_name, point, expected_value",
    [
        # Test conditional distributions at (0,0) - should be 0
        ("cond_distr_1", (0.0, 0.0), 0.0),
        ("cond_distr_2", (0.0, 0.0), 0.0),
        
        # Test conditional distributions at corner points
        ("cond_distr_1", (0.5, 0), 0.0),  # P(U₁≤0|U₂=0.5) = 0
        ("cond_distr_2", (0.0, 0.5), 0.0),  # P(U₂≤0|U₁=0.5) = 0
        
        # Test at (1,u) and (u,1) - conditional should evaluate to 1
        ("cond_distr_1", (0.7, 1.0), 1.0),  # P(U₁≤1|U₂=0.7) = 1
        ("cond_distr_2", (1.0, 0.7), 1.0),  # P(U₂≤1|U₁=0.7) = 1
    ],
)
@pytest.mark.parametrize("copula_name", [
    pytest.param(name, id=name) for name in family_representatives.keys()
])
def test_cond_distr_edge_cases(method_name, point, expected_value, copula_name):
    """
    Test conditional distributions at edge cases for different copula families.
    
    This doubly-parameterized test creates a separate test case for each combination
    of method/point/expected_value and copula family.
    
    Parameters
    ----------
    method_name : str
        Name of the method to test ('cond_distr_1' or 'cond_distr_2')
    point : tuple
        The point (u, v) at which to evaluate the conditional distribution
    expected_value : float
        The expected value of the conditional distribution at this point
    copula_name : str
        Name of the copula family to test
    """
    # Get the parameter for this copula family
    param = family_representatives[copula_name]
    
    # Create the copula instance
    cop_class = getattr(copul, copula_name)
    if param is None:
        cop = cop_class()
    elif isinstance(param, tuple):
        cop = cop_class(*param)
    else:
        cop = cop_class(param)
    
    # Get the method to test
    method = getattr(cop, method_name)
    
    # Evaluate the method at the given point
    try:
        result = method(*point)
        # Convert to float if it's a callable or wrapper
        if callable(result):
            result = float(result)
        
        # Test with expected value
        assert np.isclose(result, expected_value, rtol=1e-10, atol=1e-10), \
            f"{copula_name}.{method_name}({point}) = {result}, expected {expected_value}"
            
    except ValueError as e:
        # Re-raise with more context
        raise ValueError(f"Error evaluating {copula_name}.{method_name}({point}): {e}")


@pytest.mark.parametrize(
    "copula_name", 
    [pytest.param(name, id=name) for name in archimedean_representatives.keys()]
)
def test_pdf_values(copula_name):
    """
    Test PDF values for different copula families.
    
    Each copula family is tested as a separate test case, checking that
    the PDF value at (0.5, 0.5) is non-negative.
    
    Parameters
    ----------
    copula_name : str
        Name of the copula family to test
    """
    # Get the parameter for this copula family
    param = archimedean_representatives[copula_name]
    
    # Create the copula instance
    cop_class = getattr(copul, copula_name)
    if isinstance(param, tuple):
        cop = cop_class(*param)
    else:
        cop = cop_class(param)
    
    # Try to get the pdf, skip if unavailable
    try:
        pdf = cop.pdf()
    except PropertyUnavailableException:
        pytest.skip(f"PDF not available for {copula_name}")
        return
    
    # Evaluate the PDF at (0.5, 0.5)
    evaluated_pdf = pdf(0.5, 0.5).evalf()
    log.info(f"{copula_name} pdf at (0.5, 0.5): {evaluated_pdf}")
    
    # Assert that PDF is non-negative
    assert evaluated_pdf >= 0, f"{copula_name} PDF at (0.5, 0.5) is negative: {evaluated_pdf}"
