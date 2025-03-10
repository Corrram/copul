import numpy as np
import pytest
import sympy

from copul.families.bivcopula import BivCopula
from copul.families.other.farlie_gumbel_morgenstern import FarlieGumbelMorgenstern


@pytest.fixture
def copula():
    """Create a FarlieGumbelMorgenstern copula instance with theta=0.5 for tests."""
    return FarlieGumbelMorgenstern(theta=0.5)


def test_initialization():
    """Test that the FGM copula can be initialized with different parameters."""
    # Default initialization
    fgm = FarlieGumbelMorgenstern()
    assert str(fgm.theta) == "theta"

    # With parameter
    fgm = FarlieGumbelMorgenstern(theta=0.5)
    assert float(fgm.theta) == 0.5

    # Edge case parameters
    fgm_min = FarlieGumbelMorgenstern(theta=-1)
    assert float(fgm_min.theta) == -1

    fgm_max = FarlieGumbelMorgenstern(theta=1)
    assert float(fgm_max.theta) == 1


def test_parameter_constraints():
    """Test parameter constraints: -1 ≤ theta ≤ 1."""
    # Valid parameters
    FarlieGumbelMorgenstern(theta=-1)  # Lower bound
    FarlieGumbelMorgenstern(theta=0)  # Middle
    FarlieGumbelMorgenstern(theta=1)  # Upper bound

    # Invalid parameters (theta < -1 or theta > 1)
    with pytest.raises(ValueError):
        FarlieGumbelMorgenstern(theta=-1.1)

    with pytest.raises(ValueError):
        FarlieGumbelMorgenstern(theta=1.1)


def test_inheritance():
    """Test that FGM properly inherits from BivCopula."""
    copula = FarlieGumbelMorgenstern()
    assert isinstance(copula, BivCopula)


def test_is_symmetric(copula):
    """Test that the FGM copula is symmetric."""
    assert copula.is_symmetric is True


def test_is_absolutely_continuous(copula):
    """Test the absolutely continuous property."""
    assert copula.is_absolutely_continuous is True


def test_cdf_values(copula):
    """Test specific CDF values."""
    theta = 0.5

    def fgm_cdf(u, v, theta):
        """Compute the FGM copula CDF value."""
        return u * v + theta * u * v * (1 - u) * (1 - v)

    test_cases = [
        (0.5, 0.5, fgm_cdf(0.5, 0.5, theta)),  # Symmetric point
        (0.3, 0.7, fgm_cdf(0.3, 0.7, theta)),  # Asymmetric point
        (0.0, 0.5, 0.0),  # Boundary u=0
        (0.5, 0.0, 0.0),  # Boundary v=0
        (1.0, 0.5, 0.5),  # Boundary u=1
        (0.5, 1.0, 0.5),  # Boundary v=1
        (1.0, 1.0, 1.0),  # Boundary u=v=1
    ]

    for u, v, expected in test_cases:
        cdf_val = float(copula.cdf(u=u, v=v))
        assert abs(cdf_val - expected) < 1e-10, f"CDF value incorrect for u={u}, v={v}"


def test_pdf_values(copula):
    """Test specific PDF values."""
    theta = 0.5

    def fgm_pdf(u, v, theta):
        """Compute the FGM copula PDF value."""
        return 1 + theta * (1 - 2 * u) * (1 - 2 * v)

    test_cases = [
        (0.5, 0.5, fgm_pdf(0.5, 0.5, theta)),  # Center point
        (0.25, 0.25, fgm_pdf(0.25, 0.25, theta)),  # Both < 0.5
        (0.75, 0.75, fgm_pdf(0.75, 0.75, theta)),  # Both > 0.5
        (0.25, 0.75, fgm_pdf(0.25, 0.75, theta)),  # Mixed
    ]

    for u, v, expected in test_cases:
        pdf_val = float(copula.pdf(u=u, v=v))
        assert abs(pdf_val - expected) < 1e-10, f"PDF value incorrect for u={u}, v={v}"


def test_pdf_integrates_to_one():
    """Test that the PDF integrates to 1 over the unit square."""
    from scipy import integrate

    for theta in [-0.5, 0, 0.5]:
        copula = FarlieGumbelMorgenstern(theta=theta)

        def pdf_func(u, v):
            return float(copula.pdf(u=u, v=v))

        # Integrate PDF over unit square
        result, _ = integrate.nquad(
            lambda u, v: pdf_func(u, v),
            [[0, 1], [0, 1]],
            opts={"epsabs": 1e-6, "epsrel": 1e-6},
        )

        assert abs(result - 1.0) < 1e-6, (
            f"PDF does not integrate to 1 for theta={theta}, got {result}"
        )


def test_pdf_range(copula):
    """Test that PDF values are within the expected range."""
    # For theta=0.5, PDF should be between 0.5 and 1.5
    min_val = 1 - abs(float(copula.theta))
    max_val = 1 + abs(float(copula.theta))

    test_points = [
        (0.1, 0.1),
        (0.9, 0.9),  # Same quadrant points
        (0.1, 0.9),
        (0.9, 0.1),  # Different quadrant points
        (0.5, 0.5),  # Center point
    ]

    for u, v in test_points:
        pdf_val = float(copula.pdf(u=u, v=v))
        assert min_val <= pdf_val <= max_val, (
            f"PDF value outside expected range for u={u}, v={v}"
        )


def test_boundary_cases(copula):
    """Test that the copula behaves correctly at boundary values."""
    # Create a range of test values
    u_vals = np.linspace(0.1, 0.9, 5)

    # At (0, v) and (u, 0), copula should be 0
    for u in u_vals:
        # Get CDF values
        cdf_u0 = float(copula.cdf(u=u, v=0))
        cdf_0v = float(copula.cdf(u=0, v=u))

        assert abs(cdf_u0) < 1e-10, f"C({u},0) should be 0, got {cdf_u0}"
        assert abs(cdf_0v) < 1e-10, f"C(0,{u}) should be 0, got {cdf_0v}"

    # At (1, v), copula should be v
    # At (u, 1), copula should be u
    for u in u_vals:
        cdf_u1 = float(copula.cdf(u=u, v=1))
        cdf_1v = float(copula.cdf(u=1, v=u))

        assert abs(cdf_u1 - u) < 1e-10, f"C({u},1) should be {u}, got {cdf_u1}"
        assert abs(cdf_1v - u) < 1e-10, f"C(1,{u}) should be {u}, got {cdf_1v}"


def test_special_case_independence():
    """Test the independence case (theta=0)."""
    copula = FarlieGumbelMorgenstern(theta=0)

    test_points = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    for u, v in test_points:
        # For independence, C(u,v) = u*v
        cdf_val = float(copula.cdf(u=u, v=v))
        expected = u * v
        assert abs(cdf_val - expected) < 1e-10, (
            f"CDF incorrect for independence case at u={u}, v={v}"
        )

        # For independence, PDF = 1
        pdf_val = float(copula.pdf(u=u, v=v))
        assert abs(pdf_val - 1) < 1e-10, (
            f"PDF incorrect for independence case at u={u}, v={v}"
        )


def test_conditional_distribution():
    """Test that conditional distribution is properly defined."""
    copula = FarlieGumbelMorgenstern(theta=0.5)

    # Test points
    test_cases = [
        (0.5, 0.5),  # Center point
        (0.3, 0.7),  # Asymmetric point
        (0.7, 0.3),  # Asymmetric point
    ]

    for u, v in test_cases:
        # Get conditional distribution
        cond2 = float(copula.cond_distr_2(u=u, v=v))

        # Calculate expected value for conditional distribution
        expected = u + 0.5 * u * (1 - u) * (1 - 2 * v)

        assert abs(cond2 - expected) < 1e-10, f"cond_distr_2 incorrect for u={u}, v={v}"

        # Conditional distribution should be between 0 and 1
        assert 0 <= cond2 <= 1, f"cond_distr_2({u},{v}) = {cond2} not in [0,1]"

        # Additional property: derivative of C(u,v) w.r.t. v should equal cond_distr_2
        # This is a numerical approximation
        epsilon = 1e-6
        C_v = float(copula.cdf(u=u, v=v))
        C_v_plus_eps = float(copula.cdf(u=u, v=v + epsilon))
        numerical_derivative = (C_v_plus_eps - C_v) / epsilon

        # Allow for numerical error in the approximation
        assert abs(numerical_derivative - cond2) < 1e-5, (
            f"Numerical derivative ({numerical_derivative}) too far from cond_distr_2 ({cond2}) at u={u}, v={v}"
        )


def test_spearmans_rho():
    """Test Spearman's rho calculation."""
    # Test with different theta values
    test_cases = [(-1, -1 / 3), (-0.5, -1 / 6), (0, 0), (0.5, 1 / 6), (1, 1 / 3)]

    for theta, expected in test_cases:
        copula = FarlieGumbelMorgenstern(theta=theta)
        rho = float(copula.spearmans_rho())
        assert abs(rho - expected) < 1e-10, (
            f"Spearman's rho incorrect for theta={theta}"
        )


def test_kendalls_tau():
    """Test Kendall's tau calculation."""
    # Test with different theta values
    test_cases = [(-1, -2 / 9), (-0.5, -1 / 9), (0, 0), (0.5, 1 / 9), (1, 2 / 9)]

    for theta, expected in test_cases:
        copula = FarlieGumbelMorgenstern(theta=theta)
        tau = float(copula.kendalls_tau())
        assert abs(tau - expected) < 1e-10, f"Kendall's tau incorrect for theta={theta}"


def test_tail_dependence():
    """Test that FGM has no tail dependence."""
    # FGM copula doesn't have tail dependence
    # We don't have these methods explicitly but can test numerically

    copula = FarlieGumbelMorgenstern(theta=1)  # Maximum dependence

    # Calculate approximate upper tail dependence
    u = v = 0.999  # Very close to 1
    cdf_val = float(copula.cdf(u=u, v=v))
    # λU = lim_{u→1} (1-2u+C(u,u))/(1-u) ≈ 0
    upper_tail = (1 - 2 * u + cdf_val) / (1 - u)
    assert abs(upper_tail) < 0.1, "FGM should have negligible upper tail dependence"

    # Calculate approximate lower tail dependence
    u = v = 0.001  # Very close to 0
    cdf_val = float(copula.cdf(u=u, v=v))
    # λL = lim_{u→0} C(u,u)/u ≈ 0
    lower_tail = cdf_val / u
    assert abs(lower_tail - u) < 0.1, "FGM should have negligible lower tail dependence"


def test_dependence_range():
    """Test that FGM has limited dependence range."""
    # Spearman's rho is bounded by -1/3 and 1/3
    # Kendall's tau is bounded by -2/9 and 2/9

    min_copula = FarlieGumbelMorgenstern(theta=-1)
    max_copula = FarlieGumbelMorgenstern(theta=1)

    min_rho = float(min_copula.spearmans_rho())
    max_rho = float(max_copula.spearmans_rho())

    min_tau = float(min_copula.kendalls_tau())
    max_tau = float(max_copula.kendalls_tau())

    # Check bounds for Spearman's rho
    assert abs(min_rho - (-1 / 3)) < 1e-10, "Minimum Spearman's rho should be -1/3"
    assert abs(max_rho - (1 / 3)) < 1e-10, "Maximum Spearman's rho should be 1/3"

    # Check bounds for Kendall's tau
    assert abs(min_tau - (-2 / 9)) < 1e-10, "Minimum Kendall's tau should be -2/9"
    assert abs(max_tau - (2 / 9)) < 1e-10, "Maximum Kendall's tau should be 2/9"
