import numpy as np
import pytest

from copul.exceptions import PropertyUnavailableException
from copul.families.bivcopula import BivCopula
from copul.families.other.mardia import Mardia


@pytest.fixture
def copula():
    """Create a Mardia copula instance with theta=0.5 for tests."""
    return Mardia(theta=0.5)


def test_initialization():
    """Test that the Mardia copula can be initialized with different parameters."""
    # Default initialization
    mardia = Mardia()
    assert str(mardia.theta) == "theta"

    # With parameter (positional)
    mardia = Mardia(0.5)
    assert float(mardia.theta) == 0.5

    # With parameter (keyword)
    mardia = Mardia(theta=0.7)
    assert float(mardia.theta) == 0.7

    # Edge case parameters
    mardia_min = Mardia(theta=-1)
    assert float(mardia_min.theta) == -1

    mardia_max = Mardia(theta=1)
    assert float(mardia_max.theta) == 1


def test_parameter_constraints():
    """Test parameter constraints: -1 ≤ theta ≤ 1."""
    # Valid parameters
    Mardia(theta=-1)  # Lower bound
    Mardia(theta=0)  # Middle
    Mardia(theta=1)  # Upper bound

    # Invalid parameters (theta < -1 or theta > 1)
    with pytest.raises(ValueError):
        Mardia(theta=-1.1)

    with pytest.raises(ValueError):
        Mardia(theta=1.1)


def test_call_method(copula):
    """Test the __call__ method to create new instances."""
    # Create a new instance with different theta
    new_copula = copula(theta=0.7)
    assert float(new_copula.theta) == 0.7
    assert float(copula.theta) == 0.5  # Original unchanged

    # Create a new instance with positional parameter
    new_copula = copula(0.3)
    assert float(new_copula.theta) == 0.3


def test_inheritance():
    """Test that Mardia properly inherits from BivCopula."""
    copula = Mardia()
    assert isinstance(copula, BivCopula)


def test_is_symmetric(copula):
    """Test that the Mardia copula is symmetric."""
    assert copula.is_symmetric is True


def test_is_absolutely_continuous():
    """Test the absolutely continuous property."""
    # Only absolutely continuous when theta=0 or theta=-1
    copula1 = Mardia(theta=0)
    assert copula1.is_absolutely_continuous is True

    copula2 = Mardia(theta=-1)
    assert copula2.is_absolutely_continuous is True

    # Not absolutely continuous otherwise
    copula3 = Mardia(theta=0.5)
    assert copula3.is_absolutely_continuous is False

    copula4 = Mardia(theta=1)
    assert copula4.is_absolutely_continuous is False


def test_cdf_values(copula):
    """Test specific CDF values."""
    theta = 0.5

    def mardia_cdf(u, v, theta):
        """Compute the Mardia copula CDF value."""
        theta_sq = theta**2
        upper = min(u, v)
        lower = max(u + v - 1, 0)
        return (
            theta_sq * (1 + theta) / 2 * upper
            + (1 - theta_sq) * u * v
            + theta_sq * (1 - theta) / 2 * lower
        )

    test_cases = [
        # Format: (u, v)
        (0.5, 0.5),  # Symmetric point
        (0.3, 0.7),  # Asymmetric point
        (0.7, 0.3),  # Asymmetric point
        (0.7, 0.6),  # u+v > 1
        (0.4, 0.3),  # u+v < 1
        (0.0, 0.5),  # Boundary u=0
        (0.5, 0.0),  # Boundary v=0
        (1.0, 0.5),  # Boundary u=1
        (0.5, 1.0),  # Boundary v=1
        (1.0, 1.0),  # Boundary u=v=1
    ]

    for u, v in test_cases:
        cdf_val = float(copula.cdf(u=u, v=v))
        expected = mardia_cdf(u, v, theta)

        assert abs(cdf_val - expected) < 1e-10, (
            f"CDF value incorrect for u={u}, v={v}: got {cdf_val}, expected {expected}"
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


def test_special_cases():
    """Test special cases of the Mardia copula."""
    # Independence case: theta = 0
    indep = Mardia(theta=0)
    u, v = 0.3, 0.7
    cdf_val = float(indep.cdf(u=u, v=v))
    assert abs(cdf_val - (u * v)) < 1e-10, (
        f"C({u},{v}) should be {u * v} for independence"
    )

    # Upper Fréchet bound: theta = 1
    upper = Mardia(theta=1)
    cdf_val = float(upper.cdf(u=u, v=v))
    assert abs(cdf_val - min(u, v)) < 1e-10, (
        f"C({u},{v}) should be {min(u, v)} for theta=1"
    )

    # Lower Fréchet bound: not achievable with Mardia in general
    # But for theta = -1, it's a mixture
    lower = Mardia(theta=-1)
    cdf_val = float(lower.cdf(u=u, v=v))
    expected = (u * v + max(u + v - 1, 0)) / 2
    assert abs(cdf_val - expected) < 1e-10, f"C({u},{v}) incorrect for theta=-1"


def test_pdf_not_available(copula):
    """Test that PDF is not available for Mardia copula."""
    with pytest.raises(PropertyUnavailableException):
        copula.pdf


def test_spearmans_rho():
    """Test Spearman's rho calculation."""
    # Test with different theta values
    test_cases = [(-1, -1), (-0.5, -0.125), (0, 0), (0.5, 0.125), (1, 1)]

    for theta, expected in test_cases:
        copula = Mardia(theta=theta)
        rho = float(copula.spearmans_rho())
        assert abs(rho - expected) < 1e-10, (
            f"Spearman's rho incorrect for theta={theta}"
        )


def test_kendalls_tau():
    """Test Kendall's tau calculation."""
    # Formula: theta^3 * (theta^2 + 2) / 3
    test_cases = [
        (-1, -1),
        (-0.5, -(0.5**3) * (0.5**2 + 2) / 3),
        (0, 0),
        (0.5, 0.5**3 * (0.5**2 + 2) / 3),
        (1, 1),
    ]

    for theta, expected in test_cases:
        copula = Mardia(theta=theta)
        tau = float(copula.kendalls_tau())
        assert abs(tau - expected) < 1e-10, f"Kendall's tau incorrect for theta={theta}"


def test_tail_dependence():
    """Test tail dependence coefficients."""
    # Upper and lower tail dependence should be theta^2 * (1 + theta) / 2
    test_cases = [
        (-1, 0),
        (-0.5, 0.5**2 * (1 - 0.5) / 2),
        (0, 0),
        (0.5, 0.5**2 * (1 + 0.5) / 2),
        (1, 1),
    ]

    for theta, expected in test_cases:
        copula = Mardia(theta=theta)

        lambda_U = float(copula.lambda_U)
        assert abs(lambda_U - expected) < 1e-10, (
            f"Upper tail dependence incorrect for theta={theta}"
        )

        lambda_L = float(copula.lambda_L)
        assert abs(lambda_L - expected) < 1e-10, (
            f"Lower tail dependence incorrect for theta={theta}"
        )


def test_chatterjees_xi():
    """Test Chatterjee's xi calculation."""
    # Formula: theta^4 * (3*theta^2 + 1) / 4
    test_cases = [
        (-1, 1),
        (-0.5, 0.5**4 * (3 * 0.5**2 + 1) / 4),
        (0, 0),
        (0.5, 0.5**4 * (3 * 0.5**2 + 1) / 4),
        (1, 1),
    ]

    for theta, expected in test_cases:
        copula = Mardia(theta=theta)
        xi = float(copula.chatterjees_xi())
        assert abs(xi - expected) < 1e-10, (
            f"Chatterjee's xi incorrect for theta={theta}"
        )
