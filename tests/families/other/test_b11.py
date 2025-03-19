import numpy as np
import pytest

from copul.families.core.biv_copula import BivCopula
from copul.families.other.b11 import B11


@pytest.fixture
def copula():
    """Create a B11 copula instance with delta=0.5 for tests."""
    return B11(delta=0.5)


def test_initialization():
    """Test that the B11 copula can be initialized with different parameters."""
    # Default initialization
    b11 = B11()
    assert str(b11.delta) == "delta"

    # With parameter
    b11 = B11(delta=0.5)
    assert float(b11.delta) == 0.5

    # Edge case parameters
    b11_min = B11(delta=0)
    assert float(b11_min.delta) == 0

    b11_max = B11(delta=1)
    assert float(b11_max.delta) == 1


def test_parameter_constraints():
    """Test parameter constraints: 0 ≤ delta ≤ 1."""
    # Valid parameters
    B11(delta=0)  # Lower bound
    B11(delta=0.5)  # Middle
    B11(delta=1)  # Upper bound

    # Invalid parameters (delta < 0 or delta > 1)
    with pytest.raises(ValueError):
        B11(delta=-0.1)

    with pytest.raises(ValueError):
        B11(delta=1.1)


def test_special_cases():
    """Test special cases of the B11 copula."""
    # When delta = 0, should return IndependenceCopula
    independence = B11(delta=0)
    test_points = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

    for u, v in test_points:
        # For independence, C(u,v) = u*v
        cdf_val = float(independence.cdf(u=u, v=v))
        expected = u * v
        assert abs(cdf_val - expected) < 1e-10, (
            f"CDF incorrect for independence case at u={u}, v={v}"
        )

    # When delta = 1, should return UpperFrechet
    upper_frechet = B11(delta=1)

    for u, v in test_points:
        # For upper Frechet, C(u,v) = min(u,v)
        cdf_val = float(upper_frechet.cdf(u=u, v=v))
        expected = min(u, v)
        assert abs(cdf_val - expected) < 1e-10, (
            f"CDF incorrect for upper Frechet case at u={u}, v={v}"
        )


def test_inheritance():
    """Test that B11 properly inherits from BivCopula."""
    copula = B11()
    assert isinstance(copula, BivCopula)


def test_is_symmetric(copula):
    """Test that the B11 copula is symmetric."""
    assert copula.is_symmetric is True


def test_is_absolutely_continuous():
    """Test the absolutely continuous property."""
    # Should be absolutely continuous for delta < 1
    copula1 = B11(delta=0)
    assert copula1.is_absolutely_continuous is True

    copula2 = B11(delta=0.5)
    assert copula2.is_absolutely_continuous is True

    # Should not be absolutely continuous for delta = 1
    copula3 = B11(delta=1)
    assert copula3.is_absolutely_continuous is False


def test_cdf_values(copula):
    """Test specific CDF values."""
    delta = 0.5

    def b11_cdf(u, v, delta):
        """Compute the B11 copula CDF value."""
        return delta * min(u, v) + (1 - delta) * u * v

    test_cases = [
        (0.5, 0.5, b11_cdf(0.5, 0.5, delta)),  # Symmetric point
        (0.3, 0.7, b11_cdf(0.3, 0.7, delta)),  # u < v
        (0.7, 0.3, b11_cdf(0.7, 0.3, delta)),  # u > v
        (0.0, 0.5, 0.0),  # Boundary u=0
        (0.5, 0.0, 0.0),  # Boundary v=0
        (1.0, 0.5, 0.5),  # Boundary u=1
        (0.5, 1.0, 0.5),  # Boundary v=1
        (1.0, 1.0, 1.0),  # Boundary u=v=1
    ]

    for u, v, expected in test_cases:
        cdf_val = float(copula.cdf(u=u, v=v))
        assert abs(cdf_val - expected) < 1e-10, f"CDF value incorrect for u={u}, v={v}"


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


def test_dependence_measures():
    """Test various dependence measures."""
    # For B11, we can calculate some measures analytically
    test_cases = [
        (0, 0),  # Independence
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.75),
        (1, 1),  # Perfect positive dependence
    ]

    for delta, expected_measure in test_cases:
        copula = B11(delta=delta)

        # B11 with parameter delta has Spearman's rho = delta
        # This is because it's a convex combination with weight delta
        if hasattr(copula, "spearmans_rho"):
            rho = float(copula.rho())
            assert abs(rho - expected_measure) < 1e-10, (
                f"Spearman's rho incorrect for delta={delta}"
            )

        # Similarly, Kendall's tau = delta/3 * (3 - 2*delta)
        # (this is a theoretical result for this family)
        if hasattr(copula, "kendalls_tau"):
            tau = float(copula.tau())
            expected_tau = delta / 3 * (3 - 2 * delta)
            assert abs(tau - expected_tau) < 1e-10, (
                f"Kendall's tau incorrect for delta={delta}"
            )

        # Upper tail dependence = delta (only at corners)
        if hasattr(copula, "lambda_U"):
            lambda_U = float(copula.lambda_U)
            assert abs(lambda_U - (delta if delta == 1 else 0)) < 1e-10, (
                f"Upper tail dependence incorrect for delta={delta}"
            )

        # Lower tail dependence = delta (only at corners)
        if hasattr(copula, "lambda_L"):
            lambda_L = float(copula.lambda_L)
            assert abs(lambda_L - (delta if delta == 1 else 0)) < 1e-10, (
                f"Lower tail dependence incorrect for delta={delta}"
            )
