import numpy as np
import pytest

from copul.family.other.frechet import Frechet
from copul.family.other.lower_frechet import LowerFrechet


@pytest.fixture
def copula():
    """Create a LowerFrechet copula instance for tests."""
    return LowerFrechet()


def test_initialization():
    """Test that the LowerFrechet copula initializes correctly."""
    # Create an instance
    copula = LowerFrechet()

    # Check that alpha and beta are fixed correctly
    assert float(copula.alpha) == 0.0
    assert float(copula.beta) == 1.0

    # Check that initialization parameters are ignored
    copula_with_params = LowerFrechet(alpha=0.5, beta=0.3)
    assert float(copula_with_params.alpha) == 0.0
    assert float(copula_with_params.beta) == 1.0


def test_inheritance():
    """Test that LowerFrechet properly inherits from Frechet."""
    copula = LowerFrechet()
    assert isinstance(copula, Frechet)


def test_is_symmetric(copula):
    """Test that the LowerFrechet copula is symmetric."""
    assert copula.is_symmetric is True


def test_is_absolutely_continuous(copula):
    """Test the absolutely continuous property."""
    # LowerFrechet with alpha=0, beta=1 is not absolutely continuous
    assert copula.is_absolutely_continuous is False


def test_cdf_values(copula):
    """Test specific CDF values."""
    # For LowerFrechet, C(u,v) = max(u+v-1, 0)
    test_cases = [
        (0.5, 0.5, 0.0),  # Symmetric point: 0.5+0.5-1 = 0
        (0.3, 0.7, 0.0),  # u < v: 0.3+0.7-1 = 0
        (0.7, 0.3, 0.0),  # u > v: 0.7+0.3-1 = 0
        (0.7, 0.6, 0.3),  # u > v, u+v > 1: 0.7+0.6-1 = 0.3
        (0.6, 0.7, 0.3),  # u < v, u+v > 1: 0.6+0.7-1 = 0.3
        (0.0, 0.5, 0.0),  # Boundary u=0
        (0.5, 0.0, 0.0),  # Boundary v=0
        (1.0, 0.5, 0.5),  # Boundary u=1
        (0.5, 1.0, 0.5),  # Boundary v=1
        (1.0, 1.0, 1.0),  # Boundary u=v=1
    ]

    for u, v, expected in test_cases:
        cdf_val = float(copula.cdf(u=u, v=v))
        assert abs(cdf_val - expected) < 1e-10, (
            f"CDF value incorrect for u={u}, v={v}, got {cdf_val}, expected {expected}"
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


def test_conditional_distribution():
    """Test that conditional distribution is properly defined."""
    copula = LowerFrechet()

    # Test points
    test_cases = [
        (0.5, 0.7),  # u < v, u+v > 1
        (0.7, 0.5),  # u > v, u+v > 1
        (0.3, 0.4),  # u < v, u+v < 1
        (0.4, 0.3),  # u > v, u+v < 1
        (0.5, 0.5),  # u = v, u+v = 1
    ]

    for u, v in test_cases:
        # Get conditional distribution
        cond2 = float(copula.cond_distr_2(u=u, v=v))

        # For LowerFrechet:
        # - When u+v > 1: cond_distr_2 should be 1 (since C(u,v) = u+v-1, fully dependent on v)
        # - When u+v < 1: cond_distr_2 should be 0 (since C(u,v) = 0, independent of v)
        # - When u+v = 1: can be ambiguous due to non-differentiability

        if u + v > 1:
            expected = 1
        elif u + v < 1:
            expected = 0
        else:
            # At u+v=1, the derivative is not well-defined
            # Skip the exact comparison but ensure it's in [0,1]
            assert 0 <= cond2 <= 1, f"cond_distr_2({u},{v}) = {cond2} not in [0,1]"
            continue

        assert abs(cond2 - expected) < 1e-10, f"cond_distr_2 incorrect for u={u}, v={v}"


def test_pdf_not_available(copula):
    """Test that PDF is not available for LowerFrechet copula."""
    from copul.exceptions import PropertyUnavailableException

    with pytest.raises(PropertyUnavailableException):
        copula.pdf


def test_rho(copula):
    """Test Spearman's rho calculation."""
    # For LowerFrechet, rho = -1
    rho = float(copula.spearmans_rho())
    assert abs(rho - (-1)) < 1e-10


def test_tau(copula):
    """Test Kendall's tau calculation."""
    # For LowerFrechet, tau = -1/3
    tau = float(copula.kendalls_tau())
    expected = -1
    assert abs(tau - expected) < 1e-10, (
        f"Kendall's tau incorrect: got {tau}, expected {expected}"
    )


def test_tail_dependence(copula):
    """Test tail dependence coefficients."""
    # Upper tail dependence should be 0
    lambda_U = float(copula.lambda_U)
    assert abs(lambda_U - 0) < 1e-10

    # Lower tail dependence should be 0
    lambda_L = float(copula.lambda_L)
    assert abs(lambda_L - 0) < 1e-10


def test_xi(copula):
    """Test Chatterjee's xi calculation."""
    # For LowerFrechet, xi = 1
    xi = float(copula.chatterjees_xi())
    assert abs(xi - 1) < 1e-10, f"Chatterjee's xi incorrect: got {xi}, expected 1"


def test_immutable_parameters():
    """Test that alpha and beta cannot be changed."""
    copula = LowerFrechet()

    # Try to change parameters using __call__
    new_copula = copula(alpha=0.5, beta=0.2)

    # Parameters should remain fixed
    assert float(new_copula.alpha) == 0.0
    assert float(new_copula.beta) == 1.0
