import numpy as np
import pytest

from copul.family.frechet.frechet import Frechet
from copul.family.frechet.upper_frechet import UpperFrechet


@pytest.fixture
def copula():
    """Create an UpperFrechet copula instance for tests."""
    return UpperFrechet()


def test_initialization():
    """Test that the UpperFrechet copula initializes correctly."""
    # Create an instance
    copula = UpperFrechet()

    # Check that alpha and beta are fixed correctly
    assert float(copula.alpha) == 1.0
    assert float(copula.beta) == 0.0

    # Check that initialization parameters are ignored
    copula_with_params = UpperFrechet(alpha=0.5, beta=0.3)
    assert float(copula_with_params.alpha) == 1.0
    assert float(copula_with_params.beta) == 0.0


def test_inheritance():
    """Test that UpperFrechet properly inherits from Frechet."""
    copula = UpperFrechet()
    assert isinstance(copula, Frechet)


def test_is_symmetric(copula):
    """Test that the UpperFrechet copula is symmetric."""
    assert copula.is_symmetric is True


def test_is_absolutely_continuous(copula):
    """Test the absolutely continuous property."""
    # UpperFrechet with alpha=1, beta=0 is not absolutely continuous
    assert copula.is_absolutely_continuous is False


def test_cdf_values(copula):
    """Test specific CDF values."""
    # For UpperFrechet, C(u,v) = min(u,v)
    test_cases = [
        (0.5, 0.5, 0.5),  # Symmetric point
        (0.3, 0.7, 0.3),  # u < v
        (0.7, 0.3, 0.3),  # u > v
        (0.7, 0.6, 0.6),  # u > v
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


def test_conditional_distribution():
    """Test that conditional distribution is properly defined."""
    copula = UpperFrechet()

    # Test points
    test_cases = [
        (0.5, 0.7),  # u < v
        (0.7, 0.5),  # u > v
        (0.6, 0.6),  # u = v
    ]

    for u, v in test_cases:
        # Get conditional distribution
        cond2 = float(copula.cond_distr_2(u=u, v=v))

        # For UpperFrechet:
        # - When u < v: cond_distr_2 should be 0 (since C(u,v) = u, independent of v)
        # - When u > v: cond_distr_2 should be 1 (since C(u,v) = v, fully dependent on v)
        # - When u = v: can be ambiguous due to non-differentiability

        if u < v:
            expected = 0
        elif u > v:
            expected = 1
        else:
            # At u=v, the derivative is not well-defined
            # Skip the exact comparison but ensure it's in [0,1]
            assert 0 <= cond2 <= 1, f"cond_distr_2({u},{v}) = {cond2} not in [0,1]"
            continue

        assert abs(cond2 - expected) < 1e-10, f"cond_distr_2 incorrect for u={u}, v={v}"


def test_pdf_not_available(copula):
    """Test that PDF is not available for UpperFrechet copula."""
    from copul.exceptions import PropertyUnavailableException

    with pytest.raises(PropertyUnavailableException):
        copula.pdf


def test_rho(copula):
    """Test Spearman's rho calculation."""
    # For UpperFrechet, rho = 1
    rho = float(copula.spearmans_rho())
    assert abs(rho - 1) < 1e-10


def test_tau(copula):
    """Test Kendall's tau calculation."""
    # For UpperFrechet, tau = 1
    tau = float(copula.kendalls_tau())
    assert abs(tau - 1) < 1e-10


def test_tail_dependence(copula):
    """Test tail dependence coefficients."""
    # Upper tail dependence should be 1
    lambda_U = float(copula.lambda_U)
    assert abs(lambda_U - 1) < 1e-10

    # Lower tail dependence should be 1
    lambda_L = float(copula.lambda_L)
    assert abs(lambda_L - 1) < 1e-10


def test_xi(copula):
    """Test Chatterjee's xi calculation."""
    # For UpperFrechet, xi = 1
    xi = float(copula.chatterjees_xi())
    assert abs(xi - 1) < 1e-10


def test_pickands_function(copula):
    """Test the Pickands dependence function."""
    # Get symbolic function
    pickands_expr = copula.pickands

    # Test at specific t values
    t_vals = [0.0, 0.2, 0.5, 0.8, 1.0]

    for t in t_vals:
        pickands_val = float(pickands_expr.subs(copula.t, t))
        expected = max(t, 1 - t)
        assert abs(pickands_val - expected) < 1e-10, (
            f"Pickands function incorrect at t={t}"
        )


def test_immutable_parameters():
    """Test that alpha and beta cannot be changed."""
    copula = UpperFrechet()

    # Try to change parameters using __call__
    new_copula = copula(alpha=0.5, beta=0.2)

    # Parameters should remain fixed
    assert float(new_copula.alpha) == 1.0
    assert float(new_copula.beta) == 0.0
