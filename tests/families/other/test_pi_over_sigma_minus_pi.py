import numpy as np
import pytest

from copul.families.other.pi_over_sigma_minus_pi import PiOverSigmaMinusPi


def test_theta_value():
    """Test that theta value is fixed at 1."""
    assert PiOverSigmaMinusPi().theta == 1


@pytest.mark.instable
def test_generator_properties():
    """Test that the generator and inverse generator are properly defined."""
    # Create test values
    t_vals = np.linspace(0.1, 0.9, 5)
    y_vals = np.linspace(0.1, 5.0, 5)

    # Get the generator function
    copula = PiOverSigmaMinusPi()
    gen = copula.generator
    inv_gen = copula.inv_generator.func

    # Test generator at specific points
    for t in t_vals:
        # Expected: (1/t - 1)
        expected_gen = (1 / t) - 1
        gen_result = float(gen.subs(copula.t, t))
        assert abs(gen_result - expected_gen) < 1e-10, f"Generator wrong at t={t}"

    # Test inverse generator at specific points
    for y in y_vals:
        # Expected: 1/(1+y)
        expected_inv_gen = 1 / (1 + y)
        inv_gen_result = float(inv_gen.subs(copula.y, y))
        assert abs(inv_gen_result - expected_inv_gen) < 1e-10, (
            f"Inverse generator wrong at y={y}"
        )


def test_cdf_values():
    """Test specific CDF values."""
    copula = PiOverSigmaMinusPi()
    test_cases = [
        (
            0.5,
            0.5,
            1 / 3,
        ),  # Symmetric point: 0.5*0.5/(0.5+0.5-0.5*0.5) = 0.25/0.75 = 1/3
        (0.3, 0.7, 0.3 * 0.7 / (0.3 + 0.7 - 0.3 * 0.7)),  # Asymmetric point
        (0.0, 0.5, 0.0),  # Boundary u=0
        (0.5, 0.0, 0.0),  # Boundary v=0
        (1.0, 0.5, 0.5),  # Boundary u=1
        (0.5, 1.0, 0.5),  # Boundary v=1
        (1.0, 1.0, 1.0),  # Boundary u=v=1
    ]

    for u, v, expected in test_cases:
        cdf_val = float(copula.cdf(u=u, v=v))
        assert abs(cdf_val - expected) < 1e-10, f"CDF value incorrect for u={u}, v={v}"


def test_pdf_values():
    """Test specific PDF values."""
    copula = PiOverSigmaMinusPi()
    test_cases = [
        (
            0.5,
            0.5,
            3.56,
        ),  # Exact value at (0.5, 0.5) = 2*(0.5+0.5-0.5*0.5)/(0.5+0.5-0.5*0.5)^3 = 2*0.75/0.75^3
        (
            0.3,
            0.7,
            3.27,
        ),  # Exact value at (0.3, 0.7) = 2*(0.3+0.7-0.3*0.7)/(0.3+0.7-0.3*0.7)^3
    ]

    for u, v, expected in test_cases:
        pdf_val = float(copula.pdf(u=u, v=v))
        # Calculate the exact expected value to compare with
        denom = (u + v - u * v) ** 3
        numer = 2 * (u + v - u * v)
        expected_exact = numer / denom

        # Compare with our calculated value
        assert abs(pdf_val - expected_exact) < 1e-10, (
            f"PDF value incorrect at u={u}, v={v}"
        )
        # Also check that our expected test value is approximately correct
        assert abs(expected_exact - expected) < 0.1, (
            f"Expected test value is off at u={u}, v={v}"
        )


def test_conditional_distributions():
    """Test that conditional distributions are properly defined."""
    copula = PiOverSigmaMinusPi()
    # Test points
    u, v = 0.5, 0.6

    # Expected values
    expected_cond1 = v / (u + v - u * v) ** 2  # ∂C(u,v)/∂u at (0.5, 0.6)
    expected_cond2 = u / (u + v - u * v) ** 2  # ∂C(u,v)/∂v at (0.5, 0.6)

    # Get conditional distributions
    cond1 = float(copula.cond_distr_1(u=u, v=v))
    cond2 = float(copula.cond_distr_2(u=u, v=v))

    # Assert they're equal to expected values
    assert abs(cond1 - expected_cond1) < 1e-10, f"cond_distr_1({u},{v}) incorrect"
    assert abs(cond2 - expected_cond2) < 1e-10, f"cond_distr_2({u},{v}) incorrect"

    # Conditional distributions should be between 0 and 1
    assert 0 <= cond1 <= 1, f"cond_distr_1({u},{v}) = {cond1} not in [0,1]"
    assert 0 <= cond2 <= 1, f"cond_distr_2({u},{v}) = {cond2} not in [0,1]"


def test_boundary_cases():
    """Test that the copula behaves correctly at boundary values."""
    copula = PiOverSigmaMinusPi()
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


def test_tail_dependence():
    """Test the tail dependence coefficients."""
    copula = PiOverSigmaMinusPi()
    # Lower tail dependence should be 0.5
    lambda_L = copula.lambda_L()
    assert abs(lambda_L - 0.5) < 1e-10, "Lower tail dependence incorrect"

    # Upper tail dependence should be 0
    lambda_U = copula.lambda_U()
    assert lambda_U == 0, "Upper tail dependence should be 0"


def test_is_absolutely_continuous():
    """Test that the copula is absolutely continuous."""
    copula = PiOverSigmaMinusPi()
    assert copula.is_absolutely_continuous is True


def test_tau():
    """Test Kendall's tau value."""
    copula = PiOverSigmaMinusPi()
    tau = copula.tau()
    assert abs(tau - 1 / 3) < 1e-10, "Kendall's tau should be 1/3"
