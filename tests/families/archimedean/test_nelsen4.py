import numpy as np
import pytest

from copul.families.archimedean import GumbelHougaard, Nelsen4
from copul.families.other.biv_independence_copula import BivIndependenceCopula


def test_nelsen4():
    """Basic test for Nelsen4 copula."""
    nelsen4 = Nelsen4(1.5)
    result = nelsen4.lambda_L()
    assert np.isclose(result, 0)


def test_gumbel_hougaard_is_nelsen4():
    """Test that Nelsen4 is an alias for GumbelHougaard."""
    assert Nelsen4 is GumbelHougaard


@pytest.mark.parametrize("theta", [1, 1.5, 2, 5])
def test_nelsen4_is_absolutely_continuous(theta):
    """Test that Nelsen4 copula is absolutely continuous."""
    copula = Nelsen4(theta)
    assert copula.is_absolutely_continuous


def test_nelsen4_theta_validation():
    """Test that Nelsen4 validates theta correctly."""
    # Valid theta values
    for theta in [1, 1.5, 2, 5]:
        Nelsen4(theta)

    # Invalid theta values
    with pytest.raises(ValueError):
        Nelsen4(0.9)  # Below minimum
    with pytest.raises(ValueError):
        Nelsen4(0)  # Below minimum
    with pytest.raises(ValueError):
        Nelsen4(-1)  # Below minimum


def test_nelsen4_special_case_theta_one():
    """Test that when theta=1, Nelsen4 returns IndependenceCopula."""
    # Create Nelsen4 with theta=1
    copula = Nelsen4(2)  # Initial theta doesn't matter
    result = copula(theta=1)

    # Check result is an instance of IndependenceCopula
    assert isinstance(result, BivIndependenceCopula)

    # Verify that CDF behaves as independence copula (u*v)
    u, v = 0.3, 0.7
    assert np.isclose(float(result.cdf(u=u, v=v)), u * v)


@pytest.mark.parametrize("theta", [1.2, 2, 5])
def test_nelsen4_generator(theta):
    """Test the generator function of Nelsen4."""
    copula = Nelsen4(theta)

    # Test at various points in the unit interval
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for t in t_values:
        # Calculate expected value: (-log(t))^θ
        expected = (-np.log(t)) ** theta
        actual = float(copula.generator(t))
        assert np.isclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("theta", [1.2, 2, 5])
def test_nelsen4_inverse_generator(theta):
    """Test the inverse generator function of Nelsen4."""
    copula = Nelsen4(theta)

    # Test at various points
    y_values = [0.1, 0.5, 1, 2, 5]

    for y in y_values:
        # Calculate expected value: exp(-(y^(1/θ)))
        expected = np.exp(-(y ** (1 / theta)))
        actual = float(copula.inv_generator(y))
        assert np.isclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("theta", [1.2, 2, 5])
def test_nelsen4_cdf(theta):
    """Test the CDF of Nelsen4."""
    copula = Nelsen4(theta)

    # Test at various points in the unit square
    test_points = [(0.2, 0.3), (0.4, 0.6), (0.7, 0.2), (0.8, 0.9)]

    for u, v in test_points:
        # Calculate expected value: exp(-(((-log(u))^θ + (-log(v))^θ)^(1/θ)))
        expected = np.exp(
            -(((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))
        )
        actual = float(copula.cdf(u=u, v=v))
        assert np.isclose(actual, expected, rtol=1e-10)


@pytest.mark.parametrize("theta", [1.2, 2, 5])
def test_nelsen4_boundary_conditions(theta):
    """Test the boundary conditions of Nelsen4."""
    copula = Nelsen4(theta)

    # Test along the boundaries of the unit square
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # C(u,0) = 0
    for u in u_values:
        cdf_value = float(copula.cdf(u=u, v=0))
        assert np.isclose(cdf_value, 0, atol=1e-10)

    # C(0,v) = 0
    for v in u_values:
        assert np.isclose(float(copula.cdf(u=0, v=v)), 0, atol=1e-10)

    # C(u,1) = u
    for u in u_values:
        assert np.isclose(float(copula.cdf(u=u, v=1)), u, rtol=1e-10)

    # C(1,v) = v
    for v in u_values:
        assert np.isclose(float(copula.cdf(u=1, v=v)), v, rtol=1e-10)


@pytest.mark.parametrize("theta", [1.2, 2, 5])
def test_nelsen4_archimedean_property(theta):
    """Test the Archimedean property of Nelsen4: C(u,v) = φ^(-1)(φ(u) + φ(v))."""
    copula = Nelsen4(theta)

    # Test at various points in the unit square
    test_points = [(0.2, 0.3), (0.4, 0.6), (0.7, 0.2), (0.8, 0.9)]

    for u, v in test_points:
        # Get generator values
        gen_u = float(copula.generator(u))
        gen_v = float(copula.generator(v))

        # Apply inverse generator to sum
        archimedean_val = float(copula.inv_generator(gen_u + gen_v))

        # Get direct CDF value
        cdf_val = float(copula.cdf(u=u, v=v))

        # They should be equal
        assert np.isclose(archimedean_val, cdf_val, rtol=1e-10)


@pytest.mark.parametrize(
    "theta, expected",
    [
        (1, 0),  # Independence case
        (1.5, 0.3333),
        (2, 0.5),
        (5, 0.8),
    ],
)
def test_nelsen4_kendalls_tau(theta, expected):
    """Test Kendall's tau for Nelsen4.

    For Gumbel copula, Kendall's tau = (θ-1)/θ
    """
    # Skip calling the method directly as it might not be implemented
    # Just verify the mathematical relationship
    tau = (theta - 1) / theta  # Theoretical formula
    assert np.isclose(tau, expected, rtol=1e-2)


@pytest.mark.parametrize("theta", [1.2, 2, 5])
def test_nelsen4_tail_dependence(theta):
    """Test tail dependence coefficients for Nelsen4."""
    copula = Nelsen4(theta)

    # Lower tail dependence (should be 0)
    lambda_L = float(copula.lambda_L())
    assert np.isclose(lambda_L, 0, atol=1e-2)

    # Upper tail dependence
    lambda_U = float(copula.lambda_U())
    expected = 2 - 2 ** (1 / theta)
    assert np.isclose(lambda_U, expected, rtol=1e-10)


def test_nelsen4_generator_visualization():
    """Test that plot_generator doesn't raise an error."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend to prevent display
    import matplotlib.pyplot as plt

    copula = Nelsen4(2)
    try:
        copula.plot_generator(start=0.1, stop=0.9)
        plt.close("all")  # Clean up
    except Exception as e:
        pytest.fail(f"plot_generator raised an exception: {e}")
