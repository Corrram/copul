import numpy as np

from copul.families.archimedean.nelsen5 import Frank
from copul.families.other.biv_independence_copula import BivIndependenceCopula


def test_frank_special_cases_create():
    """Test that Frank.create correctly handles special cases."""
    # Test regular case
    regular = Frank.create(2)
    assert isinstance(regular, Frank)
    assert regular.theta == 2

    # Test special case: theta = 0 should return IndependenceCopula
    independence = Frank.create(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test with keyword arguments
    kwargs_regular = Frank.create(theta=2)
    assert isinstance(kwargs_regular, Frank)
    assert kwargs_regular.theta == 2

    kwargs_special = Frank.create(theta=0)
    assert isinstance(kwargs_special, BivIndependenceCopula)


def test_frank_special_cases_new():
    """Test that Frank.__new__ correctly handles special cases."""
    # Test regular case
    regular = Frank(2)
    assert isinstance(regular, Frank)
    assert regular.theta == 2

    # Test special case: theta = 0 should return IndependenceCopula
    independence = Frank(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test with keyword arguments
    kwargs_regular = Frank(theta=2)
    assert isinstance(kwargs_regular, Frank)
    assert kwargs_regular.theta == 2

    kwargs_special = Frank(theta=0)
    assert isinstance(kwargs_special, BivIndependenceCopula)


def test_frank_special_cases_call():
    """Test that Frank.__call__ correctly handles special cases."""
    # Create a regular instance
    copula = Frank(2)

    # Test __call__ with regular parameter
    regular = copula(3)
    assert isinstance(regular, Frank)
    assert regular.theta == 3

    # Test __call__ with special case parameter
    independence = copula(0)
    assert isinstance(independence, BivIndependenceCopula)

    # Test with keyword arguments
    kwargs_regular = copula(theta=3)
    assert isinstance(kwargs_regular, Frank)
    assert kwargs_regular.theta == 3

    kwargs_special = copula(theta=0)
    assert isinstance(kwargs_special, BivIndependenceCopula)


def test_frank_generator():
    """Test Frank copula generator function."""
    # Test with a positive theta
    frank_pos = Frank(2)
    # Generator at specific values
    t_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in t_values:
        # Calculate expected value using the formula
        expected = -np.log((np.exp(-2 * t) - 1) / (np.exp(-2) - 1))
        actual = float(frank_pos.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test with a negative theta
    frank_neg = Frank(-1)
    for t in t_values:
        # Calculate expected value using the formula
        expected = -np.log((np.exp(t) - 1) / (np.exp(1) - 1))
        actual = float(frank_neg.generator(t))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_frank_cdf():
    """Test Frank copula cumulative distribution function."""
    # Test with a positive theta
    frank_pos = Frank(2)

    # Test CDF at some specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]
    for u, v in points:
        # Calculate expected value using the formula
        theta = 2
        expected = (
            -1
            / theta
            * np.log(
                1
                + (np.exp(-theta * u) - 1)
                * (np.exp(-theta * v) - 1)
                / (np.exp(-theta) - 1)
            )
        )
        actual = float(frank_pos.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)

    # Test with a negative theta
    frank_neg = Frank(-1)
    for u, v in points:
        # Calculate expected value using the formula
        theta = -1
        expected = (
            -1
            / theta
            * np.log(
                1
                + (np.exp(-theta * u) - 1)
                * (np.exp(-theta * v) - 1)
                / (np.exp(-theta) - 1)
            )
        )
        actual = float(frank_neg.cdf(u, v))
        assert np.isclose(actual, expected, rtol=1e-5)


def test_frank_boundary_conditions():
    """Test Frank copula boundary conditions."""
    # Create a Frank copula with theta = 3
    copula = Frank(3)

    # Test C(u,0) = 0 for any u in [0,1]
    u_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    for u in u_values:
        assert np.isclose(float(copula.cdf(u, 0)), 0, atol=1e-10)
        assert np.isclose(float(copula.cdf(0, u)), 0, atol=1e-10)

    # Test C(u,1) = u and C(1,v) = v for any u,v in [0,1]
    for u in u_values:
        assert np.isclose(float(copula.cdf(u, 1)), u, rtol=1e-5)
        assert np.isclose(float(copula.cdf(1, u)), u, rtol=1e-5)


def test_frank_parameter_validation():
    """Test parameter validation for Frank copula."""
    # Frank allows any real theta except 0
    # Valid theta values
    valid_thetas = [-5, -1, 0.1, 1, 5]
    for theta in valid_thetas:
        Frank(theta)  # Should not raise error

    # Frank's special handling should prevent error for theta=0
    # and return IndependenceCopula instead
    independence = Frank(0)
    assert isinstance(independence, BivIndependenceCopula)


def test_frank_tail_dependence():
    """Test tail dependence coefficients for Frank copula."""
    # Frank copula has no tail dependence
    copula = Frank(3)

    # Lambda_L (lower tail dependence) should be 0
    assert np.isclose(float(copula.lambda_L()), 0, atol=1e-10)

    # Lambda_U (upper tail dependence) should be 0
    assert np.isclose(float(copula.lambda_U()), 0, atol=1e-10)


def test_frank_conditional_distributions():
    """Test conditional distributions of Frank copula."""
    # Create a Frank copula with theta = 2
    copula = Frank(2)

    # Test conditional distribution at specific points
    points = [(0.3, 0.4), (0.5, 0.5), (0.7, 0.2), (0.9, 0.8)]

    for u, v in points:
        # Test conditional distribution C_1(v|u)
        cond_dist = float(copula.cond_distr_1(u, v))

        # Manually calculate expected value
        theta = 2
        expr_u = np.exp(-theta * u)
        expr_v = np.exp(-theta * v) - 1
        expr = np.exp(-theta) - 1
        expected = expr_v * expr_u / (expr + (-1 + expr_u) * expr_v)

        assert np.isclose(cond_dist, expected, rtol=1e-5)


def test_frank_kendalls_tau():
    """Test Kendall's tau for Frank copula with both positive and negative theta."""
    # Known values for positive theta
    pos_thetas = [1, 2, 5, 10]
    for theta in pos_thetas:
        copula = Frank(theta)
        tau = float(copula.kendalls_tau())

        # For Frank copula with positive theta, tau should be positive
        assert tau > 0
        assert tau < 1  # Should be less than 1

        # With larger theta, tau should increase
        # Check the trend only
        if theta > 1:
            prev_copula = Frank(theta - 1)
            prev_tau = float(prev_copula.kendalls_tau())
            assert tau > prev_tau


def test_frank_negative_kendalls_tau():
    # Known values for negative theta
    neg_thetas = [-1, -2, -5, -10]
    for theta in neg_thetas:
        copula = Frank(theta)
        tau = float(copula.kendalls_tau())

        # For Frank copula with negative theta, tau should be negative
        assert tau < 0
        assert tau > -1  # Should be greater than -1

        # With smaller (more negative) theta, tau should decrease
        # Check the trend only
        if theta < -1:
            prev_copula = Frank(theta + 1)
            prev_tau = float(prev_copula.kendalls_tau())
            assert tau < prev_tau

    # For theta = 0 (independence), tau should be 0
    independence = Frank.create(0)
    assert np.isclose(float(independence.kendalls_tau()), 0, atol=1e-10)


def test_frank_spearmans_rho():
    """Test Spearman's rho for Frank copula with both positive and negative theta."""
    # Known values for positive theta
    pos_thetas = [1, 2, 5, 10]
    for theta in pos_thetas:
        copula = Frank(theta)
        rho = float(copula.spearmans_rho())

        # For Frank copula with positive theta, rho should be positive
        assert rho > 0
        assert rho < 1  # Should be less than 1

        # With larger theta, rho should increase
        # Check the trend only
        if theta > 1:
            prev_copula = Frank(theta - 1)
            prev_rho = float(prev_copula.spearmans_rho())
            assert rho > prev_rho


def test_frank_negative_spearmans_rho():
    # Known values for negative theta
    neg_thetas = [-1, -2, -5, -10]
    for theta in neg_thetas:
        copula = Frank(theta)
        rho = float(copula.spearmans_rho())

        # For Frank copula with negative theta, rho should be negative
        assert rho < 0
        assert rho > -1  # Should be greater than -1

        # With smaller (more negative) theta, rho should decrease
        # Check the trend only
        if theta < -1:
            prev_copula = Frank(theta + 1)
            prev_rho = float(prev_copula.spearmans_rho())
            assert rho < prev_rho

    # For theta = 0 (independence), rho should be 0
    independence = Frank.create(0)
    assert np.isclose(float(independence.spearmans_rho()), 0, atol=1e-10)


def test_debye_functions():
    """Test the Debye function implementations."""
    # Test _d_1 and _d_2 for a positive theta
    copula = Frank(2)
    d1 = copula._d_1()
    d2 = copula._d_2()

    # d1 and d2 should be real numbers between 0 and 1
    assert 0 < d1 < 1
    assert 0 < d2 < 1

    # For positive theta, d1 > d2
    assert d1 > d2

    # Test for a negative theta
    neg_copula = Frank(-2)
    neg_d1 = neg_copula._d_1()
    neg_d2 = neg_copula._d_2()

    # Should also be real numbers between 0 and 1
    assert 0 < neg_d1 < 1
    assert 0 < neg_d2 < 1

    # d1 should still be greater than d2
    assert neg_d1 > neg_d2
