import numpy as np
import pytest

from copul.checkerboard.bernstein import BernsteinCopula


def test_2d_mixed_degrees():
    """
    2D with different degrees, e.g. shape=(2,3) => m1=1, m2=2.
    We'll pick a simple valid theta for an 'independence-like' structure.
    """
    # shape=(2,3): first dimension has m1=1 => indexes {0,1}
    #               second dimension has m2=2 => indexes {0,1,2}
    # We'll create a simple 'corner-based' array:
    theta = np.zeros((2, 3))
    theta[1, 2] = 1.0  # put "mass" at top-right corner => akin to u^1 * v^2
    cop = BernsteinCopula(theta)
    assert cop.dim == 2
    assert cop.degrees == [1, 2]

    # cdf at point near boundary => 0
    assert cop.cdf([0.0, 0.0]) == 0.0
    # cdf at point near (1,1) => 1
    assert np.isclose(cop.cdf([1.0, 1.0]), 1.0)

    # random interior point
    u = [0.3, 0.7]
    c = cop.cdf(u)
    p = cop.pdf(u)
    assert 0.0 < c < 1.0, f"CDF should be between 0 and 1 for point {u}."
    assert p >= 0.0, f"PDF should be >=0 for point {u}."

    # check that sampling is not implemented
    with pytest.raises(NotImplementedError):
        cop.rvs(5)


def test_3d_different_degs():
    """
    3D example, shape=(1,2,3) => degrees [0,1,2].
    We'll fill in some dummy theta to ensure we can call cdf/pdf.
    """
    # shape=(1,2,3) => first dim m1=0, second dim m2=1, third dim m3=2
    theta = np.random.rand(1, 2, 3)
    # Force the 'corner' element to be largest to mimic a partial distribution
    theta[0, 1, 2] += 5.0

    cop = BernsteinCopula(theta)
    assert cop.dim == 3
    assert cop.degrees == [0, 1, 2]

    # Evaluate cdf/pdf on a few random points
    points = np.array([
        [0.1, 0.5, 0.5],
        [0.9, 0.2, 0.8],
        [1.0, 1.0, 1.0],  # boundary
        [0.0, 0.0, 0.0],  # boundary
    ])
    cvals = cop.cdf(points)
    pvals = cop.pdf(points)
    assert cvals.shape == (4,)
    assert pvals.shape == (4,)

    # boundary points => cdf=0 or 1
    assert np.isclose(cvals[-2], 1.0), "CDF at (1,1,1) ~ 1"
    assert np.isclose(cvals[-1], 0.0), "CDF at (0,0,0) ~ 0"


def test_invalid_shapes():
    """Check that we reject invalid shapes (e.g. zero dimension)."""
    with pytest.raises(ValueError):
        _ = BernsteinCopula(np.array([]))  # no dimension
    with pytest.raises(ValueError):
        _ = BernsteinCopula(np.zeros((0, 2)))  # shape=(0,2) => invalid


@pytest.mark.parametrize("point, expected", [
    ([1, 0.5], 0.5),
    ([0.5, 1], 0.5),
    ([0, 0], 0),
    ([1, 1], 1),
])
def test_cdf_edge_cases(point, expected):
    """Test edge cases for CDF."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cdf(point)
    assert np.isclose(actual, expected), f"CDF at {point} should be {expected}"

@pytest.mark.parametrize("point, expected", [
    ([.99, 0.5], 0.5),
    ([0.5, .99], 0.5),
    ([0.01, 0.01], 0),
    ([.99, .99], 1),
])
def test_cdf_edge_cases_rough(point, expected):
    """Test edge cases for CDF."""
    theta = np.ones((3, 3))  # shape=(2,2), m1=1, m2=1
    cop = BernsteinCopula(theta)
    actual = cop.cdf(point)
    assert np.isclose(actual, expected, atol=0.1), f"CDF at {point} should be {expected}"

def test_cdf_vectorized_edge_cases():
    """Test edge cases for CDF."""
    points = np.array([[1, 0.5], [0.5, 1], [0, 0], [1, 1]])
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cdf(points)
    expected = np.array([0.5, 0.5, 0, 1])
    assert np.all(np.isclose(actual, expected)), \
        f"CDF at {points} should be {expected}"
    
def test_cdf_vectorized_edge_cases_rough():
    """Test edge cases for CDF."""
    points = np.array([[.99, 0.5], [0.5, .99], [0.01, 0.01], [.99, .99]])
    theta = np.ones((3, 3))  # shape=(2,2), m1=1, m2=1
    cop = BernsteinCopula(theta)
    actual = cop.cdf(points)
    expected = np.array([0.5, 0.5, 0, 1])
    assert np.all(np.isclose(actual, expected, atol=0.1)), \
        f"CDF at {points} should be {expected}"

@pytest.mark.parametrize("point, expected", [
    ([0, 0.5], 0),       # P(U₁≤0|U₂=0.5) = 0
    ([1, 0.5], 1),       # P(U₁≤1|U₂=0.5) = 1
])
def test_cond_distr_1_edge_cases(point, expected):
    """Test edge cases for first conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_1(point)
    assert np.isclose(actual, expected), f"cond_distr_1 at {point} should be {expected}"


@pytest.mark.parametrize("point, expected", [
    ([0, 0.5], 0),    # Very small u₁
    ([1, 0.5], 0.5),  # Nearly 1 for u₁
])
def test_cond_distr_2_edge_cases(point, expected):
    """Test edge cases for second conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_2(point)
    assert np.isclose(actual, expected), f"cond_distr_2 at {point} should be {expected}"


@pytest.mark.parametrize("point, expected", [
    ([0.01, 0.5], 0),    # Very small u₁
    ([0.99, 0.5], 1),    # Nearly 1 for u₁
])
def test_cond_distr_1_edge_cases_approx(point, expected):
    """Test approximate edge cases for first conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_1(point)
    assert np.isclose(actual, expected, rtol=0.01), f"cond_distr_1 at {point} should be approximately {expected}"


@pytest.mark.parametrize("point, expected", [
    ([0.01, 0.5], 0),    # Very small u₁
    ([0.99, 0.5], 0.5),  # Nearly 1 for u₁
])
def test_cond_distr_2_edge_cases_approx(point, expected):
    """Test approximate edge cases for second conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_2(point)
    assert np.isclose(actual, expected, atol=0.01), f"cond_distr_2 at {point} should be approximately {expected}"


@pytest.mark.parametrize("point_1, point_2", [
    ([0.3, 0.7], [0.7, 0.3]),  # Test symmetry for this special case (should be equal)
    ([0.2, 0.2], [0.8, 0.8]),  # Points on diagonal (should be equal)
])
def test_cond_distr_symmetry(point_1, point_2):
    """Test that for specific symmetric theta matrices, certain symmetry properties hold."""
    # This symmetric theta should produce a symmetric copula
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    
    cd1_1 = cop.cond_distr_1(point_1)
    cd2_2 = cop.cond_distr_2(point_2)
    
    assert np.isclose(cd1_1, cd2_2), \
        f"For symmetric theta, cond_distr_1({point_1}) should equal cond_distr_2({point_2})"