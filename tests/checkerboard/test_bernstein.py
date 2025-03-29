import numpy as np
import pytest
from scipy.stats import kstest, pearsonr
from scipy.integrate import dblquad

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
    assert cop.degrees == [2, 3]

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
    assert cop.degrees == [1, 2, 3]

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
    ([0.5, 0], 0),       # P(U₁≤0|U₂=0.5) = 0
    ([0.5, 1], 1),       # P(U₁≤1|U₂=0.5) = 1
])
def test_cond_distr_1_edge_cases(point, expected):
    """Test edge cases for first conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_1(point)
    assert np.isclose(actual, expected), f"cond_distr_1 at {point} should be {expected}"


@pytest.mark.parametrize("point, expected", [
    ([0, 0.5], 0),    # Very small u₁
    ([1, 0.5], 1),  # Nearly 1 for u₁
])
def test_cond_distr_2_edge_cases(point, expected):
    """Test edge cases for second conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_2(point)
    assert np.isclose(actual, expected), f"cond_distr_2 at {point} should be {expected}"


@pytest.mark.parametrize("point, expected", [
    ([0.5, 0.01], 0),    # Very small u₁
    ([0.5, 0.99], 1),    # Nearly 1 for u₁
])
def test_cond_distr_1_edge_cases_approx(point, expected):
    """Test approximate edge cases for first conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_1(point)
    assert np.isclose(actual, expected, atol=0.01), f"cond_distr_1 at {point} should be approximately {expected}"


@pytest.mark.parametrize("point, expected", [
    ([0.01, 0.5], 0),    # Very small u₁
    ([0.99, 0.5], 1),  # Nearly 1 for u₁
])
def test_cond_distr_2_edge_cases_approx(point, expected):
    """Test approximate edge cases for second conditional distribution."""
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_2(point)
    assert np.isclose(actual, expected, atol=0.01), f"cond_distr_2 at {point} should be approximately {expected}"

@pytest.mark.parametrize("point, expected", [
    ([0.01, 0.2], 0),    # Very small u₁
    ([0.97, 0.2], 1),  # Nearly 1 for u₁
])
def test_cond_distr_2_edge_cases_approx_3_times_3(point, expected):
    """Test approximate edge cases for second conditional distribution."""
    matr = np.array([[0.26622,0.05788,0.00924],[0.05788,0.17621,0.09925],[0.00924,0.09925,0.22485]])
    theta = np.array(matr)
    cop = BernsteinCopula(theta)
    actual = cop.cond_distr_2(point)
    assert np.isclose(actual, expected, atol=0.02), f"cond_distr_2 at {point} should be approximately {expected}"

def test_cond_distr_symmetry():
    """Test that for specific symmetric theta matrices, certain symmetry properties hold."""
    # This symmetric theta should produce a symmetric copula
    theta = np.array([[0.5, 0.5], [0.5, 0.5]])
    cop = BernsteinCopula(theta)
    
    cd1_1 = cop.cond_distr_1([0.3, 0.7])
    cd2_2 = cop.cond_distr_2([0.7, 0.3])
    
    assert np.isclose(cd1_1, cd2_2)


def test_pdf_2d_independence_single_point():
    """
    For a 2D Bernstein copula with shape=(2,2) and equal corner mass = 0.25 each,
    it essentially acts like the independence copula. 
    The PDF should be ~1.0 for 0<u<1.
    """
    # shape=(2,2) => m1=1, m2=1
    # If each corner is 0.25, that yields "independence-like" behavior.
    theta = np.array([[0.25, 0.25],
                      [0.25, 0.25]])
    cop = BernsteinCopula(theta)

    # Check a few interior points
    points = [[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]]
    for p in points:
        pdf_val = cop.pdf(p)
        assert np.isclose(pdf_val, 1.0, atol=1e-2), \
            f"For point {p}, PDF should be ~1, got {pdf_val}."

def test_pdf_2d_independence_vectorized():
    """
    Same as above, but checks vectorized calls for multiple points at once.
    """
    theta = np.array([[0.25, 0.25],
                      [0.25, 0.25]])
    cop = BernsteinCopula(theta)

    pts = np.array([
        [0.5, 0.5],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.2, 0.2],
        [0.8, 0.8],
    ])
    pdf_vals = cop.pdf(pts)
    # Expect ~1.0 for all interior points
    assert np.allclose(pdf_vals, 1.0, atol=0.02), \
        f"PDF values should all be ~1, got {pdf_vals}"

def test_pdf_2d_independence_integrates_to_one():
    """
    Numerically integrate the PDF over [0,1]^2 to confirm it's ~1.
    This test can be slower depending on the integration resolution.
    """
    theta = np.array([[0.25, 0.25],
                      [0.25, 0.25]])
    cop = BernsteinCopula(theta)

    def pdf_wrapper(u, v):
        return cop.pdf([u, v])

    # Use a double integral from 0..1
    val, err = dblquad(pdf_wrapper, 0, 1, lambda _: 0, lambda _: 1)
    assert np.isclose(val, 1.0, atol=1e-2), \
        f"Integral of PDF should be ~1. Got {val} (err={err})."


def test_pdf_raises_for_out_of_bounds():
    """
    PDF should raise ValueError if input is out of [0,1].
    """
    theta = np.array([[0.25, 0.25],
                      [0.25, 0.25]])
    cop = BernsteinCopula(theta)
    with pytest.raises(ValueError):
        _ = cop.pdf([-0.1, 0.5])
    with pytest.raises(ValueError):
        _ = cop.pdf([1.1, 0.5])

    # Similarly for vectorized
    with pytest.raises(ValueError):
        _ = cop.pdf([[0.5, 0.5], [2.0, -1.0]])


# ------------------------------------------------------------------------------
#                          Tests for rvs(...)
# ------------------------------------------------------------------------------
def test_rvs_independence_2d():
    """
    Generate random variates from a 2D independence-like Bernstein Copula
    and check that marginals are ~ Uniform(0,1) and correlation is ~0.
    """
    rng = np.random.default_rng(42)
    theta = np.array([[0.25, 0.25],
                      [0.25, 0.25]])
    cop = BernsteinCopula(theta)
    n = 1000
    samples = cop.rvs(n=n, random_state=rng, approximate=True)
    
    assert samples.shape == (n, 2)
    # Check each marginal is ~Uniform(0,1) using KS test
    for dim_idx in range(2):
        stat, pval = kstest(samples[:, dim_idx], "uniform")
        assert pval > 1e-3, \
            f"Marginal {dim_idx} fails uniformity KS test with pval={pval}."

    # Check correlation is near zero
    corr, _ = pearsonr(samples[:, 0], samples[:, 1])
    assert abs(corr) < 0.05, f"Expected near-zero correlation, got {corr}."


def test_rvs_corner_distribution_2d():
    """
    If all mass is in top-right corner, the samples should concentrate near (1,1).
    """
    rng = np.random.default_rng(123)
    theta = np.array([[1.0, 0.0],
                      [0.0, 1.0]])
    # small epsilon to avoid degeneracy
    eps = 1e-10
    theta[0,0] = eps
    theta /= theta.sum()
    cop = BernsteinCopula(theta)

    n = 200
    samples = cop.rvs(n=n, random_state=rng, approximate=True)
    means = np.mean(samples, axis=0)
    assert 0.4 < means[0] < 0.7 and 0.4 < means[1] < 0.7
