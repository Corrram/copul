import numpy as np
import pytest

from copul.checkerboard.bernstein import BernsteinCopula


def test_1d_degree_0():
    """
    Degree 0 in 1D means the copula is the function C(u)=u
    only if theta=1. Then pdf=1, cdf(u)=u.
    """
    theta = np.array([1.0])  # shape=(1,), so m_1=0
    cop = BernsteinCopula(theta)
    assert cop.dim == 1
    assert cop.degrees == [0]
    # test cdf
    pts = [0.0, 0.2, 0.5, 1.0]
    for p in pts:
        assert np.isclose(cop.cdf(p), p), f"cdf({p}) should be {p}"
    # test pdf
    for p in [0.2, 0.5, 0.8]:
        assert np.isclose(cop.pdf(p), 1.0), f"pdf({p}) should be 1.0"


def test_1d_degree_1():
    """
    Degree 1 in 1D => shape=(2,).
    Suppose theta = [0, 1], which also yields C(u)=u (classic example).
    """
    theta = np.array([0.0, 1.0])  # shape=(2,), m=1
    cop = BernsteinCopula(theta)
    assert cop.dim == 1
    assert cop.degrees == [1]

    # cdf at 0.6 => should be ~0.6
    cdf_val = cop.cdf(0.6)
    assert np.isclose(cdf_val, 0.6, atol=1e-6)

    # pdf at 0.6 => 1
    pdf_val = cop.pdf(0.6)
    assert np.isclose(pdf_val, 1.0, atol=1e-6)


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
