import numpy as np
import sympy
from unittest.mock import patch, MagicMock
from copul.checkerboard.check import Check
import pytest


def test_init_numpy():
    """Test initialization with a numpy array."""
    matr = np.array([[1.0, 2.0], [3.0, 4.0]])  # Sum = 10
    check = Check(matr)
    # Matrix should be normalized to sum to 1
    expected = matr / matr.sum()
    assert np.allclose(check.matr, expected)
    assert check.matr.shape == (2, 2)
    assert check.dim == 2


def test_init_list():
    """Test initialization with a list."""
    matr = [[1.0, 2.0], [3.0, 4.0]]  # Sum = 10
    check = Check(matr)
    # Check conversion to numpy array and normalization
    expected = np.array(matr) / np.array(matr).sum()
    assert np.allclose(check.matr, expected)
    assert check.matr.shape == (2, 2)
    assert check.dim == 2


def test_init_sympy():
    """Test initialization with a sympy Matrix."""
    matr = sympy.Matrix([[1.0, 2.0], [3.0, 4.0]])  # Sum = 10
    check = Check(matr)
    expected = np.array([[0.1, 0.2], [0.3, 0.4]])  # Normalized matrix
    actual = check.matr
    assert np.allclose(actual, expected)
    assert actual.shape == (2, 2)
    assert check.dim == 2


def test_normalization():
    """Test that the matrix is properly normalized."""
    # Test with a matrix that doesn't sum to 1
    matr = np.array([[1.0, 2.0], [3.0, 4.0]])  # Sum = 10
    check = Check(matr)
    assert np.isclose(check.matr.sum(), 1.0)

    # The normalized matrix should be proportional to the original
    expected = matr / matr.sum()
    assert np.allclose(check.matr, expected)


def test_lambda_L():
    """Test the lower tail dependence calculation."""
    matr = np.array([[0.2, 0.3], [0.1, 0.4]])
    check = Check(matr)
    assert check.lambda_L() == 0


def test_lambda_U():
    """Test the upper tail dependence calculation."""
    matr = np.array([[0.2, 0.3], [0.1, 0.4]])
    check = Check(matr)
    assert check.lambda_U() == 0


def test_str_representation():
    """Test the string representation of the object."""
    matr = np.array([[0.2, 0.3], [0.1, 0.4]])
    check = Check(matr)
    assert str(check) == "CheckerboardCopula((2, 2))"


@patch("copul.checkerboard.check.codec")
def test_xi(mock_codec):
    """Test the calculation of Chatterjee's Xi."""
    mock_codec.return_value = 0.5

    # Create a Check instance
    matr = np.array([[0.2, 0.3], [0.1, 0.4]])
    check = Check(matr)

    # Use a fixed seed to ensure deterministic mock samples
    np.random.seed(42)
    mock_samples = np.random.random((10_000, 2))  # Create samples with 3 columns
    check.rvs = MagicMock(return_value=mock_samples)

    # Calculate Chatterjee's Xi
    n = 1_000
    xi = check.chatterjees_xi(n)

    # Verify rvs was called with the expected argument
    check.rvs.assert_called_once_with(n, random_state=None)

    mock_codec.assert_called_once()
    args, _ = mock_codec.call_args

    # Check that args[0] is the first column of mock_samples
    assert np.array_equal(args[0], mock_samples[:, 0])

    # Check that args[1] contains columns 1:3 of mock_samples
    assert np.array_equal(args[1], mock_samples[:, 1:3])

    # Verify the result
    assert xi == 0.5


def test_high_dimensional_matrix():
    """Test with a higher dimensional matrix."""
    matr = np.ones((2, 3, 4))  # Sum = 24
    check = Check(matr)
    assert check.matr.shape == (2, 3, 4)
    assert check.dim == 3
    assert np.isclose(check.matr.sum(), 1.0)
    # Check normalization
    expected = matr / matr.sum()
    assert np.allclose(check.matr, expected)


def test_from_data_already_uniform():
    """Test creating a copula from data that's already in [0, 1] space."""
    # Create uniform data in [0, 1]
    np.random.seed(42)  # For reproducibility
    uniform_data = np.random.random((100, 2))

    # Create a copula with 5 bins in each dimension
    copula = Check.from_data(uniform_data, num_bins=5, already_uniform=True)

    # Check properties
    assert isinstance(copula, Check)
    assert copula.matr.shape == (5, 5)
    assert copula.dim == 2
    assert np.isclose(copula.matr.sum(), 1.0)


def test_from_data_needs_transformation():
    """Test creating a copula from raw data that needs transformation."""
    # Create non-uniform data
    np.random.seed(42)
    raw_data = np.random.normal(size=(100, 2))  # Normal distribution

    # Create a copula with default number of bins
    copula = Check.from_data(raw_data)

    # Check properties
    assert isinstance(copula, Check)
    # Default num_bins should be approximately sqrt(n_samples)
    expected_bins = 4  # 100^(1/3) rounded down
    assert copula.matr.shape == (expected_bins, expected_bins)
    assert copula.dim == 2
    assert np.isclose(copula.matr.sum(), 1.0)


def test_from_data_custom_bins():
    """Test creating a copula with custom bin sizes per dimension."""
    np.random.seed(42)
    data = np.random.random((100, 3))

    # Different number of bins for each dimension
    bins = [5, 10, 15]
    copula = Check.from_data(data, num_bins=bins, already_uniform=True)

    # Check properties
    assert isinstance(copula, Check)
    assert copula.matr.shape == tuple(bins)
    assert copula.dim == 3
    assert np.isclose(copula.matr.sum(), 1.0)


def test_from_data_1d():
    """Test creating a copula from 1D data."""
    np.random.seed(42)
    data = np.random.random(100)  # 1D array

    copula = Check.from_data(data, num_bins=10, already_uniform=True)

    # Check properties
    assert isinstance(copula, Check)
    assert copula.matr.shape == (10,)
    assert copula.dim == 1
    assert np.isclose(copula.matr.sum(), 1.0)


def test_from_data_with_ties():
    """Test creating a copula from data with ties."""
    # Create data with intentional ties
    data = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.3],  # Tie in first column
            [0.2, 0.2],  # Tie in second column
            [0.3, 0.4],
        ]
    )

    copula = Check.from_data(data, num_bins=2)

    # Check properties
    assert isinstance(copula, Check)
    assert copula.matr.shape == (2, 2)
    assert copula.dim == 2
    assert np.isclose(copula.matr.sum(), 1.0)


def test_from_data_implementations():
    """Test that both implementations (with and without scipy) work."""
    np.random.seed(42)
    data = np.random.normal(size=(100, 2))

    # First, test with scipy available (normal case)
    copula1 = Check.from_data(data, num_bins=5)
    assert isinstance(copula1, Check)
    assert copula1.matr.shape == (5, 5)

    # Now, test with scipy unavailable (fallback to numpy implementation)
    with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
        has_scipy_orig = Check.from_data.__globals__.get("has_scipy", True)
        try:
            # Force has_scipy to False
            Check.from_data.__globals__["has_scipy"] = False

            copula2 = Check.from_data(data, num_bins=5)
            assert isinstance(copula2, Check)
            assert copula2.matr.shape == (5, 5)

            # Both should produce normalized matrices
            assert np.isclose(copula1.matr.sum(), 1.0)
            assert np.isclose(copula2.matr.sum(), 1.0)

            # The results might differ slightly due to different implementations,
            # but both should be valid copulas
        finally:
            # Restore original has_scipy value
            Check.from_data.__globals__["has_scipy"] = has_scipy_orig


def test_from_data_empty():
    """Test error handling with empty data."""
    data = np.array([])

    with pytest.raises(ValueError):
        Check.from_data(data)


def test_from_data_invalid_bins():
    """Test error handling with invalid bin specifications."""
    data = np.random.random((100, 2))

    # Wrong number of bin dimensions
    with pytest.raises(ValueError):
        Check.from_data(data, num_bins=[5, 10, 15])  # 3 dimensions for 2D data


def test_from_data_bin_distribution():
    """Test that the distribution of points in bins is accurate."""
    # Create data with known distribution
    np.random.seed(42)
    n_samples = 1000
    data = np.random.random((n_samples, 2))

    # Force some points into specific quadrants to test binning
    data[:250, 0] = data[:250, 0] * 0.5  # 25% in left half
    data[250:500, 0] = 0.5 + data[250:500, 0] * 0.5  # 25% in right half
    data[:500, 1] = data[:500, 1] * 0.5  # 50% in bottom half
    data[500:, 1] = 0.5 + data[500:, 1] * 0.5  # 50% in top half

    # Create a 2x2 grid
    copula = Check.from_data(data, num_bins=2, already_uniform=True)

    # The distribution should be roughly:
    # Bottom left: 25% of points
    # Bottom right: 25% of points
    # Top left: 25% of points
    # Top right: 25% of points
    assert np.isclose(copula.matr[0, 0], 0.25, atol=0.05)
    assert np.isclose(copula.matr[0, 1], 0.25, atol=0.05)
    assert np.isclose(copula.matr[1, 0], 0.25, atol=0.05)
    assert np.isclose(copula.matr[1, 1], 0.25, atol=0.05)
