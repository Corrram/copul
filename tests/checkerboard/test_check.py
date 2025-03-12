import numpy as np
import sympy
from unittest.mock import patch, MagicMock
from copul.checkerboard.check import Check


class TestCheck:
    def test_init_numpy(self):
        """Test initialization with a numpy array."""
        matr = np.array([[1.0, 2.0], [3.0, 4.0]])  # Sum = 10
        check = Check(matr)
        # Matrix should be normalized to sum to 1
        expected = matr / matr.sum()
        assert np.allclose(check.matr, expected)
        assert check.dim == (2, 2)
        assert check.d == 2

    def test_init_list(self):
        """Test initialization with a list."""
        matr = [[1.0, 2.0], [3.0, 4.0]]  # Sum = 10
        check = Check(matr)
        # Check conversion to numpy array and normalization
        expected = np.array(matr) / np.array(matr).sum()
        assert np.allclose(check.matr, expected)
        assert check.dim == (2, 2)
        assert check.d == 2

    def test_init_sympy(self):
        """Test initialization with a sympy Matrix."""
        matr = sympy.Matrix([[1.0, 2.0], [3.0, 4.0]])  # Sum = 10
        check = Check(matr)
        expected = matr / sum(matr)  # Using sum() instead of .sum() for sympy matrices
        # For sympy matrices, equality check should work
        assert check.matr == expected
        assert check.dim == (2, 2)
        assert check.d == 2

    def test_normalization(self):
        """Test that the matrix is properly normalized."""
        # Test with a matrix that doesn't sum to 1
        matr = np.array([[1.0, 2.0], [3.0, 4.0]])  # Sum = 10
        check = Check(matr)
        assert np.isclose(check.matr.sum(), 1.0)

        # The normalized matrix should be proportional to the original
        expected = matr / matr.sum()
        assert np.allclose(check.matr, expected)

    def test_lambda_L(self):
        """Test the lower tail dependence calculation."""
        matr = np.array([[0.2, 0.3], [0.1, 0.4]])
        check = Check(matr)
        assert check.lambda_L() == 0

    def test_lambda_U(self):
        """Test the upper tail dependence calculation."""
        matr = np.array([[0.2, 0.3], [0.1, 0.4]])
        check = Check(matr)
        assert check.lambda_U() == 0

    def test_str_representation(self):
        """Test the string representation of the object."""
        matr = np.array([[0.2, 0.3], [0.1, 0.4]])
        check = Check(matr)
        assert str(check) == "CheckerboardCopula((2, 2))"

    @patch("mfoci.codec")
    def test_chatterjees_xi(self, mock_codec):
        """Test the calculation of Chatterjee's Xi."""
        # Setup mock return value for mfoci.codec
        mock_codec.return_value = 0.5

        # Create a Check instance
        matr = np.array([[0.2, 0.3], [0.1, 0.4]])
        check = Check(matr)

        # Use a fixed seed to ensure deterministic mock samples
        np.random.seed(42)
        mock_samples = np.random.random((10_000, 3))  # Create samples with 3 columns
        check.rvs = MagicMock(return_value=mock_samples)

        # Calculate Chatterjee's Xi
        n = 1_000
        xi = check.chatterjees_xi(n)

        # Verify rvs was called with the expected argument
        check.rvs.assert_called_once_with(n, random_state=None)

        # Verify mfoci.codec was called correctly (avoiding direct NumPy array comparison)
        mock_codec.assert_called_once()
        args, _ = mock_codec.call_args

        # Check that args[0] is the first column of mock_samples
        assert np.array_equal(args[0], mock_samples[:, 0])

        # Check that args[1] contains columns 1:3 of mock_samples
        assert np.array_equal(args[1], mock_samples[:, 1:3])

        # Verify the result
        assert xi == 0.5

    def test_high_dimensional_matrix(self):
        """Test with a higher dimensional matrix."""
        matr = np.ones((2, 3, 4))  # Sum = 24
        check = Check(matr)
        assert check.dim == (2, 3, 4)
        assert check.d == 3
        assert np.isclose(check.matr.sum(), 1.0)
        # Check normalization
        expected = matr / matr.sum()
        assert np.allclose(check.matr, expected)
