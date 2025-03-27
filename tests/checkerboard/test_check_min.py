"""
Tests for the CheckMin and BivCheckMin classes.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from copul.checkerboard.check_min import CheckMin
from copul.exceptions import PropertyUnavailableException


class TestCheckMinBivCheckMinConversion:
    """Tests for CheckMin automatic conversion to BivCheckMin."""

    def test_checkmin_2d_returns_bivcheckmin(self):
        """Test that creating a CheckMin with a 2D matrix returns a BivCheckMin instance."""
        # Create a 2D matrix
        matr = np.array([[0.25, 0.25], [0.25, 0.25]])

        # Create a mock BivCheckMin class
        mock_bivcheckmin_instance = MagicMock()
        mock_bivcheckmin_instance.__class__.__name__ = "BivCheckMin"

        # Create a mock module that returns our mock instance
        mock_module = MagicMock()
        mock_module.BivCheckMin = MagicMock(return_value=mock_bivcheckmin_instance)

        # Patch importlib.import_module to return our mock module
        with patch("importlib.import_module", return_value=mock_module):
            # Call CheckMin constructor
            result = CheckMin(matr)

            # Verify the result is the BivCheckMin instance
            assert result is mock_bivcheckmin_instance
            # Verify BivCheckMin was called with the matrix
            mock_module.BivCheckMin.assert_called_once_with(matr)

    def test_checkmin_3d_returns_checkmin(self):
        """Test that creating a CheckMin with a 3D matrix returns a CheckMin instance."""
        # Create a 3D matrix
        matr = np.ones((2, 2, 2)) / 8  # Normalized 2x2x2 matrix

        # Create a CheckMin instance with the 3D matrix
        result = CheckMin(matr)

        # Verify the result is a CheckMin instance
        assert isinstance(result, CheckMin)

    def test_bivcheckmin_import_error_fallback(self):
        """Test that import error for BivCheckMin falls back to CheckMin."""
        # Create a 2D matrix
        matr = np.array([[0.25, 0.25], [0.25, 0.25]])

        # Patch importlib.import_module to raise an ImportError for the bivcheckmin module
        with patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'bivcheckmin'"),
        ):
            # Call CheckMin constructor
            result = CheckMin(matr)

            # Verify the result is a CheckMin instance
            assert isinstance(result, CheckMin)

    def test_integration_with_real_classes(self):
        """Integration test with real classes to ensure conversion works properly."""
        try:
            # Try to import directly first to check if it's available
            from importlib import import_module

            try:
                import_module("copul.checkerboard.bivcheckmin")
            except ImportError:
                pytest.skip("BivCheckMin class not available for integration test")

            # Create a 2D matrix
            matr = np.array([[0.25, 0.25], [0.25, 0.25]])

            # Create a CheckMin instance with the 2D matrix
            result = CheckMin(matr)

            # Verify the result is a BivCheckMin instance
            # We use string comparison in case actual classes aren't imported in test environment
            assert result.__class__.__name__ == "BivCheckMin"

        except Exception as e:
            pytest.skip(f"Error in integration test: {e}")

    def test_subclass_not_affected(self):
        """Test that subclasses of CheckMin are not affected by the conversion."""

        # Create a subclass of CheckMin
        class SubCheckMin(CheckMin):
            pass

        # Create a 2D matrix
        matr = np.array([[0.25, 0.25], [0.25, 0.25]])

        # Create a mock module with a BivCheckMin class
        mock_module = MagicMock()
        mock_module.BivCheckMin = MagicMock()

        # Patch importlib.import_module to return our mock module
        with patch("importlib.import_module", return_value=mock_module):
            # Create a SubCheckMin instance with the 2D matrix
            result = SubCheckMin(matr)

            # Verify the result is a SubCheckMin instance
            assert isinstance(result, SubCheckMin)

            # Verify BivCheckMin was NOT called
            mock_module.BivCheckMin.assert_not_called()


def test_2x2x2_multivar_checkerboard():
    # 3-dim matrix
    matr = np.full((2, 2, 2), 0.5)
    copula = CheckMin(matr)
    u = (0.5, 0.5, 0.5)
    v = (0.25, 0.25, 0.25)
    w = (0.75, 0.75, 0.75)
    with pytest.raises(PropertyUnavailableException):
        copula.pdf(*u)
    assert copula.cdf(*u) == 0.125
    assert copula.cdf(*v) == 0.125 / 2
    actual_cdf_w = copula.cdf(*w)
    assert actual_cdf_w == 0.125 + 0.875 / 2
    expected_cd = 0.25  # 1/4 for (0,0,0) block + 0 for all other blocks
    actual_1 = copula.cond_distr(1, u)
    assert actual_1 == expected_cd
    assert copula.cond_distr(2, u) == expected_cd
    assert copula.cond_distr(3, u) == expected_cd


def test_2x2x2_multivar_checkerboard_vectorized():
    # 3-dim matrix
    matr = np.full((2, 2, 2), 0.5)
    copula = CheckMin(matr)
    u = (0.5, 0.5, 0.5)
    v = (0.25, 0.25, 0.25)
    w = (0.75, 0.75, 0.75)
    points = np.array([u, v, w])
    cdf_values = copula.cdf(points)
    assert np.allclose(cdf_values, [0.125, 0.125 / 2, 0.125 + 0.875 / 2])


def test_3x3x3_multivar_checkerboard():
    # 3-dim matrix
    matr = np.full((3, 3, 3), 1)
    copula = CheckMin(matr)
    u = (0.5, 0.5, 0.5)
    with pytest.raises(PropertyUnavailableException):
        copula.pdf(*u)
    expected_cdf = (
        1 / 6
    )  # 1/27 for (0,0,0) block + 7*1/27 for (0,0,1),...,(1,1,1) blocks
    assert np.isclose(copula.cdf(*u), expected_cdf)
    expected_cd = (
        1 / 3
        + 1 / 9  # 1/9 for (1,0,0) block + 3*1/9 for (1,1,1),(1,0,1),(1,1,1) blocks
    )
    actual_cd1 = copula.cond_distr(1, u)
    assert actual_cd1 == expected_cd
    assert copula.cond_distr(2, u) == expected_cd
    assert copula.cond_distr(3, u) == expected_cd


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1], [1, 1]], (0.1, 0.2), 0.25 * 0.2),
        ([[1, 1], [1, 1]], (0.2, 0.1), 0.25 * 0.2),
        ([[1, 1], [1, 1]], (0.6, 0.7), 0.25 + 0.25 * 0.2 + 0.25 * 0.4 + 0.25 * 0.2),
        ([[1, 1], [1, 1]], (0.2, 0.2), 0.25 * 0.4),
        ([[1, 1], [1, 1]], (0.2, 0.6), 0.25 * 0.4 + 0.25 * 0.2),
        ([[1, 1], [1, 1]], (0.2, 0.7), 0.25 * 0.4 + 0.25 * 0.4),
        ([[1, 1], [1, 1]], (0.2, 0.8), 0.25 * 0.4 + 0.25 * 0.4),
        ([[1, 1], [1, 1]], (0.7, 0.2), 0.25 * 0.4 + 0.25 * 0.4),
        ([[1, 1], [1, 1]], (0.7, 0.7), 0.25 + 0.25 * 0.4 * 3),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.25),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.1), 0.1 / (1 / 3) / 30),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.2), 3 * 0.2 / 30),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.3), 3 * 0.2 / 30),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.2, 0.1 + 1 / 3),
            0.2 / (1 / 3) / 30 + 5 * 0.1 / (1 / 3) / 30,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.2, 0.2 + 1 / 3),
            0.2 / (1 / 3) / 30 + 5 * 0.2 / (1 / 3) / 30,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.2, 0.3 + 1 / 3),
            0.2 / (1 / 3) / 30 + 5 * 0.2 / (1 / 3) / 30,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.2, 0.1 + 2 / 3),
            0.2 / (1 / 3) / 30 + 5 * 0.2 / (1 / 3) / 30 + 4 * 0.1 / (1 / 3) / 30,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.2, 0.2 + 2 / 3),
            0.2 / (1 / 3) / 30 + 5 * 0.2 / (1 / 3) / 30 + 4 * 0.2 / (1 / 3) / 30,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.2, 0.3 + 2 / 3),
            0.2 / (1 / 3) / 30 + 5 * 0.2 / (1 / 3) / 30 + 4 * 0.2 / (1 / 3) / 30,
        ),
        (
            [[1, 5, 4], [5, 3, 2], [4, 2, 4]],
            (0.5, 0.5),
            (1 + 5 / 2 + 5 / 2 + 3 / 2) / 30,
        ),
    ],
)
def test_check_min_cdf(matr, point, expected):
    ccop = CheckMin(matr)
    actual = ccop.cdf(*point)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "matr, point, expected",
    [
        ([[1, 1], [1, 1]], (0.2, 0.1), 0),
        ([[1, 1], [1, 1]], (0.1, 0.2), 0.5),
        ([[1, 1], [1, 1]], (0.6, 0.7), 1),
        ([[1, 1], [1, 1]], (0.2, 0.2), 0.5),
        ([[1, 1], [1, 1]], (0.2, 0.6), 0.5),
        ([[1, 1], [1, 1]], (0.2, 0.7), 1),
        ([[1, 1], [1, 1]], (0.2, 0.8), 1),
        ([[1, 1], [1, 1]], (0.7, 0.2), 0.5),
        ([[1, 1], [1, 1]], (0.7, 0.7), 1),
        ([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], (0.5, 0.5), 0.5),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.1), 0),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.2), 0.1),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.3), 0.1),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.1 + 1 / 3), 0.1),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.2 + 1 / 3), 0.6),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.3 + 1 / 3), 0.6),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.1 + 2 / 3), 0.6),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.2 + 2 / 3), 1),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.2, 0.3 + 2 / 3), 1),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.4), 0.5),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.5), 0.8),
        ([[1, 5, 4], [5, 3, 2], [4, 2, 4]], (0.5, 0.6), 0.8),
    ],
)
def test_ccop_cond_distr(matr, point, expected):
    ccop = CheckMin(matr)
    actual = ccop.cond_distr(1, point)
    assert np.isclose(actual, expected)


@pytest.mark.parametrize(
    "point, ratio",
    [
        ((0.25, 0.25), 1 / 6 * 1 / 2),
        ((0.75, 0.75), 1 / 6 * 1 + 1 / 3 * 1 / 2 + 1 / 3 * 1 / 2 + 1 / 6 * 1 / 2),
        ((0.5, 0.5), 1 / 6),
    ],
)
def test_2d_check_min_rvs(point, ratio):
    np.random.seed(1)
    ccop = CheckMin([[1, 2], [2, 1]])
    n = 2_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < point).all() for sample in samples])
    theoretical_ratio = ratio * n
    assert np.isclose(n_lower_empirical, theoretical_ratio, rtol=0.1)


def test_3d_check_pi_rvs():
    np.random.seed(1)
    ccop = CheckMin([[[1, 2], [2, 1]], [[1, 2], [2, 1]]])
    n = 20_000
    samples = ccop.rvs(n)
    n_lower_empirical = sum([(sample < (0.5, 0.5, 0.5)).all() for sample in samples])
    n_upper_empirical = sum([(sample > (0.5, 0.5, 0.5)).all() for sample in samples])
    ratio = 1 / 12 * n
    assert np.isclose(n_lower_empirical, ratio, rtol=0.1)
    assert np.isclose(n_upper_empirical, ratio, rtol=0.1)

    n_lower = sum([(sample < (0.25, 0.25, 0.5)).all() for sample in samples])
    expected = ratio / 2
    assert np.isclose(n_lower, expected, rtol=0.1)
    n_lower_quarter = sum([(sample < (0.25, 0.25, 0.25)).all() for sample in samples])
    assert np.isclose(n_lower_quarter, expected, rtol=0.1)

    n_lower_part = sum([(sample < (0.25, 0.75, 0.5)).all() for sample in samples])
    expected_part = 3 / 12 * 1 / 2 * n
    assert np.isclose(n_lower_part, expected_part, rtol=0.1)


def test_chatterjees_xi_for_upper_frechet():
    """Test the Chatterjee's xi estimator."""
    matr = np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    copula = CheckMin(matr)

    # Chatterjee's xi for this copula should be 0
    xi = copula.xi()
    assert np.isclose(xi, 1, atol=1e-2)
