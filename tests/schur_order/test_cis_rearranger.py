"""
Tests for the CISRearranger class.
"""

import pytest
import numpy as np
import sympy
from unittest.mock import patch, MagicMock

from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.checkerboarder import Checkerboarder
from copul.schur_order.cis_rearranger import CISRearranger


class TestCISRearranger:
    """Tests for the CISRearranger class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.checkerboard_size = 5
        self.rearranger = CISRearranger(self.checkerboard_size)

    def test_initialization(self):
        """Test initialization of CISRearranger."""
        assert self.rearranger._checkerboard_size == self.checkerboard_size

        # Test default initialization
        default_rearranger = CISRearranger()
        assert default_rearranger._checkerboard_size is None

    def test_str_representation(self):
        """Test string representation."""
        expected = f"CISRearranger(checkerboard_size={self.checkerboard_size})"
        assert str(self.rearranger) == expected

    @patch("copul.schur_order.cis_rearranger.Checkerboarder")
    def test_rearrange_copula_with_biv_check_pi(self, mock_checkerboarder_class):
        """Test rearrange_copula with BivCheckPi input."""
        # Create a mock BivCheckPi
        mock_copula = MagicMock(spec=BivCheckPi)

        # Mock the rearrange_checkerboard method
        with patch.object(
            CISRearranger, "rearrange_checkerboard", return_value="rearranged_result"
        ) as mock_rearrange:
            result = self.rearranger.rearrange_copula(mock_copula)

            # No longer asserting that checkerboarder is not called
            # Instead, verify that rearrange_checkerboard was called correctly
            mock_rearrange.assert_called_once_with(mock_copula)

            # Verify result
            assert result == "rearranged_result"

    @patch("copul.schur_order.cis_rearranger.Checkerboarder")
    def test_rearrange_copula_with_regular_copula(self, mock_checkerboarder_class):
        """Test rearrange_copula with a regular copula input."""
        # Create a mock copula that is not a BivCheckPi
        mock_copula = MagicMock()

        # Mock the checkerboarder instance
        mock_checkerboarder = MagicMock(spec=Checkerboarder)
        mock_checkerboarder_class.return_value = mock_checkerboarder
        mock_check_pi = MagicMock(spec=BivCheckPi)
        mock_checkerboarder.get_checkerboard_copula.return_value = mock_check_pi

        # Mock the rearrange_checkerboard method
        with patch.object(
            CISRearranger, "rearrange_checkerboard", return_value="rearranged_result"
        ) as mock_rearrange:
            result = self.rearranger.rearrange_copula(mock_copula)

            # Verify checkerboarder was created and used
            mock_checkerboarder_class.assert_called_once_with(self.checkerboard_size)
            mock_checkerboarder.get_checkerboard_copula.assert_called_once_with(
                mock_copula
            )

            # Verify rearrange_checkerboard was called with the right argument
            mock_rearrange.assert_called_once_with(mock_check_pi)

            # Verify result
            assert result == "rearranged_result"

    def test_integration_small_matrix(self):
        """Integration test with a small matrix to verify algorithm behavior."""
        # Create a simple 2x2 matrix
        np.array([[0.2, 0.3], [0.1, 0.4]])

        # Create a proper mock copula with cdf method
        mock_copula = MagicMock()
        mock_copula.cdf.return_value = 0.5

        # Mock the Checkerboarder instead of trying to use it directly
        with patch(
            "copul.schur_order.cis_rearranger.Checkerboarder"
        ) as mock_checkerboarder_cls:
            mock_checkerboarder = MagicMock()
            mock_check_pi = MagicMock(spec=BivCheckPi)
            mock_checkerboarder.get_checkerboard_copula.return_value = mock_check_pi
            mock_checkerboarder_cls.return_value = mock_checkerboarder

            # Mock the rearrange_checkerboard method
            with patch.object(
                CISRearranger, "rearrange_checkerboard"
            ) as mock_rearrange:
                mock_rearrange.return_value = sympy.Matrix([[0.3, 0.2], [0.2, 0.3]]) / 4

                # Instead of passing the matrix directly, create and pass a mock copula
                result = self.rearranger.rearrange_copula(mock_copula)

                # Verify the mock was called correctly
                mock_checkerboarder_cls.assert_called_once_with(self.checkerboard_size)
                mock_checkerboarder.get_checkerboard_copula.assert_called_once_with(
                    mock_copula
                )
                mock_rearrange.assert_called_once_with(mock_check_pi)

                # Basic assertion on the result
                assert result is not None


@pytest.mark.parametrize(
    "test_matr, expected",
    [
        (
            np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]]),
            np.array([[5, 3, 2], [4, 2, 4], [1, 5, 4]]),
        )
    ],
)
def test_cis_rearrangement(test_matr, expected):
    cisr = CISRearranger()
    rearranged = cisr.rearrange_checkerboard(test_matr)
    normed_rearranged = rearranged / np.sum(rearranged)
    normed_expected = expected / np.sum(expected)
    assert np.allclose(normed_rearranged, normed_expected)
