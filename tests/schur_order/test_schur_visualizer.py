"""
Tests for the SchurVisualizer class and related functions.
"""

import numpy as np
import sympy as sp
from unittest.mock import patch, MagicMock
import pathlib

from copul.family import archimedean
from copul.wrapper.sympy_wrapper import SymPyFuncWrapper

# Import from the module we're testing
from copul.schur_order.schur_visualizer import SchurVisualizer, visualize_rearranged


class TestSchurVisualizer:
    """Tests for the SchurVisualizer class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create a simple copula for testing
        self.copula = archimedean.Nelsen2
        self.v = 0.5
        self.x_vals = np.linspace(0, 1, 10)  # Using fewer points for faster tests
        self.visualizer = SchurVisualizer(self.copula, v=self.v, x_vals=self.x_vals)

    def test_initialization(self):
        """Test initialization of SchurVisualizer."""
        # Test with default x_vals
        vis1 = SchurVisualizer(self.copula, v=self.v)
        assert vis1.copula == self.copula
        assert vis1._v == self.v
        assert len(vis1._x_vals) == 500  # Default value

        # Test with custom x_vals
        vis2 = SchurVisualizer(self.copula, v=self.v, x_vals=self.x_vals)
        assert vis2.copula == self.copula
        assert vis2._v == self.v
        assert np.array_equal(vis2._x_vals, self.x_vals)

    @patch("sympy.lambdify")
    def test_compute_with_regular_copula(self, mock_lambdify):
        """Test compute method with a regular copula."""
        # Setup mock
        mock_function = MagicMock()
        mock_function.side_effect = lambda x: x  # Identity function for simplicity
        mock_lambdify.return_value = mock_function

        # Setup test object with mock
        self.visualizer.copula.cond_distr_1 = MagicMock()
        self.visualizer.copula.cond_distr_1.return_value.func = sp.symbols("x")

        # Call method
        result = self.visualizer.compute()

        # Verify
        assert len(result) == len(self.x_vals)
        mock_lambdify.assert_called_once()

    def test_compute_with_biv_check(self):
        """Test compute method with BivCheck copulas."""
        # Mock BivCheckPi copula
        with patch(
            "copul.checkerboard.biv_check_pi.BivCheckPi", autospec=True
        ) as MockBivCheckPi:
            # Create a properly mocked copula
            mock_pi_copula = MockBivCheckPi.return_value
            mock_pi_copula.cond_distr_1.return_value = lambda u: u

            # Create visualizer with mocked copula
            visualizer = SchurVisualizer(mock_pi_copula, v=self.v, x_vals=self.x_vals)
            result = visualizer.compute()
            assert len(result) == len(self.x_vals)
            mock_pi_copula.cond_distr_1.assert_called()

        # Mock BivCheckMin copula
        with patch(
            "copul.checkerboard.biv_check_min.BivCheckMin", autospec=True
        ) as MockBivCheckMin:
            # Create a properly mocked copula
            mock_min_copula = MockBivCheckMin.return_value
            mock_min_copula.cond_distr_1.return_value = lambda u: u

            # Create visualizer with mocked copula
            visualizer = SchurVisualizer(mock_min_copula, v=self.v, x_vals=self.x_vals)
            result = visualizer.compute()
            assert len(result) == len(self.x_vals)
            mock_min_copula.cond_distr_1.assert_called()

    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    @patch("pathlib.Path.mkdir")
    def test_plot_for_with_sympy_wrapper(
        self, mock_mkdir, mock_close, mock_show, mock_savefig, mock_plot
    ):
        """Test plot_for method with SymPyFuncWrapper return value."""
        mock_mkdir.return_value = None

        # Setup mocks
        self.visualizer.copula.params = [sp.symbols("theta")]
        self.visualizer.copula.__call__ = MagicMock()
        self.visualizer.copula.__str__ = MagicMock(return_value="MockCopula")

        # Create mock copula instance
        mock_copula_instance = MagicMock()
        mock_copula_instance.u = sp.symbols("u")
        mock_copula_instance.__str__ = MagicMock(return_value="MockCopula")

        # Setup cond_distr_1 to return SymPyFuncWrapper
        mock_wrapper = MagicMock(spec=SymPyFuncWrapper)
        mock_wrapper.func = sp.symbols("func")
        mock_copula_instance.cond_distr_1.return_value = mock_wrapper

        self.visualizer.copula.__call__.return_value = mock_copula_instance

        # Mock lambdify
        with patch("sympy.lambdify") as mock_lambdify:
            mock_function = MagicMock()
            mock_function.side_effect = lambda x: x  # Identity function
            mock_lambdify.return_value = mock_function

            # Mock _finish_plot and _finish_rearrangend_plot to avoid actual plotting
            with (
                patch.object(self.visualizer, "_finish_plot"),
                patch.object(self.visualizer, "_finish_rearrangend_plot"),
            ):
                # Call method
                result = self.visualizer.plot_for(1.5)

                # Verify
                assert isinstance(result, dict)
                assert 1.5 in result
                assert len(result[1.5]) == len(self.x_vals)
                mock_plot.assert_called()

    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    @patch("pathlib.Path.mkdir")
    def test_plot_for_with_callable(
        self, mock_mkdir, mock_close, mock_show, mock_savefig, mock_plot
    ):
        """Test plot_for method with callable return value."""
        mock_mkdir.return_value = None

        # Setup mocks
        self.visualizer.copula.params = [sp.symbols("theta")]
        self.visualizer.copula.__call__ = MagicMock()
        self.visualizer.copula.__str__ = MagicMock(return_value="MockCopula")

        # Create mock copula instance
        mock_copula_instance = MagicMock()
        mock_copula_instance.__str__ = MagicMock(return_value="MockCopula")

        # Setup cond_distr_1 to return callable
        mock_copula_instance.cond_distr_1.return_value = lambda x, v: x * v

        self.visualizer.copula.__call__.return_value = mock_copula_instance

        # Mock _finish_plot and _finish_rearrangend_plot to avoid actual plotting
        with (
            patch.object(self.visualizer, "_finish_plot"),
            patch.object(self.visualizer, "_finish_rearrangend_plot"),
        ):
            # Call method
            result = self.visualizer.plot_for(1.5)

            # Verify
            assert isinstance(result, dict)
            assert 1.5 in result
            assert len(result[1.5]) == len(self.x_vals)
            mock_plot.assert_called()

    @patch("pathlib.Path.mkdir")
    def test_get_schur_image_path(self, mock_mkdir):
        """Test _get_schur_image_path method."""
        mock_mkdir.return_value = None

        path = SchurVisualizer._get_schur_image_path()
        assert isinstance(path, pathlib.Path)
        assert "images" in str(path)
        assert "schur" in str(path)
        mock_mkdir.assert_called_once_with(exist_ok=True)

    @patch("matplotlib.pyplot.gca")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    @patch("pathlib.Path.mkdir")
    def test_visualize_rearranged(
        self, mock_mkdir, mock_close, mock_show, mock_savefig, mock_plot, mock_gca
    ):
        """Test visualize_rearranged function."""
        mock_mkdir.return_value = None
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax

        # Call function with patch for rearranger
        # Patch the actual import path used in the module
        with (
            patch(
                "copul.schur_order.schur_visualizer.CISRearranger"
            ) as mock_rearranger,
            patch(
                "copul.schur_order.schur_visualizer.SchurVisualizer"
            ) as mock_visualizer,
            patch("copul.schur_order.schur_visualizer.BivCheckPi") as mock_biv_check_pi,
        ):
            # Set up rearranger mock
            mock_rearranger_instance = MagicMock()
            # Return a callable that won't be used for its sum() method
            rearranged_func = MagicMock()
            rearranged_func.side_effect = lambda u, v: u * v
            mock_rearranger_instance.rearrange_copula.return_value = rearranged_func
            mock_rearranger.return_value = mock_rearranger_instance

            # Set up SchurVisualizer mock
            mock_visualizer_instance = MagicMock()
            mock_visualizer_instance.compute.return_value = np.linspace(0, 1, 10)
            mock_visualizer.return_value = mock_visualizer_instance

            # Set up BivCheckPi mock
            mock_check_pi_instance = MagicMock()
            mock_biv_check_pi.return_value = mock_check_pi_instance

            # Mock copula without __call__ in the spec
            mock_copula_instance = MagicMock()
            mock_copula = MagicMock(spec=["params", "__str__", "__name__"])
            # Configure the call method separately
            mock_copula.return_value = mock_copula_instance
            mock_copula.params = [sp.symbols("theta")]
            mock_copula.__str__.return_value = "MockCopula"
            mock_copula.__name__ = "MockCopula"

            # Call the function to test
            visualize_rearranged(mock_copula, [1.5, 2.0], 0.3, grid_size=5)

            # Verify
            mock_rearranger.assert_called_once_with(5)
            mock_ax.plot.assert_called()
            mock_savefig.assert_called()
            mock_show.assert_called()
            mock_close.assert_called()
