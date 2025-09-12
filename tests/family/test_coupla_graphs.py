"""
Tests for the CopulaGraphs class.
"""

import pytest
from unittest.mock import MagicMock

from copul.family.copula_graphs import CopulaGraphs


class TestCopulaGraphs:
    """Tests for the CopulaGraphs class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create mock copulas
        self.mock_copula_with_suffix = MagicMock()
        type(self.mock_copula_with_suffix).__name__ = "GaussianCopula"
        self.mock_copula_with_suffix.intervals = ["rho"]
        self.mock_copula_with_suffix.rho = 0.5

        self.mock_copula_without_suffix = MagicMock()
        type(self.mock_copula_without_suffix).__name__ = "Clayton"
        self.mock_copula_without_suffix.intervals = ["theta"]
        self.mock_copula_without_suffix.theta = 1.5

        self.mock_copula_multiple_params = MagicMock()
        type(self.mock_copula_multiple_params).__name__ = "StudentTCopula"
        self.mock_copula_multiple_params.intervals = ["rho", "nu"]
        self.mock_copula_multiple_params.rho = 0.3
        self.mock_copula_multiple_params.nu = 4

        self.mock_copula_no_params = MagicMock()
        type(self.mock_copula_no_params).__name__ = "IndependenceCopula"
        self.mock_copula_no_params.intervals = []

    def test_initialization(self):
        """Test initialization of CopulaGraphs."""
        # Test with default add_params
        graph1 = CopulaGraphs(self.mock_copula_with_suffix)
        assert graph1._copula == self.mock_copula_with_suffix
        assert graph1._add_params is True

        # Test with add_params=False
        graph2 = CopulaGraphs(self.mock_copula_without_suffix, add_params=False)
        assert graph2._copula == self.mock_copula_without_suffix
        assert graph2._add_params is False

    def test_get_copula_title_with_suffix_and_params(self):
        """Test get_copula_title with a copula name that already has 'Copula' suffix and with parameters."""
        graph = CopulaGraphs(self.mock_copula_with_suffix)
        title = graph.get_copula_title()

        # Should be "GaussianCopula (rho=0.5)"
        assert "GaussianCopula" in title
        assert "rho=0.5" in title
        assert title == "GaussianCopula (rho=0.5)"

    def test_get_copula_title_without_suffix_with_params(self):
        """Test get_copula_title with a copula name that doesn't have 'Copula' suffix but with parameters."""
        graph = CopulaGraphs(self.mock_copula_without_suffix)
        title = graph.get_copula_title()

        # Should be "Clayton Copula (theta=1.5)"
        assert "Clayton Copula" in title
        assert "theta=1.5" in title
        assert title == "Clayton Copula (theta=1.5)"

    def test_get_copula_title_with_suffix_no_params(self):
        """Test get_copula_title with 'Copula' suffix but add_params=False."""
        graph = CopulaGraphs(self.mock_copula_with_suffix, add_params=False)
        title = graph.get_copula_title()

        # Should be just "GaussianCopula" without parameters
        assert title == "GaussianCopula"
        assert "rho" not in title

    def test_get_copula_title_without_suffix_no_params(self):
        """Test get_copula_title without 'Copula' suffix and add_params=False."""
        graph = CopulaGraphs(self.mock_copula_without_suffix, add_params=False)
        title = graph.get_copula_title()

        # Should be "Clayton Copula" without parameters
        assert title == "Clayton Copula"
        assert "theta" not in title

    def test_get_copula_title_multiple_params(self):
        """Test get_copula_title with multiple parameters."""
        graph = CopulaGraphs(self.mock_copula_multiple_params)
        title = graph.get_copula_title()

        # Should include both parameters
        assert "StudentTCopula" in title
        assert "rho=0.3" in title
        assert "nu=4" in title
        # Parameters should be comma-separated
        assert ", " in title

    def test_get_copula_title_no_params_available(self):
        """Test get_copula_title with a copula that has no parameters."""
        graph = CopulaGraphs(self.mock_copula_no_params)
        title = graph.get_copula_title()

        # Should be just the copula name
        assert title == "IndependenceCopula"

    def test_get_copula_title_empty_intervals(self):
        """Test get_copula_title with empty intervals list but add_params=True."""
        mock_copula = MagicMock()
        type(mock_copula).__name__ = "FrankCopula"
        mock_copula.intervals = []  # Empty intervals

        graph = CopulaGraphs(mock_copula)
        title = graph.get_copula_title()

        # Should be just the copula name without parameter parentheses
        assert title == "FrankCopula"
        assert "(" not in title

    def test_get_copula_title_with_float_parameter(self):
        """Test get_copula_title with a parameter that is a float."""
        mock_copula = MagicMock()
        type(mock_copula).__name__ = "GumbelCopula"
        mock_copula.intervals = ["alpha"]
        mock_copula.alpha = 2.718  # Float value

        graph = CopulaGraphs(mock_copula)
        title = graph.get_copula_title()

        # Should display the float parameter correctly
        assert "GumbelCopula" in title
        assert "alpha=2.718" in title

    def test_get_copula_title_with_int_parameter(self):
        """Test get_copula_title with a parameter that is an integer."""
        mock_copula = MagicMock()
        type(mock_copula).__name__ = "DiscreteModel"
        mock_copula.intervals = ["k"]
        mock_copula.k = 3  # Integer value

        graph = CopulaGraphs(mock_copula, add_params=True)
        title = graph.get_copula_title()

        # Should display the integer parameter and add "Copula" suffix
        assert "DiscreteModel Copula" in title
        assert "k=3" in title

    def test_get_copula_title_parameter_ordering(self):
        """Test that parameters appear in the same order as in intervals."""
        mock_copula = MagicMock()
        type(mock_copula).__name__ = "ComplexCopula"
        mock_copula.intervals = ["c", "a", "b"]  # Order matters
        mock_copula.a = 1
        mock_copula.b = 2
        mock_copula.c = 3

        graph = CopulaGraphs(mock_copula)
        title = graph.get_copula_title()

        # Parameters should appear in the same order as in intervals
        param_part = title.split("(")[1].split(")")[0]
        assert param_part == "c=3, a=1, b=2"


@pytest.mark.parametrize(
    "copula_name, intervals, params, add_params, expected",
    [
        ("GaussianCopula", ["rho"], {"rho": 0.7}, True, "GaussianCopula (rho=0.7)"),
        ("Clayton", ["theta"], {"theta": 2.5}, True, "Clayton Copula (theta=2.5)"),
        ("FrankCopula", [], {}, True, "FrankCopula"),
        ("GumbelCopula", ["alpha"], {"alpha": 1.5}, False, "GumbelCopula"),
        (
            "StudentCopula",
            ["rho", "nu"],
            {"rho": 0.5, "nu": 3},
            True,
            "StudentCopula (rho=0.5, nu=3)",
        ),
    ],
)
def test_copula_title_parametrized(
    copula_name, intervals, params, add_params, expected
):
    """Parametrized test for get_copula_title with various inputs."""
    mock_copula = MagicMock()
    type(mock_copula).__name__ = copula_name
    mock_copula.intervals = intervals

    # Set attributes based on params dict
    for param, value in params.items():
        setattr(mock_copula, param, value)

    graph = CopulaGraphs(mock_copula, add_params=add_params)
    title = graph.get_copula_title()

    assert title == expected
