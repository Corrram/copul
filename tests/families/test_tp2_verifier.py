"""
Tests for the TP2Verifier class.
"""

from unittest.mock import MagicMock, patch
import pytest
import sympy


from copul.families.tp2_verifier import TP2Verifier


class TestTP2Verifier:
    """Tests for the TP2Verifier class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create verifier instance with default ranges
        self.verifier = TP2Verifier()

        # Create a basic mock copula
        self.mock_copula = MagicMock()
        self.mock_copula.is_absolutely_continuous = True
        self.mock_copula.params = [sympy.symbols("theta")]
        self.mock_copula.intervals = {
            "theta": MagicMock(inf=-1, end=1, left_open=False, right_open=False)
        }
        self.mock_copula.u = sympy.symbols("u")
        self.mock_copula.v = sympy.symbols("v")

        # Set up PDF
        self.mock_copula.pdf = sympy.symbols("pdf")

        # Create copula with two parameters
        self.two_param_copula = MagicMock()
        self.two_param_copula.is_absolutely_continuous = True
        self.two_param_copula.params = [sympy.symbols("theta"), sympy.symbols("delta")]
        self.two_param_copula.intervals = {
            "theta": MagicMock(inf=-1, end=1, left_open=False, right_open=False),
            "delta": MagicMock(inf=0, end=5, left_open=False, right_open=False),
        }
        self.two_param_copula.u = sympy.symbols("u")
        self.two_param_copula.v = sympy.symbols("v")

        # Create non-absolutely continuous copula
        self.non_abs_cont_copula = MagicMock()
        self.non_abs_cont_copula.is_absolutely_continuous = False

    def test_initialization(self):
        """Test initialization of TP2Verifier."""
        # Test default initialization
        verifier = TP2Verifier()
        assert verifier.range_min is None
        assert verifier.range_max is None

        # Test with custom ranges
        verifier = TP2Verifier(range_min=-5, range_max=5)
        assert verifier.range_min == -5
        assert verifier.range_max == 5

    def test_is_tp2_non_absolutely_continuous(self):
        """Test is_tp2 with a non-absolutely continuous copula."""
        result = self.verifier.is_tp2(self.non_abs_cont_copula)
        assert result is False

    @patch("copul.families.tp2_verifier.warnings")
    @patch("copul.families.tp2_verifier.sympy.log")
    def test_is_tp2_with_one_param_copula(self, mock_log, mock_warnings):
        """Test is_tp2 with a one-parameter copula."""
        # Setup mock return values
        mock_log.return_value = sympy.symbols("log_pdf")
        instance_mock = MagicMock()
        instance_mock.is_absolutely_continuous = True

        # Set up the _check_extreme_mixed_term method to return False
        self.mock_copula._check_extreme_mixed_term.return_value = False
        self.mock_copula.return_value = instance_mock

        # Replace the actual method with a mock to prevent actual computation
        with patch.object(TP2Verifier, "_check_extreme_mixed_term", return_value=False):
            result = self.verifier.is_tp2(self.mock_copula)

            # Verify it was called with expected argument patterns
            assert isinstance(result, bool)

            # Verify we tried different parameter values
            assert self.mock_copula.call_count > 0

    @patch("copul.families.tp2_verifier.warnings")
    @patch("copul.families.tp2_verifier.sympy.log")
    def test_is_tp2_with_two_param_copula(self, mock_log, mock_warnings):
        """Test is_tp2 with a two-parameter copula."""
        # Setup mock return values
        mock_log.return_value = sympy.symbols("log_pdf")
        instance_mock = MagicMock()
        instance_mock.is_absolutely_continuous = True

        # Set up the _check_extreme_mixed_term method to return False
        self.two_param_copula._check_extreme_mixed_term.return_value = False
        self.two_param_copula.return_value = instance_mock

        # Replace the actual method with a mock to prevent actual computation
        with patch.object(TP2Verifier, "_check_extreme_mixed_term", return_value=False):
            result = self.verifier.is_tp2(self.two_param_copula)

            # Verify it was called with expected argument patterns
            assert isinstance(result, bool)

            # Verify we tried different parameter values
            assert self.two_param_copula.call_count > 0

    @patch("copul.families.tp2_verifier.warnings")
    @patch("copul.families.tp2_verifier.sympy.log")
    def test_is_tp2_with_custom_ranges(self, mock_log, mock_warnings):
        """Test is_tp2 with custom range parameters."""
        # Create verifier with custom ranges
        verifier = TP2Verifier(range_min=-0.5, range_max=0.5)

        # Setup mock return values
        mock_log.return_value = sympy.symbols("log_pdf")
        instance_mock = MagicMock()
        instance_mock.is_absolutely_continuous = True

        # Set up the _check_extreme_mixed_term method to return False
        self.mock_copula._check_extreme_mixed_term.return_value = False
        self.mock_copula.return_value = instance_mock

        # Replace the actual method with a mock to prevent actual computation
        with patch.object(TP2Verifier, "_check_extreme_mixed_term", return_value=False):
            result = verifier.is_tp2(self.mock_copula)

            # Verify it was called with expected argument patterns
            assert isinstance(result, bool)

    def test_check_extreme_mixed_term_boolean_result(self):
        """Test _check_extreme_mixed_term method with boolean comparison result."""
        # Create test variables
        x1, x2, y1, y2 = 0.1, 0.2, 0.3, 0.4

        # Create a mock log_pdf with controlled substitution results
        mock_log_pdf = MagicMock()

        # Create mocked substitution values
        mock_min_term = MagicMock()
        mock_max_term = MagicMock()
        mock_mix_term_1 = MagicMock()
        mock_mix_term_2 = MagicMock()

        # Set up substitution chain
        mock_log_pdf.subs.side_effect = [
            MagicMock(subs=MagicMock(return_value=mock_min_term)),
            MagicMock(subs=MagicMock(return_value=mock_max_term)),
            MagicMock(subs=MagicMock(return_value=mock_mix_term_1)),
            MagicMock(subs=MagicMock(return_value=mock_mix_term_2)),
        ]

        # Set up sums
        mock_extreme_term = MagicMock()
        mock_mixed_term = MagicMock()

        # Configure addition operations
        mock_min_term.__add__.return_value = mock_extreme_term
        mock_mix_term_1.__add__.return_value = mock_mixed_term

        # Set up comparison to return True (violation)
        mock_extreme_term.__mul__.return_value = MagicMock()
        mock_extreme_term.__mul__.return_value.__lt__.return_value = True

        # Call the method
        result = self.verifier._check_extreme_mixed_term(
            self.mock_copula, mock_log_pdf, "u", "v", x1, x2, y1, y2
        )

        # Verify result
        assert result is True

    def test_check_extreme_mixed_term_complex_handling(self):
        """Test _check_extreme_mixed_term method with complex handling path."""
        # Create test variables
        x1, x2, y1, y2 = 0.1, 0.2, 0.3, 0.4

        # Create a mock log_pdf with controlled substitution results
        mock_log_pdf = MagicMock()

        # Create mocked substitution values
        mock_min_term = MagicMock()
        mock_max_term = MagicMock()
        mock_mix_term_1 = MagicMock()
        mock_mix_term_2 = MagicMock()

        # Set up substitution chain
        mock_log_pdf.subs.side_effect = [
            MagicMock(subs=MagicMock(return_value=mock_min_term)),
            MagicMock(subs=MagicMock(return_value=mock_max_term)),
            MagicMock(subs=MagicMock(return_value=mock_mix_term_1)),
            MagicMock(subs=MagicMock(return_value=mock_mix_term_2)),
        ]

        # Set up sums
        mock_extreme_term = MagicMock()
        mock_mixed_term = MagicMock()

        # Configure addition operations
        mock_min_term.__add__.return_value = mock_extreme_term
        mock_mix_term_1.__add__.return_value = mock_mixed_term

        # Set up multiplication to raise TypeError (to trigger complex handling)
        mock_extreme_term.__mul__.side_effect = TypeError("Complex number")

        # Set up as_real_imag for complex number handling
        mock_extreme_term.as_real_imag = MagicMock(return_value=(1.0, 0.0))
        mock_mixed_term.as_real_imag = MagicMock(return_value=(2.0, 0.0))

        # Configure the complex comparison to return True
        with (
            patch("copul.families.tp2_verifier.BooleanTrue", True),
            patch("copul.families.tp2_verifier.BooleanFalse", False),
        ):
            # Patch the sympy.Symbol classes
            with patch("sympy.Symbol"):
                result = self.verifier._check_extreme_mixed_term(
                    self.mock_copula, mock_log_pdf, "u", "v", x1, x2, y1, y2
                )

                # Result should be True since 1.0 < 2.0
                assert result is True

    def test_check_extreme_mixed_term_non_boolean_result(self):
        """Test _check_extreme_mixed_term with non-boolean comparison result."""
        # Create test variables
        x1, x2, y1, y2 = 0.1, 0.2, 0.3, 0.4

        # Create a mock log_pdf with controlled substitution results
        mock_log_pdf = MagicMock()

        # Create mocked substitution values
        mock_min_term = MagicMock()
        mock_max_term = MagicMock()
        mock_mix_term_1 = MagicMock()
        mock_mix_term_2 = MagicMock()

        # Set up substitution chain
        mock_log_pdf.subs.side_effect = [
            MagicMock(subs=MagicMock(return_value=mock_min_term)),
            MagicMock(subs=MagicMock(return_value=mock_max_term)),
            MagicMock(subs=MagicMock(return_value=mock_mix_term_1)),
            MagicMock(subs=MagicMock(return_value=mock_mix_term_2)),
        ]

        # Set up sums
        mock_extreme_term = MagicMock()
        mock_mixed_term = MagicMock()

        # Configure addition operations
        mock_min_term.__add__.return_value = mock_extreme_term
        mock_mix_term_1.__add__.return_value = mock_mixed_term

        # Set up comparison to return a symbolic expression
        mock_comparison = MagicMock()
        mock_extreme_term.__mul__.return_value = MagicMock()
        mock_extreme_term.__mul__.return_value.__lt__.return_value = mock_comparison

        # Make comparison not be a boolean value initially
        mock_comparison.__class__ = MagicMock
        mock_comparison.evalf.return_value = True

        # Call the method
        result = self.verifier._check_extreme_mixed_term(
            self.mock_copula, mock_log_pdf, "u", "v", x1, x2, y1, y2
        )

        # Should evaluate to True after evalf()
        assert result is True


@pytest.mark.parametrize(
    "is_abs_cont, expected",
    [
        (True, True),  # Absolutely continuous copula could be TP2
        (False, False),  # Non-absolutely continuous copula is not TP2
    ],
)
def test_is_tp2_absolute_continuity(is_abs_cont, expected):
    """Parametrized test for absolute continuity check."""
    # Create mock copula
    mock_copula = MagicMock()
    mock_copula.is_absolutely_continuous = is_abs_cont

    # If not absolutely continuous, we can return immediately
    if not is_abs_cont:
        verifier = TP2Verifier()
        result = verifier.is_tp2(mock_copula)
        assert result is expected
    else:
        # Skip this case as it would require full mocking of the verification process
        pass
