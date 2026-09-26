import numpy as np
import pytest
from scipy import stats

from copul.chatterjee import (
    test_independence,
    xi_n_with_ci,
    xi_ncalculate,
    xi_null_variance,
    xi_nvarcalculate,
)


class TestXiNCalculate:
    """Tests for the xi_ncalculate function."""

    def test_perfect_positive_correlation(self):
        """Test Xi_n with perfectly positively correlated data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = xi_ncalculate(x, y)
        # 1 - 3 / (n + 1) = 0.5 for n = 5 (tends to 1 for large n)
        assert result == pytest.approx(0.5)

    def test_perfect_negative_correlation(self):
        """Test Xi_n with perfectly negatively correlated data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        result = xi_ncalculate(x, y)
        # Based on actual behavior, Xi_n for perfect negative correlation is also 0.5
        assert result == pytest.approx(0.5)

    def test_no_correlation(self):
        """Test Xi_n with uncorrelated data."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)
        result = xi_ncalculate(x, y)
        # For independent data, Xi_n should be close to 0
        assert -0.3 < result < 0.3

    def test_linear_correlation(self):
        """Test Xi_n with linearly correlated data."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = 2 * x + 1 + np.random.normal(0, 0.1, 100)
        result = xi_ncalculate(x, y)
        # For strongly correlated data, Xi_n should be high
        assert result > 0.7

    def test_nonlinear_correlation(self):
        """Test Xi_n with nonlinearly correlated data."""
        np.random.seed(42)
        x = np.random.rand(100)
        y = x**2 + np.random.normal(0, 0.05, 100)
        result = xi_ncalculate(x, y)
        # Should detect non-linear dependence
        assert result > 0.5

    def test_periodic_relationship(self):
        """Test Xi_n with periodic relationship."""
        np.random.seed(42)
        x = np.linspace(0, 4 * np.pi, 100)
        y = np.sin(x) + np.random.normal(0, 0.1, 100)
        result = xi_ncalculate(x, y)
        # Should detect periodic dependence
        assert result > 0.3

    def test_constant_data(self):
        """Test Xi_n with constant data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([7, 7, 7, 7, 7])  # Constant y
        result = xi_ncalculate(x, y)
        # Chatterjee's xi_n is undefined for constant y (0/0)
        assert np.isnan(result)

        # Constant x
        x_const = np.array([3, 3, 3, 3, 3])
        y_var = np.array([1, 2, 3, 4, 5])
        # When x is constant, all ranks are tied
        result_const_x = xi_ncalculate(x_const, y_var)
        # Just verify it runs without error and returns a numeric value
        assert isinstance(result_const_x, (int, float, np.number))

    def test_different_length_vectors(self):
        """Different lengths are an error (used to be silently truncated)."""
        with pytest.raises(ValueError, match="same length"):
            xi_ncalculate(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))

    def test_nan_values_raise(self):
        """NaN input is an error (used to return 0.5)."""
        with pytest.raises(ValueError, match="NaN"):
            xi_ncalculate([1, np.nan, 3], [1, 2, 3])

    def test_empty_vectors(self):
        """Test Xi_n with empty vectors."""
        x = np.array([])
        y = np.array([])

        # Test if function returns NaN for empty vectors
        result = xi_ncalculate(x, y)
        assert np.isnan(result)

    def test_single_element_vectors(self):
        """Test Xi_n with single-element vectors."""
        x = np.array([1])
        y = np.array([2])

        # Test if function returns NaN for single-element vectors
        result = xi_ncalculate(x, y)
        assert np.isnan(result)


class TestXiNVarCalculate:
    """Tests for the xi_nvarcalculate function."""

    def test_variance_always_nonnegative(self):
        """Test that variance is always non-negative."""
        np.random.seed(42)

        # Test with various data types
        data_pairs = [
            (
                np.array([1, 2, 3, 4, 5]),
                np.array([1, 2, 3, 4, 5]),
            ),  # Perfect correlation
            (
                np.array([1, 2, 3, 4, 5]),
                np.array([5, 4, 3, 2, 1]),
            ),  # Perfect negative correlation
            (np.random.rand(100), np.random.rand(100)),  # Uncorrelated
            (
                np.random.rand(100),
                2 * np.random.rand(100) + np.random.normal(0, 0.1, 100),
            ),  # Correlated with noise
        ]

        for x, y in data_pairs:
            result = xi_nvarcalculate(x, y)
            assert result >= 0, f"Variance is negative: {result}"

    def test_variance_calculation(self):
        """Test that variance calculation produces reasonable values."""
        np.random.seed(42)

        # Create datasets of different sizes
        sizes = [20, 50, 100, 200]

        for size in sizes:
            x = np.random.rand(size)
            y = x + np.random.normal(0, 0.2, size)  # Correlated data
            variance = xi_nvarcalculate(x, y)

            # Just verify it produces a reasonable non-negative value
            assert variance >= 0
            assert isinstance(variance, (int, float, np.number))

    def test_different_length_vectors(self):
        """Different lengths are an error."""
        with pytest.raises(ValueError, match="same length"):
            xi_nvarcalculate(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))

    def test_nan_values(self):
        """NaN input is an error."""
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="NaN"):
            xi_nvarcalculate(x, y)

    def test_matches_quadratic_reference(self):
        """The O(n log n) implementation equals the original O(n^2) sums."""
        rng = np.random.default_rng(3)
        for n in [3, 7, 40]:
            x = rng.random(n)
            y = x + rng.normal(0, 0.4, n)
            ref = _xi_var_reference(x, y)
            assert xi_nvarcalculate(x, y) == pytest.approx(ref, abs=1e-12)


class TestComparisonWithOtherMeasures:
    """Tests comparing Xi_n with other correlation measures."""

    def test_compare_with_pearson(self):
        """Compare Xi_n with Pearson correlation for different relationships."""
        np.random.seed(42)
        n = 100

        # Linear relationship
        x_linear = np.random.rand(n)
        y_linear = 2 * x_linear + 1 + np.random.normal(0, 0.1, n)

        xi_linear = xi_ncalculate(x_linear, y_linear)
        pearson_linear = stats.pearsonr(x_linear, y_linear)[0]

        # Both should detect strong linear dependence
        assert xi_linear > 0.7
        assert pearson_linear > 0.7

        # Non-monotonic relationship (sine wave)
        x_sine = np.linspace(-np.pi, np.pi, n)
        y_sine = np.sin(x_sine) + np.random.normal(0, 0.1, n)

        xi_sine = xi_ncalculate(x_sine, y_sine)
        pearson_sine = stats.pearsonr(x_sine, y_sine)[0]

        # Xi_n should detect non-monotonic dependence better than Pearson
        # Since a full sine wave has zero linear correlation
        assert abs(xi_sine) > abs(pearson_sine)

    def test_compare_with_spearman(self):
        """Compare Xi_n with Spearman rank correlation for different relationships."""
        np.random.seed(42)
        n = 100

        # Monotonic but non-linear relationship
        x_nonlin = np.random.rand(n)
        y_nonlin = x_nonlin**3 + np.random.normal(0, 0.05, n)

        xi_nonlin = xi_ncalculate(x_nonlin, y_nonlin)
        spearman_nonlin = stats.spearmanr(x_nonlin, y_nonlin)[0]

        # Both should detect monotonic non-linear dependence
        assert xi_nonlin > 0.5
        assert spearman_nonlin > 0.5

        # Non-monotonic relationship
        x_parabola = np.linspace(-1, 1, n)
        y_parabola = x_parabola**2 + np.random.normal(0, 0.05, n)

        xi_parabola = xi_ncalculate(x_parabola, y_parabola)
        spearman_parabola = stats.spearmanr(x_parabola, y_parabola)[0]

        # Xi_n should potentially detect non-monotonic dependence better than Spearman
        # For a parabola centered at 0, Spearman should be close to 0
        assert abs(xi_parabola) > abs(spearman_parabola)


def _xi_var_reference(xvec, yvec):
    """Original quadratic-time variance estimator (for regression testing)."""
    n = len(xvec)
    yrank = (np.argsort(np.argsort(yvec)) + 1)[np.argsort(xvec)]
    yrank1 = np.concatenate((yrank[1:n], [yrank[n - 1]]))
    yrank2 = np.concatenate((yrank[2:n], [yrank[n - 1]] * 2))
    yrank3 = np.concatenate((yrank[3:n], [yrank[n - 1]] * 3))
    term1 = np.minimum(yrank, yrank1)
    term2 = np.minimum(yrank, yrank2)
    term3 = np.minimum(yrank2, yrank3)
    term4 = np.array([np.sum(yrank[i] <= term1[np.arange(n) != i]) for i in range(n)], float)
    term5 = np.minimum(yrank1, yrank2)
    s6 = np.array([np.sum(np.minimum(term1[i], term1[np.arange(n) != i])) for i in range(n)])
    v = 36 * (
        np.mean((term1 / n) ** 2)
        + 2 * np.mean(term1 * term2 / n**2)
        - 2 * np.mean(term1 * term3 / n**2)
        + 4 * np.mean(term4 * term1 / (n * (n - 1)))
        - 2 * np.mean(term4 * term5 / (n * (n - 1)))
        + np.mean(s6 / (n * (n - 1)))
        - 4 * np.mean(term1 / n) ** 2
    )
    return max(0.0, v)


def _xi_ties_reference(x, y):
    """Chatterjee (2021) definition with ties in y (x without ties)."""
    n = len(x)
    ys = np.asarray(y)[np.argsort(x)]
    r = np.array([np.sum(y <= yi) for yi in ys])
    l_ = np.array([np.sum(y >= yi) for yi in ys])
    return 1 - n * np.sum(np.abs(np.diff(r))) / (2 * np.sum(l_ * (n - l_)))


class TestChatterjeeInference:
    def test_ties_in_y_follow_chatterjee_definition(self):
        rng = np.random.default_rng(0)
        x = rng.random(200)
        y = np.floor(4 * x + rng.random(200))  # heavy ties
        assert xi_ncalculate(x, y) == pytest.approx(_xi_ties_reference(x, y))

    def test_ties_in_x_are_broken_randomly_but_reproducibly(self):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 3, 300)
        y = rng.random(300)
        a = xi_ncalculate(x, y, random_state=5)
        assert a == xi_ncalculate(x, y, random_state=5)
        vals = {xi_ncalculate(x, y, random_state=s) for s in range(5)}
        assert len(vals) > 1

    def test_confidence_interval_uses_sqrt_n(self):
        rng = np.random.default_rng(2)
        x = rng.random(4000)
        y = x + rng.normal(0, 0.3, 4000)
        xi, (lo, hi) = xi_n_with_ci(x, y)
        se = np.sqrt(xi_nvarcalculate(x, y) / 4000)
        assert (lo, hi) != (0.0, 1.0)
        assert hi - lo == pytest.approx(2 * 1.959963984540054 * se, rel=1e-9)
        assert lo < xi < hi and hi - lo < 0.1

    def test_independence_test_uses_null_variance(self):
        rng = np.random.default_rng(3)
        n = 1000
        x, y = rng.random(n), rng.random(n)
        xi, p, reject = test_independence(x, y)
        expected = 1 - stats.norm.cdf(np.sqrt(n) * xi / np.sqrt(0.4))
        assert p == pytest.approx(expected)
        x2 = rng.random(n)
        _, p2, reject2 = test_independence(x2, np.sin(8 * x2))
        assert p2 < 1e-10 and reject2

    def test_null_variance_with_ties(self):
        assert xi_null_variance(np.arange(10.0)) == 0.4
        y = np.repeat([0.0, 1.0, 2.0], 50)
        v = xi_null_variance(y)
        assert 0.4 < v < 1.0
