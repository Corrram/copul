import numpy as np
import pytest

from copul.family.other.independence_copula import IndependenceCopula


class TestIndependenceCopula:
    """Test class for IndependenceCopula"""

    @pytest.fixture
    def copula(self):
        """Create a test copula instance"""
        return IndependenceCopula(dimension=3)

    def test_init(self):
        """Test initialization with different dimensions"""
        # Test default dimension
        copula1 = IndependenceCopula()
        assert copula1.dim == 2

        # Test custom dimension
        copula2 = IndependenceCopula(dimension=4)
        assert copula2.dim == 4
        assert len(copula2.u_symbols) == 4

    def test_call_method(self, copula):
        """Test __call__ method"""
        new_copula = copula()
        assert new_copula.dim == copula.dim
        assert new_copula is not copula  # Should be a new instance

    def test_is_absolutely_continuous(self, copula):
        """Test is_absolutely_continuous property"""
        assert copula.is_absolutely_continuous is True

    def test_is_symmetric(self, copula):
        """Test is_symmetric property"""
        assert copula.is_symmetric is True

    def test_cdf(self, copula):
        """Test CDF computation"""
        # Test boundary cases
        assert float(copula.cdf(0, 0.5, 0.8)) == 0
        assert float(copula.cdf(0.5, 0, 0.8)) == 0
        assert float(copula.cdf(0.5, 0.8, 0)) == 0
        assert float(copula.cdf(1, 1, 1)) == 1

        # Test interior point
        assert abs(float(copula.cdf(0.3, 0.4, 0.5)) - (0.3 * 0.4 * 0.5)) < 1e-10

    def test_cdf_vectorized(self, copula):
        """Test vectorized CDF computation"""
        # Create test data
        u = np.array([0, 0.3, 0.7, 1])
        v = np.array([0, 0.4, 0.6, 1])
        w = np.array([0, 0.5, 0.8, 1])

        # Compute vectorized CDF
        cdf_values = copula.cdf_vectorized(u, v, w)

        # Check shape
        assert cdf_values.shape == u.shape

        # Check values
        expected_values = u * v * w
        np.testing.assert_allclose(cdf_values, expected_values)

        # Test wrong number of arguments
        with pytest.raises(ValueError):
            copula.cdf_vectorized(u, v)  # Missing one argument

    def test_pdf(self, copula):
        """Test PDF computation"""
        # PDF should be constant 1 everywhere
        for values in [(0.1, 0.2, 0.3), (0.5, 0.5, 0.5), (0.9, 0.8, 0.7)]:
            point = copula.pdf(u1=values[0], u2=values[1], u3=values[2])
            assert float(point) == 1

    def test_pdf_vectorized(self, copula):
        """Test vectorized PDF computation"""
        # Create test data
        u = np.array([0.1, 0.3, 0.7, 0.9])
        v = np.array([0.2, 0.4, 0.6, 0.8])
        w = np.array([0.3, 0.5, 0.7, 0.9])

        # Compute vectorized PDF
        pdf_values = copula.pdf_vectorized(u, v, w)

        # Check shape
        assert pdf_values.shape == u.shape

        # Check values (all should be 1)
        np.testing.assert_allclose(pdf_values, np.ones_like(u))

    def test_cond_distr(self, copula):
        """Test conditional distribution computation"""
        # Test first variable
        cond_1 = float(copula.cond_distr(1, [0.3, 0.4, 0.5]))
        assert abs(cond_1 - 0.2) < 1e-10

        # Test second variable
        cond_2 = float(copula.cond_distr(2, [0.3, 0.4, 0.5]))
        assert abs(cond_2 - 0.15) < 1e-10

        # Test third variable
        cond_3 = float(copula.cond_distr(3, [0.3, 0.4, 0.5]))
        assert abs(cond_3 - 0.3 * 0.4) < 1e-10

        # Test invalid index
        with pytest.raises(ValueError):
            copula.cond_distr(4, [0.3, 0.4, 0.5])  # Index out of bounds

        with pytest.raises(ValueError):
            copula.cond_distr(0, [0.3, 0.4, 0.5])  # Index out of bounds

    def test_dependence_measures(self, copula):
        """Test all dependence measures"""
        # All dependence measures should be 0 for independence
        assert copula.kendalls_tau() == 0
        assert copula.spearmans_rho() == 0
        assert copula.lambda_L() == 0
        assert copula.lambda_U() == 0

    def test_rvs(self, copula):
        """Test random sample generation"""
        # Test sample shape
        samples = copula.rvs(100)
        assert samples.shape == (100, 3)

        # All values should be in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

        # Test with random state for reproducibility
        samples1 = copula.rvs(10, random_state=42)
        samples2 = copula.rvs(10, random_state=42)
        np.testing.assert_array_equal(samples1, samples2)


def test_independence_cdf_formula():
    """Test that the CDF formula is correct for various dimensions"""
    # Test 2D
    copula2d = IndependenceCopula(dimension=2)
    assert abs(float(copula2d.cdf(0.3, 0.4)) - (0.3 * 0.4)) < 1e-10

    # Test 3D
    copula3d = IndependenceCopula(dimension=3)
    assert abs(float(copula3d.cdf(0.3, 0.4, 0.5)) - (0.3 * 0.4 * 0.5)) < 1e-10

    # Test 4D
    copula4d = IndependenceCopula(dimension=4)
    assert (
        abs(float(copula4d.cdf(0.3, 0.4, 0.5, 0.6)) - (0.3 * 0.4 * 0.5 * 0.6)) < 1e-10
    )


def test_independence_properties():
    """Test independence-specific properties of the copula"""
    # Create copulas of different dimensions
    dimensions = [2, 3, 4, 5]

    for dim in dimensions:
        copula = IndependenceCopula(dimension=dim)

        # Parameters should be empty
        assert len(copula.params) == 0
        assert copula.intervals == {}

        # Both symmetry properties should be true
        assert copula.is_symmetric is True
        assert copula.is_absolutely_continuous is True

        # All dependence measures should be 0
        assert copula.kendalls_tau() == 0
        assert copula.spearmans_rho() == 0
        assert copula.lambda_L() == 0
        assert copula.lambda_U() == 0
