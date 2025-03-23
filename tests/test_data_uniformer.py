import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import stats
from copul.data_uniformer import DataUniformer


class TestDataUniformer:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.uniformer = DataUniformer()
        
        # Sample data with different patterns for testing
        self.data_basic = np.array([
            [0.5, 2.0, -1.0],
            [1.0, 0.0, 3.0],
            [1.5, -1.0, 2.0],
            [2.0, 1.0, 0.0]
        ])
        
        # Data with ties for testing tie handling
        self.data_with_ties = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 2.0, 3.0],
            [3.0, 3.0, 3.0]
        ])
    
    def test_basic_transformation(self):
        """Test basic data transformation to uniform distribution."""
        transformed = self.uniformer.uniform(self.data_basic)
        
        # Check shape preservation
        assert transformed.shape == self.data_basic.shape
        
        # Check range of values
        assert np.all(transformed >= 0)
        assert np.all(transformed <= 1)
        
        # Check specific transformation using the first column
        col0 = self.data_basic[:, 0]
        expected_ranks = stats.rankdata(col0, method='average') / (len(col0) + 1)
        assert_array_almost_equal(transformed[:, 0], expected_ranks)
    
    def test_ties_handling(self):
        """Test handling of tied values."""
        transformed = self.uniformer.uniform(self.data_with_ties)
        
        # First column has two 1.0s, which should get the same rank
        col0 = self.data_with_ties[:, 0]
        expected_ranks0 = stats.rankdata(col0, method='average') / (len(col0) + 1)
        assert_array_almost_equal(transformed[:, 0], expected_ranks0)
        
        # First two values in column 0 should have same rank
        assert transformed[0, 0] == transformed[1, 0]
        
        # Second column has three 2.0s
        col1 = self.data_with_ties[:, 1]
        expected_ranks1 = stats.rankdata(col1, method='average') / (len(col1) + 1)
        assert_array_almost_equal(transformed[:, 1], expected_ranks1)
        
        # First three values in column 1 should have same rank
        assert transformed[0, 1] == transformed[1, 1] == transformed[2, 1]
        
        # Third column has three 3.0s which should all get the same rank
        assert transformed[0, 2] == transformed[2, 2] == transformed[3, 2]
    
    def test_empty_data(self):
        """Test handling of empty data."""
        # Empty 2D array (0 rows, 3 columns)
        empty_data = np.zeros((0, 3))
        transformed = self.uniformer.uniform(empty_data)
        
        # Should return an empty array with the same shape
        assert transformed.shape == empty_data.shape
        assert transformed.size == 0
    
    def test_single_row(self):
        """Test data with only one row."""
        single_row = np.array([[1.0, 2.0, 3.0]])
        transformed = self.uniformer.uniform(single_row)
        
        # With one sample, rank should be 1, then divided by (1+1)
        expected = np.ones_like(single_row) * 0.5
        assert_array_almost_equal(transformed, expected)
    
    def test_single_column(self):
        """Test data with only one column."""
        single_col = np.array([[1.0], [2.0], [3.0], [4.0]])
        transformed = self.uniformer.uniform(single_col)
        
        # Ranks should be [1, 2, 3, 4], then divided by (4+1)
        expected = np.array([[1/5], [2/5], [3/5], [4/5]])
        assert_array_almost_equal(transformed, expected)
    
    def test_constant_column(self):
        """Test handling of constant columns."""
        # Column with all the same value
        constant_data = np.array([
            [1.0, 5.0, 3.0],
            [2.0, 5.0, 4.0],
            [3.0, 5.0, 5.0],
            [4.0, 5.0, 6.0]
        ])
        
        transformed = self.uniformer.uniform(constant_data)
        
        # All values in the second column should get the same rank
        assert_array_almost_equal(
            transformed[:, 1],
            np.ones(4) * 2.5 / (4 + 1)  # average rank is 2.5, divided by n+1
        )
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        # Data with NaN values
        data_with_nan = np.array([
            [1.0, np.nan, 3.0],
            [np.nan, 2.0, 4.0],
            [3.0, 3.0, np.nan],
            [4.0, 4.0, 6.0]
        ])
        
        transformed = self.uniformer.uniform(data_with_nan)
        
        # NaNs should remain NaNs
        assert np.isnan(transformed[0, 1])
        assert np.isnan(transformed[1, 0])
        assert np.isnan(transformed[2, 2])
        
        # Non-NaN values should be transformed correctly
        # First column without NaN: [1.0, 3.0, 4.0]
        expected_col0 = np.array([1/4, np.nan, 2/4, 3/4])  # ranks ÷ (3+1)
        
        # Need to handle NaN comparison specially
        mask0 = ~np.isnan(transformed[:, 0])
        assert_array_almost_equal(transformed[mask0, 0], expected_col0[mask0])
    
    def test_extreme_values(self):
        """Test with extreme values to check numerical stability."""
        extreme_data = np.array([
            [1e-10, 1e10, -1e10],
            [2e-10, 2e10, -2e10],
            [3e-10, 3e10, -3e10],
            [4e-10, 4e10, -4e10]
        ])
        
        transformed = self.uniformer.uniform(extreme_data)
        
        # Despite extreme values, ranks should be uniformly distributed
        expected_col0 = np.array([1/5, 2/5, 3/5, 4/5])
        assert_array_almost_equal(transformed[:, 0], expected_col0)
    
    def test_integer_data(self):
        """Test with integer data."""
        int_data = np.array([
            [1, 5, 9],
            [2, 6, 8],
            [3, 7, 7],
            [4, 8, 6]
        ])
        
        transformed = self.uniformer.uniform(int_data)
        
        # Verify transformation is correct for integer data
        for j in range(int_data.shape[1]):
            col = int_data[:, j]
            expected = stats.rankdata(col, method='average') / (len(col) + 1)
            assert_array_almost_equal(transformed[:, j], expected)
    
    def test_mixed_data_types(self):
        """Test with mixed data types."""
        # Create an array with mixed types (integers and floats)
        mixed_data = np.array([
            [1, 5.5, 9],
            [2.2, 6, 8.8],
            [3, 7.7, 7],
            [4.4, 8, 6.6]
        ])
        
        transformed = self.uniformer.uniform(mixed_data)
        
        # Verify transformation is correct for mixed data
        for j in range(mixed_data.shape[1]):
            col = mixed_data[:, j]
            expected = stats.rankdata(col, method='average') / (len(col) + 1)
            assert_array_almost_equal(transformed[:, j], expected)
    
    def test_result_type(self):
        """Test that the result is always a float array."""
        # Test with integer input
        int_data = np.array([[1, 2], [3, 4]], dtype=int)
        transformed = self.uniformer.uniform(int_data)
        assert transformed.dtype == float
        
        # Test with mixed input
        mixed_data = np.array([[1, 2.0], [3, 4]], dtype=object)
        transformed = self.uniformer.uniform(mixed_data)
        assert transformed.dtype == float