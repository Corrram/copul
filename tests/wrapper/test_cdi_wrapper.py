import unittest
from sympy import symbols, diff

# Import the wrapper classes being tested
from copul.wrapper.cdi_wrapper import CDiWrapper
from copul.wrapper.cd1_wrapper import CD1Wrapper
from copul.wrapper.cd2_wrapper import CD2Wrapper


class TestCDiWrapper(unittest.TestCase):
    """
    Tests for the CDiWrapper class for multivariate conditional distributions
    """

    def setUp(self):
        """Set up test fixtures"""
        # Define symbols for multivariate case
        self.u1, self.u2, self.u3, self.u4 = symbols("u1 u2 u3 u4", positive=True)

        # Simple product copula in different dimensions
        self.cdf_2d = self.u1 * self.u2
        self.cdf_3d = self.u1 * self.u2 * self.u3
        self.cdf_4d = self.u1 * self.u2 * self.u3 * self.u4

        # Create corresponding partial derivatives
        self.cd1_2d = diff(self.cdf_2d, self.u1)  # = u2
        self.cd2_2d = diff(self.cdf_2d, self.u2)  # = u1

        self.cd1_3d = diff(self.cdf_3d, self.u1)  # = u2*u3
        self.cd2_3d = diff(self.cdf_3d, self.u2)  # = u1*u3
        self.cd3_3d = diff(self.cdf_3d, self.u3)  # = u1*u2

        self.cd1_4d = diff(self.cdf_4d, self.u1)  # = u2*u3*u4
        self.cd2_4d = diff(self.cdf_4d, self.u2)  # = u1*u3*u4
        self.cd3_4d = diff(self.cdf_4d, self.u3)  # = u1*u2*u4
        self.cd4_4d = diff(self.cdf_4d, self.u4)  # = u1*u2*u3

    def test_initialization(self):
        """Test initialization of CDiWrapper"""
        wrapper = CDiWrapper(self.cd1_3d, i=1)
        self.assertEqual(wrapper.condition_index, 1)
        self.assertEqual(str(wrapper.func), str(self.cd1_3d))

        wrapper = CDiWrapper(self.cd2_3d, i=2)
        self.assertEqual(wrapper.condition_index, 2)
        self.assertEqual(str(wrapper.func), str(self.cd2_3d))

    def test_boundary_condition_zero(self):
        """Test boundary condition when ui = 0"""
        # Test in 2D
        wrapper = CDiWrapper(self.cd1_2d, i=1)
        result = wrapper(u1=0, u2=0.7)
        self.assertEqual(float(result.func), 0)

        wrapper = CDiWrapper(self.cd2_2d, i=2)
        result = wrapper(u1=0.7, u2=0)
        self.assertEqual(float(result.func), 0)

        # Test in 3D
        wrapper = CDiWrapper(self.cd1_3d, i=1)
        result = wrapper(u1=0, u2=0.7, u3=0.8)
        self.assertEqual(float(result.func), 0)

        wrapper = CDiWrapper(self.cd2_3d, i=2)
        result = wrapper(u1=0.7, u2=0, u3=0.8)
        self.assertEqual(float(result.func), 0)

        wrapper = CDiWrapper(self.cd3_3d, i=3)
        result = wrapper(u1=0.7, u2=0.8, u3=0)
        self.assertEqual(float(result.func), 0)

    def test_boundary_condition_one(self):
        """Test boundary condition when ui = 1"""
        # Test in 2D
        wrapper = CDiWrapper(self.cd1_2d, i=1)
        result = wrapper(u1=1, u2=0.7)
        self.assertEqual(float(result.func), 1)

        wrapper = CDiWrapper(self.cd2_2d, i=2)
        result = wrapper(u1=0.7, u2=1)
        self.assertEqual(float(result.func), 1)

        # Test in 3D
        wrapper = CDiWrapper(self.cd1_3d, i=1)
        result = wrapper(u1=1, u2=0.7, u3=0.8)
        self.assertEqual(float(result.func), 1)

        wrapper = CDiWrapper(self.cd2_3d, i=2)
        result = wrapper(u1=0.7, u2=1, u3=0.8)
        self.assertEqual(float(result.func), 1)

        wrapper = CDiWrapper(self.cd3_3d, i=3)
        result = wrapper(u1=0.7, u2=0.8, u3=1)
        self.assertEqual(float(result.func), 1)

    def test_normal_evaluation(self):
        """Test evaluation of conditional distribution at non-boundary points"""
        # Test in 2D
        wrapper = CDiWrapper(self.cd1_2d, i=1)
        result = wrapper(u1=0.5, u2=0.7)
        self.assertEqual(float(result.func), 0.7)

        wrapper = CDiWrapper(self.cd2_2d, i=2)
        result = wrapper(u1=0.7, u2=0.5)
        self.assertEqual(float(result.func), 0.7)

        # Test in 3D
        wrapper = CDiWrapper(self.cd1_3d, i=1)
        result = wrapper(u1=0.5, u2=0.7, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)

        wrapper = CDiWrapper(self.cd2_3d, i=2)
        result = wrapper(u1=0.7, u2=0.5, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)

        wrapper = CDiWrapper(self.cd3_3d, i=3)
        result = wrapper(u1=0.7, u2=0.8, u3=0.5)
        self.assertEqual(float(result.func), 0.7 * 0.8)

    def test_4d_case(self):
        """Test with 4D copula"""
        # Test boundary conditions
        wrapper = CDiWrapper(self.cd1_4d, i=1)
        result = wrapper(u1=0, u2=0.7, u3=0.8, u4=0.9)
        self.assertEqual(float(result.func), 0)

        result = wrapper(u1=1, u2=0.7, u3=0.8, u4=0.9)
        self.assertEqual(float(result.func), 1)

        # Test normal evaluation
        result = wrapper(u1=0.5, u2=0.7, u3=0.8, u4=0.9)
        self.assertEqual(float(result.func), 0.7 * 0.8 * 0.9)

        # Test with a different condition index
        wrapper = CDiWrapper(self.cd3_4d, i=3)
        result = wrapper(u1=0.7, u2=0.8, u3=0.5, u4=0.9)
        self.assertEqual(float(result.func), 0.7 * 0.8 * 0.9)


class TestCD1CD2Compatibility(unittest.TestCase):
    """
    Tests to ensure CD1Wrapper and CD2Wrapper are compatible with legacy code
    and work correctly with both bivariate and multivariate cases.
    """

    def setUp(self):
        """Set up test fixtures"""
        self.u, self.v = symbols("u v", positive=True)
        self.cdf_biv = self.u * self.v
        self.cd1_biv = diff(self.cdf_biv, self.u)  # = v
        self.cd2_biv = diff(self.cdf_biv, self.v)  # = u

        self.u1, self.u2, self.u3 = symbols("u1 u2 u3", positive=True)
        self.cdf_multi = self.u1 * self.u2 * self.u3
        self.cd1_multi = diff(self.cdf_multi, self.u1)  # = u2*u3
        self.cd2_multi = diff(self.cdf_multi, self.u2)  # = u1*u3

    def test_cd1_bivariate(self):
        """Test CD1Wrapper with bivariate case"""
        wrapper = CD1Wrapper(self.cd1_biv)

        # Test boundary conditions
        result = wrapper(u=0, v=0.7)
        self.assertEqual(float(result.func), 0.7)

        result = wrapper(u=1, v=0.7)
        self.assertEqual(float(result.func), 0.7)

        # Test normal evaluation
        result = wrapper(u=0.5, v=0.7)
        self.assertEqual(float(result.func), 0.7)

    def test_cd2_bivariate(self):
        """Test CD2Wrapper with bivariate case"""
        wrapper = CD2Wrapper(self.cd2_biv)

        # Test boundary conditions
        result = wrapper(u=0.7, v=0)
        self.assertEqual(float(result.func), 0.7)

        result = wrapper(u=0.7, v=1)
        self.assertEqual(float(result.func), 0.7)

        # Test normal evaluation
        result = wrapper(u=0.7, v=0.5)
        self.assertEqual(float(result.func), 0.7)

    def test_cd1_multivariate(self):
        """Test CD1Wrapper with multivariate case"""
        wrapper = CD1Wrapper(self.cd1_multi)

        # Test boundary conditions
        result = wrapper(u1=0, u2=0.7, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)

        result = wrapper(u1=1, u2=0.7, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)

        # Test normal evaluation
        result = wrapper(u1=0.5, u2=0.7, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)

    def test_cd2_multivariate(self):
        """Test CD2Wrapper with multivariate case"""
        wrapper = CD2Wrapper(self.cd2_multi)

        # Test boundary conditions
        result = wrapper(u1=0.7, u2=0, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)

        result = wrapper(u1=0.7, u2=1, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)

        # Test normal evaluation
        result = wrapper(u1=0.7, u2=0.5, u3=0.8)
        self.assertEqual(float(result.func), 0.7 * 0.8)
