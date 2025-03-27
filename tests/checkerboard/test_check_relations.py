import numpy as np

from copul.checkerboard.biv_check_min import BivCheckMin
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.checkerboard.biv_check_w import BivCheckW


def test_rho_example():
    """Test rho for the example matrix from the original code."""
    matr = np.array([[1, 5, 4], [5, 3, 2], [4, 2, 4]])
    ccop_min = BivCheckMin(matr)
    ccop_pi = BivCheckPi(matr)
    ccop_w = BivCheckW(matr)
    xi_min = ccop_min.xi()
    xi_pi = ccop_pi.xi()
    xi_w = ccop_w.xi()

    # Check range and expected sign (this matrix has positive dependence)
    assert xi_pi < xi_min
    assert xi_pi < xi_w
    assert np.isclose(xi_min, xi_w, atol=1e-5)
