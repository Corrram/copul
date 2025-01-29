import numpy as np
import pytest

from copul.schur_order.cis_rearranger import CISRearranger


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
    matr_sum = np.sum(test_matr)
    cisr = CISRearranger()
    rearranged = cisr.rearrange_checkerboard(test_matr)
    result_matr = rearranged * matr_sum
    assert np.allclose(result_matr, expected)
