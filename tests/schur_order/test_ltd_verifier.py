import numpy as np
import pytest
from copul.checkerboard.biv_check_pi import BivCheckPi


@pytest.mark.parametrize(
    "matr, expected",
    [
        (  # LTD matrix
            np.array(
                [
                    [3, 0, 0],
                    [0, 1, 2],
                    [0, 2, 1],
                ]
            ),
            True,
        ),
        (  # Non-LTD matrix
            np.array(
                [
                    [4, 0, 0.0],
                    [0, 1, 3],
                    [0.0, 3, 1],
                ]
            ),
            False,
        ),
    ],
)
def test_checkerboards(matr, expected):
    cb = BivCheckPi(matr)
    result = cb.is_ltd()
    assert (
        result == expected
    ), f"Expected {expected}, but got {result} for matrix:\n{matr}"
