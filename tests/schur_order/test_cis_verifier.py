import numpy as np
import pytest
from copul.checkerboard.biv_check_pi import BivCheckPi


@pytest.mark.parametrize(
    "matr, expected",
    [
        (  # CIS matrix
            np.array(
                [
                    [0.2, 0.1, 0.0],
                    [0.1, 0.3, 0.1],
                    [0.0, 0.1, 0.2],
                ]
            ),
            True,
        ),
        (  # Non-CIS matrix
            np.array(
                [
                    [0.1, 0.2, 0.0],
                    [0.2, 0.1, 0.1],
                    [0.0, 0.1, 0.2],
                ]
            ),
            False,
        ),
        (  # Another CIS matrix
            np.array(
                [
                    [0.3, 0.2],
                    [0.1, 0.4],
                ]
            ),
            True,
        ),
        (  # Another Non-CIS matrix
            np.array(
                [
                    [0.1, 0.3],
                    [0.4, 0.2],
                ]
            ),
            False,
        ),
    ],
)
def test_checkerboards(matr, expected):
    cb = BivCheckPi(matr)
    result = cb.is_cis()[0]
    assert (
        result == expected
    ), f"Expected {expected}, but got {result} for matrix:\n{matr}"
