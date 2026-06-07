import numpy as np
import pytest
from copul.checkerboard.biv_check_pi import BivCheckPi
from copul.schur_order.ltd_verifier import LTDVerifier


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
    assert result == expected, (
        f"Expected {expected}, but got {result} for matrix:\n{matr}"
    )


@pytest.mark.parametrize(
    "matr, expected",
    [
        (
            np.ones((3, 3)),
            {
                "is_ltd": True,
                "is_lti": True,
                "is_rti": True,
                "is_rtd": True,
            },
        ),
        (
            np.eye(3),
            {
                "is_ltd": True,
                "is_lti": False,
                "is_rti": True,
                "is_rtd": False,
            },
        ),
        (
            np.fliplr(np.eye(3)),
            {
                "is_ltd": False,
                "is_lti": True,
                "is_rti": False,
                "is_rtd": True,
            },
        ),
    ],
)
def test_tail_monotonicity_verifier_methods(matr, expected):
    cb = BivCheckPi(matr)
    verifier = LTDVerifier()

    actual = {
        "is_ltd": verifier.is_ltd(cb),
        "is_lti": verifier.is_lti(cb),
        "is_rti": verifier.is_rti(cb),
        "is_rtd": verifier.is_rtd(cb),
    }

    assert actual == expected


def test_checkerboard_tail_monotonicity_methods_forward_to_verifier():
    cb = BivCheckPi(np.eye(3))

    assert cb.is_ltd() is True
    assert cb.is_lti() is False
    assert cb.is_rti() is True
    assert cb.is_rtd() is False
