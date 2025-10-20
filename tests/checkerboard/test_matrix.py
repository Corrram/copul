from copul.checkerboard.matrix import from_matrix


def test_check_pi_matrix():
    ccop = from_matrix([[1, 0], [0, 1]])
    assert ccop.matr.tolist() == [[0.5, 0], [0, 0.5]]
    assert ccop.__class__.__name__ == "BivCheckPi"

    ccop = from_matrix([[1, 0], [0, 1]], "CheckMin")
    assert ccop.matr.tolist() == [[0.5, 0], [0, 0.5]]
    assert ccop.__class__.__name__ == "BivCheckMin"
