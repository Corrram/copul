from copul.checkerboard.check_pi import CheckPi
from copul.checkerboard.check_min import CheckMin
from copul.checkerboard.biv_check_w import BivCheckW


def from_matrix(matrix, checkerboard_type="CheckPi"):
    if checkerboard_type == "CheckPi":
        return CheckPi(matrix)
    elif checkerboard_type == "CheckMin":
        return CheckMin(matrix)
    elif checkerboard_type == "CheckW":
        return BivCheckW(matrix)
    else:
        raise ValueError("Invalid checkerboard type")
