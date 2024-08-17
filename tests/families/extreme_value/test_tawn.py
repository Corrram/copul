from copul.families.extreme_value.tawn import Tawn


def test_marshall_olkin():
    cop = Tawn(1 / 3, 1, 2)
    assert cop.is_ci
