from copul import Nelsen11


def test_nelsen11():
    nelsen = Nelsen11(0.5)
    assert nelsen.is_symmetric
