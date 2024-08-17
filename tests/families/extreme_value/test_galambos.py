from copul import Galambos


def test_galambos():
    cop = Galambos(0.5)
    pickands = cop.pickands(0.5)
    assert pickands == 0.875
