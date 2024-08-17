from copul import Nelsen12


def test_nelsen12():
    nelsen = Nelsen12(0.5)
    inv_gen = nelsen.inv_generator(0.5)
    assert inv_gen == 0.8
