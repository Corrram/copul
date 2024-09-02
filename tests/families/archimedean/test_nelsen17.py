from copul import Nelsen17


def test_nelsen17():
    copula = Nelsen17()
    copula1 = copula(1)
    assert [*copula.intervals] == ["theta"]
    assert [*copula1.intervals] == []
