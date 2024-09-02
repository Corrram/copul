from copul import Nelsen18


def test_nelsen18():
    copula = Nelsen18()
    cdf = copula.cdf
    cdf3 = cdf(u=3)
    free_symbols = {str(symbol) for symbol in cdf.func.free_symbols}
    free_symbols3 = {str(symbol) for symbol in cdf3.func.free_symbols}
    assert free_symbols == {"u", "v", "theta"}
    assert free_symbols3 == {"v", "theta"}
