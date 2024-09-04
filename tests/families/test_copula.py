from copul.families.copula import Copula


def test_3d_clayton():
    cdf = "(x**(-theta) + y**(-theta) + z**(-theta) - 2)**(-1/theta)"
    copula_family = Copula.from_cdf(cdf, "theta")
    copula = copula_family(0.5)
