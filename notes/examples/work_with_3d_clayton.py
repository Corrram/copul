import copul
from copul.family.archimedean.multivariate_clayton import MultivariateClayton


def main():
    # func_2d = "(x**(-theta) + y**(-theta) - 1)**(-1/theta)"
    # copul.from_cdf(func_2d)
    # copula_2d = copulas_2d(theta=3)
    # copula_2d.scatter_plot()

    clayton = MultivariateClayton(theta=10, dimension=2)
    clayton.scatter_plot(n=1_000, approximate=True)
    clayton = MultivariateClayton(theta=0.1, dimension=3)
    clayton.scatter_plot(n=1_000, approximate=True)

    func = "(x**(-theta) + y**(-theta) + z**(-theta) - 2)**(-1/theta)"
    copulas = copul.from_cdf(func)
    copula = copulas(theta=8)
    result = copula.cdf(u1=0.5, u2=0.5, u3=0.5)
    copulas.cond_distr(1)
    copulas.cond_distr(2)
    copulas.pdf()
    copula.scatter_plot(approximate=True)

    # func_4d = "(x**(-theta) + y**(-theta) + z**(-theta) + w**(-theta) - 3)**(-1/theta)"
    # copulas_4d = copul.from_cdf(func_4d)
    # copula_4d = copulas_4d(theta=3)
    # copula_4d.scatter_plot(approximate=True)
    print(result)


if __name__ == "__main__":
    main()
