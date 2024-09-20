import copul


def main():
    func = "(x**(-theta) + y**(-theta) + z**(-theta) - 2)**(-1/theta)"
    copulas = copul.from_cdf(func)
    copula = copulas(theta=0.5)
    result = copula.cdf(u1=0.5, u2=0.5, u3=0.5)
    cd1 = copulas.cond_distr(1)
    cd2 = copulas.cond_distr(2)
    pdf = copulas.pdf()
    print(result)


if __name__ == "__main__":
    main()
