import copul


def main():
    matr1 = [
        [3, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 2, 1, 0],
        [0, 0, 0, 3],
    ]
    matr2 = [
        [2, 2, 0, 0],
        [0, 0, 2, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    matr3 = [
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ]
    matr4 = [
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ]
    matrices = {
        "LTD+RTI does not imply xi<=tau": matr1,
        "LTD does not imply xi<=rho": matr2,
        "SI does not imply xi^T<=tau": matr3,
        "LTD+RTI does not imply xi^T<=rho": matr4,
    }
    for name, matr in matrices.items():
        check = copul.BivCheckPi(matr)

        print(
            f"name: {name}, xi: {check.chatterjees_xi()}, xi^T: {check.transpose().chatterjees_xi()}, tau: {check.kendalls_tau()}, rho: {check.spearmans_rho()}, SI: {check.is_si()}, LTD: {check.is_ltd()}, RTI: {check.is_rti()}, LTI: {check.is_lti()}, RTD: {check.is_rtd()}")


if __name__ == "__main__":
    main()