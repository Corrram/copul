import copul as cp
import numpy as np


def simulate():
    minimal_value = 0
    i = 0

    # Ensure numpy prints arrays in one line, no truncation
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)

    while True:
        i += 1
        ccop = cp.BivCheckMin.generate_randomly()
        footrule = ccop.spearmans_footrule()
        xi = ccop.chatterjees_xi()
        if footrule + xi < minimal_value:
            minimal_value = footrule + xi
            print(
                f"Iteration {i}: Footrule ({footrule}), Xi ({xi}), Sum = {footrule + xi}, n={ccop.n}"
            )
            matrix_str = (
                "["
                + " ".join(
                    "[" + " ".join(f"{x:.8f}" for x in row) + "]" for row in ccop.matr
                )
                + "]"
            )
            print(f"Matrix: {matrix_str}\n")


def analyze_minimizer():
    # matr = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    matr = [[0, 1], [1, 0]]
    ccop_min = cp.BivCheckMin(matr)
    ccop_pi = cp.BivCheckPi(matr)
    ccop_w = cp.BivCheckW(matr)
    xi_min = ccop_min.chatterjees_xi()
    footrule_min = ccop_min.spearmans_footrule()
    xi_pi = ccop_pi.chatterjees_xi()
    xi_w = ccop_w.chatterjees_xi()
    footrule_w = ccop_w.spearmans_footrule()
    footrule_pi = ccop_pi.spearmans_footrule()

    sum_min = xi_min + footrule_min
    sum_w = xi_w + footrule_w
    sum_pi = xi_pi + footrule_pi
    print(f"Sum min = {sum_min}, Sum w = {sum_w}, Sum pi = {sum_pi}")
    print(f"Xi pi = {xi_pi}, footrule pi = {footrule_pi}")


if __name__ == "__main__":
    analyze_minimizer()
    # simulate()
    print("Done!")
