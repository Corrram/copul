import itertools

import numpy as np
import sympy

from copul.families import archimedean
from copul.schur_order.checkerboarder import Checkerboarder
from copul.schur_order.cis_rearranger import CISRearranger


class SchurOrderVerifier:
    def __init__(self, copula, n_theta=40, chess_board_size=10):
        self.copula = copula
        self._n_theta = n_theta
        self._chess_board_size = chess_board_size

    def verify(self, range_min=None, range_max=None):
        range_min = -10 if range_min is None else range_min
        interval = self.copula.intervals[str(self.copula.params[0])]
        range_min = float(max(interval.inf, range_min)) + 0.01
        range_max = 10 if range_max is None else range_max
        range_max = float(min(interval.end, range_max)) - 0.01
        checkerboarder = Checkerboarder(self._chess_board_size)
        ccop = checkerboarder.compute_check_pi(self.copula)
        cond_distributions = []
        thetas = np.linspace(range_min, range_max, self._n_theta)
        for theta in thetas:
            my_ccop = ccop.subs(self.copula.theta, theta)
            rearranged_ccop = CISRearranger().rearrange_checkerboard(my_ccop)
            cond_dens = sympy.Matrix.zeros(
                rearranged_ccop.shape[0], rearranged_ccop.shape[1]
            )
            for k, l_ in np.ndindex(rearranged_ccop.shape):
                cond_dens[k, l_] = sum(rearranged_ccop[i, l_] for i in range(k + 1))
            cond_distr = sympy.Matrix.zeros(cond_dens.shape[0], cond_dens.shape[1])
            for k, l_ in np.ndindex(cond_dens.shape):
                cond_distr[k, l_] = sum(cond_dens[k, j] for j in range(l_ + 1))
            cond_distributions.append(cond_distr)
        positively_ordered = True
        for i in range(len(cond_distributions) - 1):
            smaller_cop = cond_distributions[i]
            larger_cop = cond_distributions[i + 1]
            if positively_ordered and not self._is_pointwise_lower_equal(
                smaller_cop, larger_cop
            ):
                print(f"Not positively Schur ordered at {thetas[i]} / {thetas[i + 1]}.")
                counterexample = [
                    f"{i} {j}, diff: {larger_cop[i, j] - smaller_cop[i, j]}"
                    for i, j in itertools.product(
                        range(smaller_cop.rows), range(smaller_cop.cols)
                    )
                    if not smaller_cop[i, j] <= larger_cop[i, j] + 0.0000000001
                ]
                print(counterexample)
                positively_ordered = False
            if not positively_ordered and not self._is_pointwise_lower_equal(
                larger_cop, smaller_cop
            ):
                print(f"Not negatively Schur ordered at {thetas[i]} / {thetas[i + 1]}.")
                counterexample = [
                    f"{i} {j}, diff: {larger_cop[i, j] - smaller_cop[i, j]}"
                    for i, j in itertools.product(
                        range(larger_cop.rows), range(larger_cop.cols)
                    )
                    if not larger_cop[i, j] <= smaller_cop[i, j] + 0.0000000001
                ]
                print(counterexample)
                return False
        if positively_ordered:
            print("Positively Schur ordered.")
        else:
            print("Negatively Schur ordered.")
        return True

    @staticmethod
    def _is_pointwise_lower_equal(cdf1, cdf2):
        if cdf1.shape != cdf2.shape:
            raise ValueError("Matrices must have the same shape.")

        return all(
            cdf1[i, j] <= cdf2[i, j] + 0.0000000001
            for i, j in itertools.product(range(cdf1.rows), range(cdf1.cols))
        )


if __name__ == "__main__":
    for i in [2, 8, 15, 18, 21]:
        print(f"Nelsen{i}")
        copula = getattr(archimedean, f"Nelsen{i}")()
        SchurOrderVerifier(copula).verify(1)
