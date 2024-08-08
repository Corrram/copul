import numpy as np

from copul import CheckerboardCopula
from copul.schur_order.cis_rearranger import CISRearranger


class DoubleStochasticMatrix:
    def __init__(self, n=3, iterations=None):
        self.n = n
        if iterations is None:
            iterations = n**2
        self.iterations = iterations
        self.matrix = self.generate_double_stochastic_matrix()

    @staticmethod
    def normalize_rows(matrix):
        return matrix / matrix.sum(axis=1, keepdims=True)

    @staticmethod
    def normalize_columns(matrix):
        return matrix / matrix.sum(axis=0, keepdims=True)

    def generate_double_stochastic_matrix(self):
        matrix = np.random.rand(self.n, self.n)
        # generate random number between zero and ten
        rand = np.random.randint(0, 10)
        # add random number to each element of the main diagonal of the matrix
        np.fill_diagonal(matrix, matrix.diagonal() + rand)
        for i in range(self.iterations):
            matrix = self.normalize_rows(matrix)
            matrix = self.normalize_columns(matrix)
        return matrix

    def __str__(self):
        return str(self.matrix)


if __name__ == "__main__":
    ccop = DoubleStochasticMatrix(n=20).matrix
    ccop_rearranged = CISRearranger().rearrange_checkerboard(ccop)
    copula = CheckerboardCopula(ccop_rearranged)
    rho = copula.rho()
    tau = copula.tau()
    xi = copula.xi()
    print("Done!")
