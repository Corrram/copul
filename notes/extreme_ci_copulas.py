import copul as cp
import sympy as sp
from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix


def diag_normalized_blocks(block_sizes, sk_iter=10):
    """
    Given [n1, n2, …, nk], return a block-diagonal matrix whose blocks
    all have uniform row- and column-sums = 1, except that if a block
    is size 3 we first zero its (0,0) and (2,2) entries and then
    re-normalize it via Sinkhorn–Knopp.
    """
    blocks = []
    for ni in block_sizes:
        if ni == 3:
            # start from the all-ones block,
            # zero the upper-left and lower-right
            B = sp.ones(3, 3)
            B[0, 2] = 0
            B[2, 0] = 0
            # convert to a mutable Matrix
            B = sp.Matrix(B)

            # Sinkhorn–Knopp: alternate row- and column-normalization
            for _ in range(sk_iter):
                # normalize each row to sum to 1
                for i in range(3):
                    row_sum = sum(B[i, j] for j in range(3))
                    for j in range(3):
                        B[i, j] = B[i, j] / row_sum
                # normalize each column to sum to 1
                for j in range(3):
                    col_sum = sum(B[i, j] for i in range(3))
                    for i in range(3):
                        B[i, j] = B[i, j] / col_sum

            blocks.append(B)

        else:
            # the usual J_n/n
            blocks.append(sp.ones(ni, ni) / ni)

    return BlockDiagMatrix(*blocks).as_explicit()


if __name__ == "__main__":
    # build the modified 7×7 block-diagonal ([2,3,1,1])
    M = diag_normalized_blocks([2, 3, 1, 1])

    # verify that every row and column sums to 1:
    print("row sums:", [sum(M[i, j] for j in range(7)) for i in range(7)])
    print("col sums:", [sum(M[i, j] for i in range(7)) for j in range(7)])

    # feed it into cp.BivCheckPi as before
    ccop = cp.BivCheckPi(M.tolist())
    print("rho_S =", ccop.rho())
    print("tau   =", ccop.tau())
    print("xi    =", ccop.xi())
    print("eta   =", ccop.eta())
