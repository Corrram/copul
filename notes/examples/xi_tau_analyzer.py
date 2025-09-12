import copul as cp
import numpy as np

C = np.ones((12, 12)) / 144


def cycle_update(A, i, j, ll, k, delta):
    """+delta at (i,j) and (l,k), -delta at (i,k) and (l,j)."""
    A[i, j] += delta
    A[ll, k] += delta
    A[i, k] -= delta
    A[ll, j] -= delta


cycle_update(C, 1, 5, 2, 6, 0.005)  # big peak  near (1/6,1/2)
cycle_update(C, 0, 11, 1, 0, 0.004)  # moderate peak near (0,1)
cycle_update(C, 3, 0, 4, 1, 0.004)  # moderate peak near (1/3,0)

C = np.array(
    [
        [1, 1, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 0],
        [1, 0, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 1],
        [0, 1, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 1],
    ]
)
ccop = cp.BivCheckPi(C)
ccop_r = cp.CISRearranger().rearrange_checkerboard(ccop)
cis = cp.BivCheckPi(ccop_r).is_cis()
ccop_r = cp.BivCheckPi(ccop_r)
tau = ccop.kendalls_tau()
xi = ccop.chatterjees_xi()
ccop.plot_pdf()
ccop.plot_cond_distr_1()
ccop.plot_cdf()
print("Done!")
