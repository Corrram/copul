import copul as cp
import numpy as np


def simulate_one(n):
    # 1) draw n permutations all at once via argsort of uniforms
    perms = np.argsort(np.random.rand(n, n), axis=1)  # shape (n,n)
    # 2) draw n exponential random variables
    # a = np.random.exponential(size=n)  # shape (n,)
    a = np.abs(np.random.standard_cauchy(size=n))  # shape (n,)
    # a**1.5
    # a = a**1.5

    # 3) build weighted sum of permuted identity matrices:
    #    M[j,k] = sum_i a[i] * 1{perms[i,j] == k}
    #    -> we can do this in one np.add.at call
    rows = np.repeat(np.arange(n)[None, :], n, axis=0)  # shape (n,n)
    cols = perms  # shape (n,n)
    weights = np.broadcast_to(a[:, None], (n, n))  # shape (n,n)
    M = np.zeros((n, n), float)
    np.add.at(M, (rows.ravel(), cols.ravel()), weights.ravel())

    # 4) feed into copul
    return M


def simulate(num_iters=1_000_000):
    rearranger = cp.CISRearranger()
    n_max = 3
    for i in range(1, num_iters + 1):
        n = np.random.randint(2, n_max+1)
        matr = simulate_one(n)
        matr2 = simulate_one(n)
        ccop = cp.BivCheckPi(matr)
        ccop_r_matr = rearranger.rearrange_checkerboard(ccop)
        ccop_r = cp.BivCheckPi(ccop_r_matr)
        ccop2 = cp.BivCheckPi(matr2)
        ccop2_r_matr = rearranger.rearrange_checkerboard(ccop2)
        ccop2_r = cp.BivCheckPi(ccop2_r_matr)
        tau1 = ccop_r.tau()
        tau2 = ccop2_r.tau()
        matr_avg = (ccop_r_matr + ccop2_r_matr) / 2
        ccop_avg = cp.BivCheckPi(matr_avg)
        tau_avg = ccop_avg.tau()
        diff = tau_avg - (tau1 + tau2) / 2
        if diff > 1e-3:
            print(
                f"Iteration {i}: tau_avg={tau_avg:.4f} > (tau1={tau1:.4f} + tau2={tau2:.4f}) / 2, diff={diff:.4f}."
            )
            print(f"Matrix 1:\n{ccop_r_matr}")
            print(f"Matrix 2:\n{ccop2_r_matr}")
            print(f"Average Matrix:\n{matr_avg}")   
            ccop_r.plot_pdf()
            ccop2_r.plot_pdf()   
            exit()

        if i % 1_000 == 0:
            print(f"Iteration {i} completed.")


def main():
    matr1 = [[1/5, 2/15,  0  ], [2/15, 1/15, 2/15], [ 0  , 2/15, 1/5]]
    matr2 = [[9/60, 7/60, 4/60],
 [7/60, 6/60, 7/60],
 [4/60, 7/60, 9/60]]
    ccop1 = cp.BivCheckPi(matr1)
    ccop2 = cp.BivCheckPi(matr2)
    tau1 = ccop1.tau()
    tau2 = ccop2.tau()
    ccop_avg = cp.BivCheckPi((np.array(matr1) + np.array(matr2)) / 2)
    tau_avg = ccop_avg.tau()
    diff = tau_avg - (tau1 + tau2) / 2
    print(f"tau1: {tau1}, tau2: {tau2:}, tau_avg: {tau_avg:}, diff: {diff:}")

if __name__ == "__main__":
    main()
    # i = 0
    # while True:
    #     i += 1
    #     simulate2()
    #     if i % 1_000 == 0:
    #         print(f"Iteration {i} completed.")
