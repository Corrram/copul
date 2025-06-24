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


def main(num_iters=1_000_000):
    # rearranger = cp.CISRearranger()
    n_max = 10
    for i in range(1, num_iters + 1):
        n = np.random.randint(2, n_max)
        matr = simulate_one(n)
        ccop = cp.BivCheckMin(matr)
        xi = ccop.xi()
        xi = np.clip(xi, 0,1)
        cop = cp.RhoMinusXiMaximalCopula.from_xi(xi)
        tau_max = cop.tau()
        tau_max_alt = cop.to_checkerboard().tau()
        tau = ccop.tau()
        if tau == 1:
            continue
        if tau > tau_max + 1e-12:
            print(
                f"Iteration {i}: tau={tau} exceeds maximal tau={tau_max} for n={n} (xi={xi}, b={cop.b})."
            )
            print(f"Matrix:\n{ccop.matr}")
        if i % 1_000 == 0:
            print(f"Iteration {i} completed.")


if __name__ == "__main__":
    main()
    # i = 0
    # while True:
    #     i += 1
    #     simulate2()
    #     if i % 1_000 == 0:
    #         print(f"Iteration {i} completed.")
