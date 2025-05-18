import copul as cp
import numpy as np


# generate 10 permutations of (1,...,10) randomly
def generate_permutations(n, num_permutations=None):
    if num_permutations is None:
        num_permutations = n
    permutations = []
    for _ in range(num_permutations):
        perm = np.random.permutation(n)
        permutations.append(perm)
    return permutations


# generate n exponential random variables
def generate_exponential_random_variables(n, scale=1.0):
    return np.random.exponential(scale, n)


# cauchy distribution
def generate_cauchy_random_variables(n, scale=1.0):
    values = np.random.standard_cauchy(size=n) * scale
    # make absolute
    return np.abs(values)


def simulate():
    # simulate n from 1 to 50
    n = np.random.randint(1, 51)
    permutations = generate_permutations(n)
    a = generate_cauchy_random_variables(n)
    identity_matr = np.eye(n)
    # shuffle the identity matrix according to the permutations
    shuffled_matrices = []
    for perm in permutations:
        shuffled_matrix = identity_matr[perm, :]
        shuffled_matrices.append(shuffled_matrix)
    # sum(a* shuffled_matrices)
    for i in range(len(shuffled_matrices)):
        shuffled_matrices[i] = shuffled_matrices[i] * a[i]
    # sum the shuffled matrices
    shuffled_matrix_sum = np.sum(shuffled_matrices, axis=0)
    ccop = cp.BivCheckPi(shuffled_matrix_sum)
    ccop_r_matr = cp.CISRearranger().rearrange_checkerboard(ccop)
    ccop_r = cp.BivCheckPi(ccop_r_matr)
    ccop_r.scatter_plot()
    ccop_r.plot_cond_distr_1()
    cis, cds = ccop_r.is_cis()
    xi = ccop_r.xi()
    tau = ccop_r.tau()
    if xi > tau:
        matr = ccop_r.matr
        print(f"xi > tau: {xi} > {tau}; is cis {cis}; for the matrix:\n{matr}")


def simulate2():
    # simulate n from 1 to 50
    n = np.random.randint(1, 1_000)
    unif_u = np.random.uniform()
    permutation = generate_permutations(n, 1)[0]
    grid_size = int(np.ceil(n**(unif_u)))
    ccop = cp.ShuffleOfMin(permutation).to_check_pi(grid_size)
    ccop_r_matr = cp.CISRearranger().rearrange_checkerboard(ccop)
    ccop_r = cp.BivCheckPi(ccop_r_matr)
    is_symmetric = ccop_r.is_symmetric
    # ccop_r.scatter_plot()
    cis, cds = ccop_r.is_cis()
    assert cis
    xi = ccop_r.xi()
    tau = ccop_r.tau()
    if xi > tau:
        matr = ccop_r.matr
        print(f"xi > tau: {xi} > {tau}; is cis {cis}; for the matrix:\n{matr}")


def simulate_one(n, rearranger):
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
    ccop = cp.BivCheckPi(M)
    ccop_r_matr = rearranger.rearrange_checkerboard(ccop)
    ccop_r = cp.BivCheckPi(ccop_r_matr)
    return ccop_r


def main(num_iters=1_000_000):
    rearranger = cp.CISRearranger()  # instantiate once
    for i in range(1, num_iters + 1):
        n = np.random.randint(1, 51)
        ccop_r = simulate_one(n, rearranger)
        # ccop_r.scatter_plot()
        xi = ccop_r.xi()
        tau = ccop_r.tau()
        cis, cds = ccop_r.is_cis()
        if xi > tau:
            print(f"xi > tau: {xi:.4f} > {tau:.4f}; cis: {cis}; matr:\n{ccop_r.matr}")

        if i % 10_000 == 0:
            print(f"Iteration {i} completed.")


if __name__ == "__main__":
    main()
    # i = 0
    # while True:
    #     i += 1
    #     simulate2()
    #     if i % 1_000 == 0:
    #         print(f"Iteration {i} completed.")
