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
    ltd_counter = 0
    for i in range(1, num_iters + 1):
        n = np.random.randint(2, n_max)
        matr = simulate_one(n)
        ccop = cp.BivCheckPi(matr)
        ccop_min = cp.BivCheckMin(matr)
        # ccop_r_matr = rearranger.rearrange_checkerboard(ccop)
        # ccop_r = cp.BivCheckPi(ccop_r_matr)
        # ccop_r.scatter_plot()
        is_ltd = cp.LTDVerifier().is_ltd(ccop)
        is_ltd_min = cp.LTDVerifier().is_ltd(ccop_min)
        if (not is_ltd_min) and (not is_ltd):
            # print(f"Iteration {i}: LTD property violated for n={n}.")
            if i % 10_000 == 0:
                print(f"Iteration {i} completed.")
            continue
        ltd_counter += 1
        footrule = ccop.footrule()
        ginis_gamma = ccop.ginis_gamma()
        if ginis_gamma < footrule - 1e-8:
            is_plod = cp.PLODVerifier().is_plod(ccop)
            tau = ccop.tau()
            print(
                f"Iteration {i}: tau={tau:.4f}, footrule={footrule:.4f}, ginis_gamma={ginis_gamma:.4f} for n={n}."
            )
            print(f"Matrix:\n{ccop.matr}")
            cis, cds = ccop.is_cis()
            print(f"CIS: {cis}, CDS: {cds}, LTD: {is_ltd}, PLOD: {is_plod}")
        footrule_min = ccop_min.footrule()
        ginis_gamma_min = ccop_min.ginis_gamma()
        if ginis_gamma_min < footrule_min - 1e-8:
            is_plod_min = cp.PLODVerifier().is_plod(ccop_min)
            cis_min, cds_min = ccop_min.is_cis()
            tau_min = ccop_min.tau()
            print(
                f"Iteration {i}: tau_min={tau_min:.4f}, footrule_min={footrule_min:.4f}, ginis_gamma_min={ginis_gamma_min:.4f} for n={n}."
            )
            print(f"Matrix Min:\n{ccop_min.matr}")
            cis, cds = ccop_min.is_cis()
            print(
                f"CIS Min: {cis_min}, CDS Min: {cds_min}, LTD Min: {is_ltd_min}, PLOD Min: {is_plod_min}"
            )

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
