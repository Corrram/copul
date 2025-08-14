import copul as cp
import numpy as np


def main(num_iters=1_000_000):
    rearranger = cp.CISRearranger()
    n_max = 3
    for i in range(1, num_iters + 1):
        n = np.random.randint(2, n_max + 1)
        ccop = cp.BivCheckPi.generate_randomly([2, 3])
        ccop_r_matr = rearranger.rearrange_checkerboard(ccop)
        ccop_r = cp.BivCheckPi(ccop_r_matr)
        is_cis_1, is_cds_1 = ccop_r.is_cis()
        footrule = ccop_r.footrule()
        gamma = ccop_r.ginis_gamma()
        if footrule < gamma:
            print(
                f"Iteration {i}: Footrule ({footrule}) > Gini's gamma ({gamma}) for n={n}."
            )
            print(f"Matrix:\n{ccop_r_matr}")
            exit()

        if i % 1_000 == 0:
            print(f"Iteration {i} completed.")


if __name__ == "__main__":
    main()
