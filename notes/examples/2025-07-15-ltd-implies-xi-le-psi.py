import copul as cp
import numpy as np


def main(num_iters=1_000_000):
    # rearranger = cp.CISRearranger()
    n_max = 10
    ltd_counter = 0
    for i in range(1, num_iters + 1):
        ccop = cp.BivCheckPi.generate_randomly([2, n_max])
        ccop_min = cp.BivCheckMin(ccop)
        is_ltd = cp.LTDVerifier().is_ltd(ccop)
        is_ltd_min = cp.LTDVerifier().is_ltd(ccop_min)
        if (not is_ltd_min) and (not is_ltd):
            if i % 10_000 == 0:
                print(f"Iteration {i} completed.")
            continue
        ltd_counter += 1
        footrule = ccop.footrule()
        xi = ccop.xi()
        if xi > footrule + 1e-8:
            is_plod = cp.PLODVerifier().is_plod(ccop)
            print(
                f"Iteration {i}: footrule={footrule:.4f}, xi={xi:.4f} for n={n}."
            )
            print(f"Matrix:\n{ccop.matr}")
            cis, cds = ccop.is_cis()
            print(f"CIS: {cis}, CDS: {cds}, LTD: {is_ltd}, PLOD: {is_plod}")
        footrule_min = ccop_min.footrule()
        xi_min = ccop_min.xi()
        if xi_min > footrule_min + 1e-8:
            is_plod_min = cp.PLODVerifier().is_plod(ccop_min)
            cis_min, cds_min = ccop_min.is_cis()
            print(
                f"Iteration {i}: footrule_min={footrule_min:.4f}, xi_min={xi_min:.4f} for n={n}."
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
