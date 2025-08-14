import copul as cp


def main(num_iters=1_000_000):
    # rearranger = cp.CISRearranger()
    ltd_counter = 0
    for i in range(1, num_iters + 1):
        ccop = cp.BivCheckPi.generate_randomly()
        ccop_min = cp.BivCheckMin(ccop)
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
                f"Iteration {i}: tau={tau:.4f}, footrule={footrule:.4f}, ginis_gamma={ginis_gamma:.4f} for n={ccop.n}."
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
                f"Iteration {i}: tau_min={tau_min:.4f}, footrule_min={footrule_min:.4f}, ginis_gamma_min={ginis_gamma_min:.4f} for n={ccop.n}."
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
