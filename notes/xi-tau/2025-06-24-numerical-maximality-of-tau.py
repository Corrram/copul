import copul as cp
import numpy as np


def main(num_iters=1_000_000):
    # rearranger = cp.CISRearranger()
    for i in range(1, num_iters + 1):
        ccop = cp.BivCheckMin.generate_randomly()
        xi = ccop.chatterjees_xi()
        xi = np.clip(xi, 0, 1)
        cop = cp.XiRhoBoundaryCopula.from_xi(xi)
        tau_max = cop.kendalls_tau()
        cop.to_checkerboard().kendalls_tau()
        tau = ccop.kendalls_tau()
        if tau == 1:
            continue
        if tau > tau_max + 1e-12:
            print(
                f"Iteration {i}: tau={tau} exceeds maximal tau={tau_max} for n={ccop.m} (xi={xi}, b={cop.b})."
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
