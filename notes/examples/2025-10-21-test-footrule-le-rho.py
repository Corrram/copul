import copul as cp
import numpy as np


def simulate_convexity_check():
    """
    Randomly searches for a counterexample to the convexity of C -> sqrt(xi(C)).

    A counterexample is found if:
    sqrt(xi(C)) > lambda * sqrt(xi(C1)) + (1-lambda) * sqrt(xi(C2))

    where C = lambda*C1 + (1-lambda)*C2
    """
    max_violation = 0
    i = 0

    # Ensure numpy prints arrays in one line, no truncation
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)

    print("--- Starting Random Simulation for Xi/Psi SI relation ---")
    print(
        "Looking for (LHS - RHS) > 0, where LHS=sqrt(xi(C)), RHS=lambda*sqrt(xi(C1)) + ...\n"
    )

    while True:
        i += 1

        # 1. Generate two random copulas
        # We use BivCheckMin to get objects with a .matr attribute
        c1_raw = cp.BivCheckPi.generate_randomly()
        c1 = cp.BivCheckPi(cp.CISRearranger().rearrange_checkerboard(c1_raw))
        footrule = c1.spearmans_footrule()
        rho = c1.spearmans_rho()
        if footrule > rho:
            print(f"{footrule}, {rho}")
            print(c1.matr)

        if i % 50000 == 0:
            print(f"Iteration {i}, max violation so far: {max_violation}")


if __name__ == "__main__":
    simulate_convexity_check()
    print("Done!")
