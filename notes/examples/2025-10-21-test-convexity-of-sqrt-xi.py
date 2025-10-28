import copul as cp
import numpy as np
import math  # Added for sqrt


def simulate_convexity_check():
    """
    Randomly searches for a counterexample to the convexity of C -> sqrt(xi(C)).

    A counterexample is found if:
    sqrt(xi(C)) > lambda * sqrt(xi(C1)) + (1-lambda) * sqrt(xi(C2))

    where C = lambda*C1 + (1-lambda)*C2
    """
    max_violation = 0  # We are looking for the largest positive violation
    i = 0
    lam = 0.5  # Fix lambda at 0.5 for simplicity

    # Ensure numpy prints arrays in one line, no truncation
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)

    print("--- Starting Random Simulation for Convexity Violation ---")
    print(
        "Looking for (LHS - RHS) > 0, where LHS=sqrt(xi(C)), RHS=lambda*sqrt(xi(C1)) + ...\n"
    )

    while True:
        i += 1

        # 1. Generate two random copulas
        # We use BivCheckMin to get objects with a .matr attribute
        c1 = cp.BivCheckPi.generate_randomly()
        c2 = cp.BivCheckPi.generate_randomly(grid_size=c1.matr.shape[0])

        # 2. Create the convex combination C = lambda*C1 + (1-lambda)*C2
        # We do this by combining their matrices
        matr_c = lam * c1.matr + (1 - lam) * c2.matr

        # Create the new copula object from the combined matrix
        # We use BivCheckMin to treat the matrix as a literal checkerboard copula
        c = cp.BivCheckPi(matr_c)

        # 3. Get xi for all three copulas
        xi_c1 = c1.chatterjees_xi()
        xi_c2 = c2.chatterjees_xi()
        xi_c = c.chatterjees_xi()

        # 4. Check for violation
        # Add small epsilon to avoid math domain error for xi=0
        epsilon = 1e-15
        lhs = math.sqrt(xi_c + epsilon)
        rhs = lam * math.sqrt(xi_c1 + epsilon) + (1 - lam) * math.sqrt(xi_c2 + epsilon)

        violation = lhs - rhs

        # 5. Report if this is the biggest violation found so far
        if violation > max_violation:
            max_violation = violation
            print(f"--- Iteration {i}: New Max Violation Found ---")
            print(f"  Violation (LHS - RHS): {violation}")
            print(f"  LHS (sqrt(xi(C))): {lhs}  (from xi(C) = {xi_c})")
            print(f"  RHS (lambda*sqrt(xi(C1)) + ...): {rhs}")
            print(f"    - C1: n={c1.n}, xi(C1) = {xi_c1}")
            print(f"    - C2: n={c2.n}, xi(C2) = {xi_c2}")
            print(f"    - Lambda: {lam}\n")

        if i % 50000 == 0:
            print(f"Iteration {i}, max violation so far: {max_violation}")


if __name__ == "__main__":
    simulate_convexity_check()
    print("Done!")
