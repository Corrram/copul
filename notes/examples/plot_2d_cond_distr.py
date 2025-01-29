import matplotlib.pyplot as plt
import numpy as np

import copul as cp


def main():
    # Define the range of v for plotting
    v_values = np.linspace(0, 1, 100)

    # Conditional distributions for Plackett copula
    cop1 = cp.Plackett(0.1).cond_distr_1(u=0.2)
    cop2 = cp.Plackett(0.1).cond_distr_1(u=0.5)
    cop3 = cp.Plackett(0.1).cond_distr_1(u=0.8)

    # Evaluate the functions at each v value
    cop1_values = [cop1(v) for v in v_values]
    cop2_values = [cop2(v) for v in v_values]
    cop3_values = [cop3(v) for v in v_values]

    # Plot the three functions of v for Plackett copula
    plt.figure()
    plt.plot(v_values, cop1_values, label="u=0.2")
    plt.plot(v_values, cop2_values, label="u=0.5")
    plt.plot(v_values, cop3_values, label="u=0.8")
    plt.title("Plackett copula conditional distributions")
    plt.xlabel("v")
    plt.ylabel("Conditional Distribution")
    plt.legend()
    plt.grid()
    plt.savefig("plackett_conditional_distributions.png")
    plt.show()

    # Conditional distributions for Joe copula
    joe_cop1 = cp.Joe(3).cond_distr_1(u=0.2)
    joe_cop2 = cp.Joe(3).cond_distr_1(u=0.5)
    joe_cop3 = cp.Joe(3).cond_distr_1(u=0.8)

    # Evaluate the functions at each v value
    joe_cop1_values = [joe_cop1(v) for v in v_values]
    joe_cop2_values = [joe_cop2(v) for v in v_values]
    joe_cop3_values = [joe_cop3(v) for v in v_values]

    # Plot the three functions of v for Joe copula
    plt.figure()
    plt.plot(v_values, joe_cop1_values, label="u=0.2")
    plt.plot(v_values, joe_cop2_values, label="u=0.5")
    plt.plot(v_values, joe_cop3_values, label="u=0.8")
    plt.title("Joe copula conditional distributions")
    plt.xlabel("v")
    plt.ylabel("Conditional Distribution")
    plt.legend()
    plt.grid()
    plt.savefig("joe_conditional_distributions.png")
    plt.show()


if __name__ == "__main__":
    main()
