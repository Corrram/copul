import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata


def chatterjee_xi(x, y):
    """Calculates Chatterjee's rank correlation coefficient xi."""
    n = len(x)
    ind = np.argsort(x)
    y_sorted = y[ind]
    r = rankdata(y_sorted, method="average")
    diffs = np.abs(np.diff(r))
    xi = 1 - (3 * np.sum(diffs)) / (n**2 - 1)
    return xi


def generate_plots():
    np.random.seed(42)
    n_samples = 2000
    r_values = [0.1, 0.4, 0.9]

    # X ~ U(-pi, pi)
    X = np.random.uniform(-np.pi, np.pi, n_samples)
    # epsilon ~ N(0, 1)
    epsilon = np.random.normal(0, 1, n_samples)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # REPLACED Model 3 with a Sinusoidal model
    models = [
        ("Linear", lambda r, x, e: r * x + (1 - r) * e),
        ("Quadratic", lambda r, x, e: r * x**2 + (1 - r) * e),
        ("Sinusoidal", lambda r, x, e: r * np.sin(2 * x) + (1 - r) * e),
    ]

    latex_formulas = [
        r"$Y_1 = rX + (1-r)\varepsilon$",
        r"$Y_2 = rX^2 + (1-r)\varepsilon$",
        r"$Y_3 = r \sin(2X) + (1-r)\varepsilon$",
    ]

    for row_idx, (model_name, model_func) in enumerate(models):
        for col_idx, r in enumerate(r_values):
            ax = axes[row_idx, col_idx]

            Y = model_func(r, X, epsilon)

            rho, _ = spearmanr(X, Y)
            xi = chatterjee_xi(X, Y)

            # Plot
            ax.scatter(X, Y, alpha=0.5, s=5, c="royalblue")

            title_str = f"r = {r}\n" + r"$\rho = {:.3f}, \quad \xi = {:.3f}$".format(
                rho, xi
            )
            ax.set_title(title_str, fontsize=12)

            if row_idx == 2:
                ax.set_xlabel("X")
            if col_idx == 0:
                ax.set_ylabel(f"{model_name}\nYG")

    for row_idx, formula in enumerate(latex_formulas):
        fig.text(
            0.02,
            0.78 - row_idx * 0.27,
            formula,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.suptitle(
        "Comparison of Spearman's "
        + r"$\rho$"
        + " and Chatterjee's "
        + r"$\xi$"
        + "\nfor varying noise levels (r)",
        fontsize=16,
    )
    plt.subplots_adjust(left=0.1)

    plt.savefig("correlation_comparison_refined.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    generate_plots()
