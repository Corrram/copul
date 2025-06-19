import numpy as np


def xi_n(epsilon, n, c=1.0):
    """
    Approximate ξ_n(X,Y) = ε + O(1/n).

    Parameters
    ----------
    epsilon : float or array‐like
        Your chosen ε > 0.
    n : int
        Sample size (should be large for the asymptotic to kick in).
    c : float, optional
        Coefficient of the 1/n correction (default 1.0).

    Returns
    -------
    xi_approx : same shape as epsilon
        ε + c/n
    """
    return np.array(epsilon) + c / n


def S_n(epsilon, n, c=1.0):
    """
    Approximate |S_n(X,Y)| = 1 - sqrt(2/27) * (1-ε)^(3/2) + O(1/n).

    Parameters
    ----------
    epsilon : float or array‐like
        Your chosen ε > 0.
    n : int
        Sample size.
    c : float, optional
        Coefficient of the 1/n correction (default 1.0).

    Returns
    -------
    S_approx : same shape as epsilon
        1 - sqrt(2/27)*(1-ε)**1.5 + c/n
    """
    eps = np.array(epsilon)
    return 1 - np.sqrt(2 / 27) * (1 - eps) ** 1.5 + c / n


if __name__ == "__main__":
    # example usage
    epsilons = [0.01, 0.1, 0.2, 0.5]
    n = 1000

    xi_vals = xi_n(epsilons, n, c=0.5)  # use c=0.5 for a smaller O(1/n) term
    S_vals = S_n(epsilons, n, c=0.5)

    for ε, xi_hat, S_hat in zip(epsilons, xi_vals, S_vals):
        print(f"ε={ε:>4}  →  ξₙ≈{xi_hat:.4f},  |Sₙ|≈{S_hat:.4f}")
