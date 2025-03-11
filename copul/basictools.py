import numpy as np


def monte_carlo_integral(func, n_samples=10_000, x=1, y=1, vectorized=False):
    samples_x = np.random.rand(n_samples) * x
    samples_y = np.random.rand(n_samples) * y

    if vectorized:
        values = func(samples_x, samples_y)
    else:
        values = np.array([func(xi, yi) for xi, yi in zip(samples_x, samples_y)])

    return np.mean(values) * x * y
