import numpy as np


def monte_carlo_integral(func, n_samples=10_000, x=1, y=1, vectorized_func=False):
    samples_x = np.random.rand(n_samples) * x
    samples_y = np.random.rand(n_samples) * y
    if vectorized_func:
        result = np.mean(func(samples_x, samples_y))
    else:
        zipped_list = zip(samples_x, samples_y)
        func_values = [func(x, y) for x, y in zipped_list]
        result = np.mean(func_values)
    return result
