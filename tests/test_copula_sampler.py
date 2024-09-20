import random

import numpy as np

import copul
from copul.copula_sampler import CopulaSampler


def test_rvs_from_upper_frechet():
    copula = copul.UpperFrechet()
    sampler = CopulaSampler(copula)
    results = sampler.rvs(3)
    assert len(results) == 3
    for result in results:
        assert len(result) == 2
        assert np.isclose(result[0], result[1])


def test_rvs_from_lower_frechet():
    copula = copul.LowerFrechet()
    sampler = CopulaSampler(copula)
    results = sampler.rvs(3)
    assert len(results) == 3
    for result in results:
        assert len(result) == 2
        assert np.isclose(result[0], 1 - result[1])


def test_rvs_from_independence_copula():
    random.seed(42)
    copula = copul.IndependenceCopula()
    sampler = CopulaSampler(copula)
    results = sampler.rvs(100)
    corr = np.corrcoef(results[:, 0], results[:, 1])[0, 1]
    assert np.abs(corr) < 0.1
