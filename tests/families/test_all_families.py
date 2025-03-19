import logging

import numpy as np
import pytest

import copul
from copul.exceptions import PropertyUnavailableException
from tests.family_representatives import (
    archimedean_representatives,
    family_representatives,
)

log = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "point, expected", [((0, 0), 0), ((0, 0.5), 0), ((1, 0.5), 0.5), ((1, 1), 1)]
)
def test_cdf_edge_cases(point, expected):
    for copula, param in family_representatives.items():
        cop = getattr(copul, copula)
        if isinstance(param, tuple):
            cop = cop(*param)
        else:
            cop = cop(param)
        evaluated_cdf = cop.cdf(*point)
        actual = evaluated_cdf.evalf()
        assert np.isclose(actual, expected), f"{copula} CDF: {actual}"


@pytest.mark.parametrize(
    "method_name, point, expected",
    [
        ("cond_distr_1", (0, 0), 0),
        # ("cond_distr_1", (0, 1), 1),  # ToDo: Check why this sometimes fails from cli
        ("cond_distr_2", (0, 0), 0),
        # ("cond_distr_2", (1, 0), 1),
    ],
)
def test_cond_distr_edge_cases(method_name, point, expected):
    for copula, param in family_representatives.items():
        cop = getattr(copul, copula)
        if isinstance(param, tuple):
            cop = cop(*param)
        else:
            cop = cop(param)
        method = getattr(cop, method_name)
        try:
             func = method(*point)
        except ValueError as e:
            raise ValueError(
                f"Error evaluating {copula}.{method_name}({point}): {e}"
            ) from e
        evaluated_func = float(func)     
        assert np.isclose(evaluated_func, expected), f"{copula} {method_name}: {evaluated_func}"


def test_pdfs():
    # ToDo: Add tests for non-archimedean copula pdfs
    for copula, param in archimedean_representatives.items():
        cop = getattr(copul, copula)
        if isinstance(param, tuple):
            cop = cop(*param)
        else:
            cop = cop(param)
        try:
            pdf = cop.pdf
        except PropertyUnavailableException:
            continue
        evaluated_cdf = pdf(0.5, 0.5).evalf()
        log.info(f"{copula} pdf: {evaluated_cdf}")
        assert evaluated_cdf >= 0
