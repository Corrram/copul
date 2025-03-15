import pytest

import copul
from tests.family_representatives import family_representatives


@pytest.mark.parametrize("t", [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
def test_pickands(t):
    ev_copulas = copul.Families.list_by_category("Extreme_Value")
    for copula in ev_copulas:
        # if copula == "T_EV":
        #     continue
        params = family_representatives[copula]
        if not isinstance(params, tuple):
            params = (params,)
        copula_family = getattr(copul.Families, copula)
        cop = copula_family.cls(*params)
        result = cop.pickands(t=t)
        msg = f"Pickands function for {copula} is not in [min(t, 1-t), 1]"
        try:
            assertion = min(t, 1 - t) <= result <= 1
        except TypeError as e:
            raise AssertionError(f"Error in {copula} with t={t}") from e
        assert assertion, msg
