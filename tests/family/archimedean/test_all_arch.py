import numpy as np
import copul


def test_all_generators():
    arch_copulas = copul.Families.list_by_category("Archimedean")
    for copula in arch_copulas:
        cop = getattr(copul.Families, copula).cls()
        if copula in ["CLAYTON", "NELSEN1", "NELSEN7"]:
            cop = cop(
                0.5
            )  # needed because generator value for clayton at 0 depends on theta
        elif copula == "NELSEN18":
            cop = cop(2.5)
        try:
            gen_0 = cop.generator(t=0)
        except TypeError:
            raise TypeError(f"Generator at 0 for {copula} is not a float")
        try:
            gen_0_float = float(gen_0)
        except TypeError:
            raise TypeError(f"Generator at 0 for {copula} is not a float")
        try:
            expected = float(cop._generator_at_0)
        except AttributeError:
            raise AttributeError(f"Generator at 0 for {copula} is not defined")
        assert gen_0_float == expected, f"Generator at 0 for {copula} is not correct"


def test_all_inv_generators():
    arch_copulas = copul.Families.list_by_category("Archimedean")
    for copula in arch_copulas:
        cop = getattr(copul.Families, copula).cls()
        if copula in ["CLAYTON", "NELSEN1", "NELSEN7"]:
            cop = cop(
                0.5
            )  # needed because generator value for clayton at 0 depends on theta
        elif copula == "NELSEN18":
            cop = cop(2.5)
        try:
            gen_0 = cop.inv_generator(y=0)
        except TypeError:
            raise TypeError(f"Inv Generator at 0 for {copula} not callable")
        try:
            gen_1 = cop.inv_generator(y=cop._generator_at_0)
        except TypeError:
            raise TypeError(f"Inv Generator at 1 for {copula} not callable")
        try:
            gen_0_float = float(gen_0)
        except TypeError:
            raise TypeError(f"Inv Generator at 0 for {copula} is not a float")
        try:
            gen_1_float = float(gen_1)
        except TypeError:
            raise TypeError(
                f"Inv Generator at {cop._generator_at_0} for {copula} is not a float but {gen_1}"
            )
        assert np.isclose(gen_0_float, 1), f"Inv Generator at 0 for {copula} is not 1"
        assert np.isclose(gen_1_float, 0), f"Inv Generator at 1 for {copula} is not 0"
