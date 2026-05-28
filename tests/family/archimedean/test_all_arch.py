import numpy as np
import pytest
import copul


ROUND_TRIP_T_VALUES = np.array([0.1, 0.25, 0.5, 0.75, 0.9])


def _representative_theta(copula_cls):
    interval = getattr(copula_cls, "theta_interval", None)
    if interval is None:
        return None

    special_cases = set(getattr(copula_cls, "special_cases", {}))
    invalid_params = set(getattr(copula_cls, "invalid_params", set()))

    for theta in (0.5, 1.5, 2.5, 3.0, -0.5, -2.0, 1.0, 2.0):
        if theta in special_cases or theta in invalid_params:
            continue
        if theta in interval:
            return theta

    raise AssertionError(f"No representative theta found for {copula_cls.__name__}")


def _representative_copula(family_name):
    copula_cls = getattr(copul.Families, family_name).cls
    theta = _representative_theta(copula_cls)
    return copula_cls() if theta is None else copula_cls(theta)


@pytest.mark.parametrize("family_name", copul.Families.list_by_category("Archimedean"))
def test_generator_and_inverse_generator_round_trip_numerically(family_name):
    if family_name == "NELSEN18":
        pytest.xfail("Nelsen18 inverse generator does not round-trip interior values")

    cop = _representative_copula(family_name)

    for t in ROUND_TRIP_T_VALUES:
        generated = float(cop.generator(t=float(t)))
        restored_t = float(cop.inv_generator(y=generated))
        regenerated = float(cop.generator(t=restored_t))

        assert np.isfinite(generated), f"{family_name} generator is not finite at t={t}"
        assert np.isclose(restored_t, t, rtol=1e-10, atol=1e-10), (
            f"{family_name} inverse generator failed at t={t}: "
            f"got phi^-1(phi(t))={restored_t}"
        )
        assert np.isclose(regenerated, generated, rtol=1e-9, atol=1e-10), (
            f"{family_name} generator failed after inverse at y={generated}: "
            f"got phi(phi^-1(y))={regenerated}"
        )


@pytest.mark.parametrize("family_name", copul.Families.list_by_category("Archimedean"))
def test_inverse_generator_clamps_above_finite_generator_range(family_name):
    if family_name == "NELSEN18":
        pytest.xfail("Nelsen18 inverse generator does not clamp above generator(0)")

    cop = _representative_copula(family_name)

    try:
        generator_at_0 = float(cop._generator_at_0)
    except TypeError:
        pytest.fail(f"{family_name} has a non-numeric generator_at_0")

    if not np.isfinite(generator_at_0):
        pytest.skip(f"{family_name} has unbounded generator range")

    y_above_range = generator_at_0 + max(1e-6, abs(generator_at_0) * 1e-6)

    # Use positional evaluation to exercise the family's raw inverse expression.
    actual = float(cop.inv_generator(y_above_range))

    assert np.isclose(actual, 0, atol=1e-10), (
        f"{family_name} inverse generator should clamp to 0 above "
        f"generator(0)={generator_at_0}, got {actual} at y={y_above_range}"
    )


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
