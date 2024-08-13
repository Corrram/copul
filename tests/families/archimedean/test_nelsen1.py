import pytest
from copul.families.archimedean import Nelsen1


@pytest.mark.parametrize("theta, expected", [(2, True), (0, True), (-0.5, False)])
def test_is_absolutely_continuous(theta, expected):
    copula = Nelsen1(theta)
    assert copula.is_absolutely_continuous == expected, (
        f"Failed for theta={theta}: Expected {expected}, "
        f"but got {copula.is_absolutely_continuous}"
    )
