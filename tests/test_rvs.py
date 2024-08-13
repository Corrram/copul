import copul


def test_rvs_for_clayton():
    clayton = copul.archimedean.Clayton(theta=2)
    samples = clayton.rvs(1000)
    assert samples.shape == (1000, 2)