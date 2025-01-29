import copul

# def test_3d_clayton():
#     theta = 3
#     cdf = f"(x**(-{theta}) + y**(-{theta}) + z**(-{theta}) - 2)**(-1/{theta})"
#     copula = copul.from_cdf(cdf)
#     sample_data = copula.rvs(1)
#     _ccop = copul.from_data(sample_data)
#     # ccop.scatter_plot()
#     assert sample_data.shape == (10, 3)
