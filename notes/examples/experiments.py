import numpy as np
from scipy.stats import norm, multivariate_normal

# Parameters
u = 0.5
v = 0.5
rho = 0.5

# Transform the uniform marginals to standard normal
z_u = norm.ppf(u)
z_v = norm.ppf(v)

# Mean vector and covariance matrix for the bivariate normal distribution
mean = [0, 0]
cov = [[1, rho], [rho, 1]]

# Compute the value of the bivariate normal CDF at (z_u, z_v)
copula_value = multivariate_normal.cdf([z_u, z_v], mean=mean, cov=cov)

print(f"Gaussian copula value at (0.5, 0.5) with rho={rho}: {copula_value}")
