import copul as cp

copula = cp.from_cdf("u*v + u*(1-u)*max(0,sin(2*pi*v))/(2*pi)")
cdf = copula.cdf(0.50, 0)
pi = cp.BivIndependenceCopula()
ccop = copula.to_check_pi()
xi = ccop.xi()
tau = ccop.tau()
is_cis, is_cds = ccop.is_cis()
print(f"xi = {xi}, tau = {tau}, is_cis = {is_cis}")