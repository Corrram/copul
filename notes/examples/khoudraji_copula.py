import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mfoci
from pandas.plotting import scatter_matrix
import copul as cp

# 1) Match Kendall's tau = 0.95 for both base copulas
tau = 0.95
theta_clayton = 2 * tau / (1 - tau)  # inversion for Clayton: τ = θ/(θ+2)
theta_gumbel = 1 / (1 - tau)  # inversion for Gumbel: τ = 1 – 1/θ

# 2) Instantiate the two Archimedean copulas
clayton = cp.Clayton(theta=theta_clayton)
gumbel = cp.GumbelHougaard(theta=theta_gumbel)


# 3) Khoudraji transform: if (U1,V1)~C1 and (U2,V2)~C2 i.i.d., then
#    (U, V) = (U1^a * U2^(1-a),  V1^b * V2^(1-b))
def sample_khoudraji(n, cop1, cop2, a, b):
    u1, v1 = cop1.rvs(n).T
    u2, v2 = cop2.rvs(n).T
    u = u1**a * u2 ** (1 - a)
    v = v1**b * v2 ** (1 - b)
    return np.column_stack([u, v])


# 4) Draw 1 000 samples with shape = (0.95, 0.6)
samples = sample_khoudraji(1000, clayton, gumbel, a=0.95, b=0.5)
df = pd.DataFrame(samples, columns=["U", "V"])

# 5) Scatterplot matrix (diagonals = histograms)
scatter_matrix(df, diagonal="hist", alpha=0.6)
plt.suptitle("Khoudraji–Clayton/Gumbel Copula (τ=0.95; shape=(0.95,0.6))")
plt.show()
xi = mfoci.codec(df.iloc[:, 0], df.iloc[:, 1])
xi_transpose = mfoci.codec(df.iloc[:, 1], df.iloc[:, 0])
tau = df.corr(method="kendall").iloc[0, 1]
print(f"tau = {tau}, xi = {xi}, xi_transpose = {xi_transpose}")
