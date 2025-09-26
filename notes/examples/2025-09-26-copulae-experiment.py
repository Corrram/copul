import numpy as np
import pandas as pd
from copulae.empirical import EmpiricalCopula
from copulae.core import pseudo_obs

np.random.seed(1)
df = pd.DataFrame(np.random.randn(300, 3), columns=list("ABC"))
U  = pseudo_obs(df.values)

ec_cb = EmpiricalCopula(U, smoothing='checkerboard', offset=1e-6)
sim   = ec_cb.random(1000)
cdf = ec_cb.cdf([0.5,0.5,0.5])
print(sim)