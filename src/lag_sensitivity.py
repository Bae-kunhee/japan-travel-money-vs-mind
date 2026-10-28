"""Appendix. How sensitive is theta to the lag length K? (results/lag_sensitivity.csv)"""
import warnings

import numpy as np
import pandas as pd

from config import RESULTS
from panel_models import load_panel, open_sample, japan_sample, lag_cols, m1_fit, m2_fit, sum_vector

warnings.filterwarnings("ignore")
P = load_panel()
S, J = open_sample(P), japan_sample(P)
rows = []
for K in range(13):
    o, *_ = m2_fit(S, K)
    m1, X1 = m1_fit(K, J)
    a = sum_vector(X1.columns, lag_cols(K))
    rows.append({"spec": "distributed_lag", "K": K, "m2_theta": o["theta"], "m2_se": o["se"], "m2_aic": o["aic"],
                 "m1_theta": a @ m1.params.values, "m1_se": np.sqrt(a @ m1.cov_params().values @ a), "m1_aic": m1.aic})
# moving-average FX regressor (one coefficient) as a collinearity-free alternative
for K in [0, 3, 6, 9, 12]:
    cols = lag_cols(K)
    S2 = S.copy(); S2["ma"] = S2[cols].mean(axis=1)
    S2[cols] = 0.0; S2["ln_fx"] = S2["ma"]
    o, *_ = m2_fit(S2, 0)
    J2 = J.copy(); J2["ln_fx"] = J2[cols].mean(axis=1)
    m1, _ = m1_fit(0, J2)
    rows.append({"spec": "moving_average", "K": K, "m2_theta": o["theta"], "m2_se": o["se"], "m2_aic": o["aic"],
                 "m1_theta": m1.params["ln_fx"], "m1_se": m1.bse["ln_fx"], "m1_aic": m1.aic})
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "lag_sensitivity.csv", index=False)
print(out.round(3).to_string(index=False))
