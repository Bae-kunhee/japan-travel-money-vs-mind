"""Step 5. Does the response to yen weakness change with its size? No-issue periods only
(results/yen_tiers.json).

rel = (mean of ln FX over lags 0..10) - (own 2018 average ln FX)   'how cheap the currency has been lately'
"""
import json
import warnings

import numpy as np
import pandas as pd

from config import RESULTS, K_MAIN as K
from panel_models import load_panel, open_sample, lag_cols, m2_fit

warnings.filterwarnings("ignore")
OUT = RESULTS / "yen_tiers.json"
P = load_panel()
S = open_sample(P)
R = {}

N = S[S.boycott == 0].copy()                            # no-issue sample (closure already dropped)
cols = lag_cols(K)
N["fxw"] = N[cols].mean(axis=1)
base = N[N.year == 2018].groupby("country").ln_fx.mean()
N["rel"] = N.fxw - N.country.map(base)
N["rel_pct"] = 100 * (np.exp(N.rel) - 1)


def fit_with(extra, coefs):
    Z = N.copy()
    for c in cols:                                       # FX enters only through `extra`
        Z[c] = 0.0
    out, fit, X, V = m2_fit(Z, K, extra=extra)
    cl = list(X.columns)
    res = {c: [float(fit.params[c]), float(np.sqrt(V[cl.index(c), cl.index(c)]))] for c in coefs if c in cl}
    return res, fit, X, V


# (a) linear benchmark on the no-issue sample
lin, *_ = fit_with(N[["rel"]].rename(columns={"rel": "rel_lin"}), ["rel_lin"])
R["linear"] = lin

# (b) tier dummies (reference: 0~+5% dearer than own 2018 level)
bins = [-np.inf, -0.05, 0.0, 0.05, 0.10, np.inf]
labels = ["yen_lt_m5", "m5_0", "ref_0_5", "p5_10", "p10_plus"]
N["tier"] = pd.cut(N.rel_pct / 100, bins, labels=labels, right=False)
Dm = pd.get_dummies(N.tier, dtype=float).drop(columns=["ref_0_5"])
R["tiers"], *_ = fit_with(Dm, list(Dm.columns))
R["tier_counts"] = pd.crosstab(N.country, N.tier).to_dict()

# (c) piecewise linear: depreciation slope, appreciation slope, extra slope beyond -5%
pw = pd.DataFrame({"dep": np.minimum(N.rel, 0.0), "app": np.maximum(N.rel, 0.0),
                   "dep_beyond5": np.minimum(N.rel - np.log(0.95), 0.0)}, index=N.index)
pwr, fp, Xp, Vp = fit_with(pw, ["dep", "app", "dep_beyond5"])
cl = list(Xp.columns)
i = cl.index("dep_beyond5")
pwr["z_extra_slope"] = float(fp.params["dep_beyond5"] / np.sqrt(Vp[i, i]))      # H0: no kink at -5%
a = np.zeros(len(cl)); a[cl.index("dep")] = 1; a[cl.index("app")] = -1
pwr["z_asym"] = float((a @ fp.params.values) / np.sqrt(a @ Vp @ a))             # H0: symmetric
R["piecewise"] = pwr

# (d) Japan months by tier (descriptive)
J = N[N.country == "JP"]
desc = J.groupby("tier").agg(months=("t", "size"), rel_pct=("rel_pct", "mean"), idx_2018=("idx_2018", "mean"))
R["japan_tiers_desc"] = desc.reset_index().astype({"tier": str}).to_dict(orient="records")

# (e) predicted change for sustained yen levels (linear no-issue elasticity), relative to 2018
J18 = P[(P.country == "JP") & (P.year == 2018)]
fx18, v18 = J18.fx.mean(), J18.visitors.mean() / 1000
b, se = lin["rel_lin"]
R["prediction"] = {f"{p}%": {"pct": float(100 * (np.exp(b * np.log(1 - p / 100)) - 1)),
                             "lo": float(100 * (np.exp((b + 1.96 * se) * np.log(1 - p / 100)) - 1)),
                             "hi": float(100 * (np.exp((b - 1.96 * se) * np.log(1 - p / 100)) - 1)),
                             "krw_per_100jpy": float(fx18 * (1 - p / 100)),
                             "monthly_visitors_k": float(v18 * np.exp(b * np.log(1 - p / 100)))}
                   for p in (0, 5, 10, 15, 20)}

json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
print("linear", R["linear"])
print("tiers", {k: [round(x, 3) for x in v] for k, v in R["tiers"].items()})
print("piecewise", {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in pwr.items()})
print("wrote", OUT)
