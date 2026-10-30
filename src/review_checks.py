"""Checks added after the external review (2026-10-30). Outputs results/review_checks.json.

1. Sample composition: 600 country-months -> 463 used.
2. Effect-size difference D = (-gamma) - theta*ln(0.9) (log scale), avoids a ratio with theta near 0.
3. Japan-specific FX elasticity: theta_JP and the ratio computed with it.
4. Boycott window without 2020-01..02 (drop those months for every country).
5. Event study: Japan x month dummies 2018-07..2020-02 (pre-trend check), DK covariance.
6. Japan vs comparison average, seasonally adjusted, 2017-01..2020-02 (data for the trend figure).
7. Cluster-robust (CR1, t(4)) interval for the 10% yen effect, used in the effect figure.
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import RESULTS, K_MAIN as K, COUNTRIES, ym
from panel_models import load_panel, open_sample, m2_fit, dk_cov, lag_cols, sum_vector, design

warnings.filterwarnings("ignore")
LN09 = np.log(0.9)
P = load_panel()
S = open_sample(P)
R = {}

# ------------------------------------------------------------------ 1. sample composition
comp = []
for c in COUNTRIES:
    p = P[P.country == c]
    comp.append({"country": c, "months_2016_2025": int(len(p)), "closed_months": int(p.closed.sum()),
                 "missing_visitors_open": int(((p.closed == 0) & p.ln_v.isna()).sum()),
                 "used": int(((p.closed == 0) & p.ln_v.notna()).sum()),
                 "closed_from": "2020-03", "reopen": f"{2015 + int(p.reopen.iloc[0]) // 12}-{int(p.reopen.iloc[0]) % 12 + 1:02d}"})
R["sample"] = {"rows": comp, "total": int(len(P)), "used": int(len(S)),
               "unique_months": int(S.t.nunique()),
               "note": "FX lags 0..12 are built on the calendar series from 2015-01, so every K uses the same 463 rows; "
                       "the closure gap is dropped, and the Driscoll-Kraay kernel (bandwidth 3) treats 2020-02 and the "
                       "first reopened month as adjacent periods."}

# ------------------------------------------------------------------ 2. D = -gamma - theta ln 0.9
o, fit, X, V = m2_fit(S, K)
cols = list(X.columns)
a = sum_vector(cols, [c for c in lag_cols(K) if c in cols])
b = sum_vector(cols, ["boycott"])
w = -b - LN09 * a                       # D = -gamma - theta * ln0.9
D = float(w @ fit.params.values)
seD = float(np.sqrt(w @ V @ w))
R["D"] = {"D": D, "se_dk": seD, "ci_dk": [D - 1.96 * seD, D + 1.96 * seD], "z": D / seD,
          "boycott_log": float(-fit.params["boycott"]), "yen10_log": float(o["theta"] * LN09),
          "worst_case": float(-fit.params["boycott"] - (-2.582351365939865) * LN09)}

# ------------------------------------------------------------------ 3. Japan-specific theta
lagsum = S[lag_cols(K)].sum(axis=1)
oj, fj, Xj, Vj = m2_fit(S, K, extra=((S.country == "JP") * lagsum).rename("jp_fx").to_frame())
cj = list(Xj.columns)
aj = sum_vector(cj, [c for c in lag_cols(K) if c in cj] + ["jp_fx"])
th_jp = float(aj @ fj.params.values)
ga_jp = float(fj.params["boycott"])
R["japan_specific"] = {"theta_JP": th_jp, "se_dk": float(np.sqrt(aj @ Vj @ aj)), "theta_others": oj["theta"],
                       "gamma": ga_jp, "yen10_pct": float(100 * (np.exp(th_jp * LN09) - 1)),
                       "ratio": float(-ga_jp / (th_jp * LN09))}

# ------------------------------------------------------------------ 4. drop 2020-01..02 for all countries
S4 = S[~S.t.isin([ym(2020, 1), ym(2020, 2)])]
o4, f4, *_ = m2_fit(S4, K)
R["drop_2020_JanFeb"] = {"theta": o4["theta"], "se": o4["se"], "gamma": o4["boycott"][0], "gamma_se": o4["boycott"][1],
                         "boycott_pct": float(100 * (np.exp(o4["boycott"][0]) - 1)),
                         "yen10_pct": float(100 * (np.exp(o4["theta"] * LN09) - 1)),
                         "ratio": float(-o4["boycott"][0] / (o4["theta"] * LN09)), "n": o4["n"]}

# ------------------------------------------------------------------ 5. event study
win = list(range(ym(2018, 7), ym(2020, 2) + 1))
S5 = S.copy()
S5["boycott"] = 0.0
ev = pd.DataFrame({f"jp_{t}": ((S5.country == "JP") & (S5.t == t)).astype(float) for t in win}, index=S5.index)
X5 = design(S5, K, extra=ev)
f5 = sm.OLS(S5.ln_v, X5).fit()
V5 = dk_cov(S5.ln_v, X5, S5)
c5 = list(X5.columns)
es = []
for t in win:
    i = c5.index(f"jp_{t}")
    es.append({"t": t, "month": f"{2015 + t // 12}-{t % 12 + 1:02d}", "coef": float(f5.params.iloc[i]),
               "se": float(np.sqrt(V5[i, i]))})
es = pd.DataFrame(es)
pre_idx = [c5.index(f"jp_{t}") for t in win if t < ym(2019, 7)]
bp = f5.params.values[pre_idx]
Vp = V5[np.ix_(pre_idx, pre_idx)]
W = float(bp @ np.linalg.pinv(Vp) @ bp)
boy_mean = float(es[es.t >= ym(2019, 7)].coef.mean())
rel = boy_mean - float(bp.mean())                 # boycott months relative to the 12 months just before
R["event_study"] = {"coefs": es.to_dict("records"), "pre_mean": float(bp.mean()), "pre_max_abs": float(np.abs(bp).max()),
                    "pre_wald": W, "pre_wald_df": len(pre_idx), "pre_wald_p": float(stats.chi2.sf(W, len(pre_idx))),
                    "boycott_mean": boy_mean, "boycott_rel_pre_log": rel,
                    "boycott_rel_pre_pct": float(100 * (np.exp(rel) - 1)),
                    "ratio_rel_pre": float(-rel / (o["theta"] * LN09)),
                    "reference": "Japan months outside 2018-07..2020-02"}
es.to_csv(RESULTS / "event_study_japan.csv", index=False)

# ------------------------------------------------------------------ 6. Japan vs comparators, seasonally adjusted
F = P.copy()
base = F[(F.t >= ym(2016, 1)) & (F.t <= ym(2019, 6))]
sm_ = base.groupby(["country", "month"]).ln_v.mean().rename("sm")
F = F.join(sm_, on=["country", "month"])
F["sa"] = F.ln_v - F.sm
pre = F[(F.t >= ym(2018, 7)) & (F.t <= ym(2019, 6))].groupby("country").sa.mean().rename("pre")
F = F.join(pre, on="country")
F["idx"] = 100 * np.exp(F.sa - F.pre)
T = F[(F.t >= ym(2017, 1)) & (F.t <= ym(2020, 2))]
tr = pd.DataFrame({"t": sorted(T.t.unique())})
tr["japan"] = tr.t.map(T[T.country == "JP"].set_index("t").idx)
tr["comparators"] = tr.t.map(T[T.country != "JP"].groupby("t").idx.mean())
tr.to_csv(RESULTS / "trend_japan_vs_comparators.csv", index=False)
R["trend_note"] = ("index = 100 * exp(seasonally adjusted ln visitors - its 2018-07..2019-06 mean); seasonal means "
                   "from 2016-01..2019-06; comparators = simple mean of VN, TH, PH, TW indices")

# ------------------------------------------------------------------ 7. cluster-robust interval for the yen effect
I = json.load(open(RESULTS / "inference_checks.json"))
th0, se_cr1 = I["cr1"]["theta"], I["cr1"]["se"]
q = stats.t.ppf(0.975, 4)
lo, hi = th0 - q * se_cr1, th0 + q * se_cr1
pc = lambda th: float(100 * (np.exp(th * LN09) - 1))
R["yen10_intervals"] = {"point": pc(th0), "dk": [pc(o["theta"] + 1.96 * o["se"]), pc(o["theta"] - 1.96 * o["se"])],
                        "cr1_t4": [pc(hi), pc(lo)],
                        "theta_dk_ci": [o["theta"] - 1.96 * o["se"], o["theta"] + 1.96 * o["se"]],
                        "theta_cr1_t4_ci": [lo, hi]}
g, sg = o["boycott"]
R["boycott_intervals"] = {"gamma_dk_ci": [g - 1.96 * sg, g + 1.96 * sg],
                          "pct_dk": [float(100 * (np.exp(g - 1.96 * sg) - 1)), float(100 * (np.exp(g + 1.96 * sg) - 1))]}

json.dump(R, open(RESULTS / "review_checks.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("event_study", "sample")}, indent=1, default=lambda v: round(float(v), 4)))
print("sample", R["sample"]["rows"])
print("event study pre: mean %.3f max|b| %.3f Wald %.1f df %d p %.3f" % (R["event_study"]["pre_mean"], R["event_study"]["pre_max_abs"],
      W, len(pre_idx), R["event_study"]["pre_wald_p"]))
print(es.round(3).to_string(index=False))
