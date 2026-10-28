"""Step 3. Money (yen) vs mind (boycott / social issues): effect sizes and the break-even yen fall
(results/money_vs_mind.json).

break-even: the yen fall d such that theta * ln(1 - d) = -gamma
"""
import json
import warnings

import numpy as np
import pandas as pd

from config import RESULTS, K_MAIN as K, ym
from panel_models import (load_panel, open_sample, japan_sample, lag_cols, m1_fit, m2_fit,
                          theta_gamma, breakeven, sum_vector)

warnings.filterwarnings("ignore")
OUT = RESULTS / "money_vs_mind.json"
P = load_panel()
S = open_sample(P)
R = {}

# ------------------------------------------------------------------ 1. main: average boycott effect
th, ga, vt, vg, cov, fit, X, V = theta_gamma(S, K)
R["main"] = {"theta": th, "theta_se": np.sqrt(vt), "boycott": ga, "boycott_se": np.sqrt(vg),
             "boycott_pct": 100 * (np.exp(ga) - 1),
             "boycott_ci_pct": [100 * (np.exp(ga - 1.96 * np.sqrt(vg)) - 1), 100 * (np.exp(ga + 1.96 * np.sqrt(vg)) - 1)],
             "yen10_pct": 100 * (np.exp(th * np.log(0.9)) - 1),
             "ratio_boycott_to_yen10": ga / (th * np.log(0.9)),
             "breakeven": breakeven(th, ga, vt, vg, cov)}
# ratio CI by simulation from the joint normal of (theta, gamma)  (Fieller-style check)
rng = np.random.default_rng(0)
draws = rng.multivariate_normal([th, ga], [[vt, cov], [cov, vg]], 200000)
ok = draws[:, 0] < 0
dd = -draws[ok, 1] / draws[ok, 0]
R["main"]["breakeven_sim_ci_fall_pct"] = [float(100 * (1 - np.exp(np.percentile(dd, q)))) for q in (97.5, 2.5)]
R["main"]["share_draws_theta_negative"] = float(ok.mean())

# ------------------------------------------------------------------ 2. boycott month by month (peak vs average)
Sm = S.copy()
months = list(range(ym(2019, 7), ym(2020, 2) + 1))
extra = pd.DataFrame({f"boy_{t}": ((Sm.country == "JP") & (Sm.t == t)).astype(float) for t in months}, index=Sm.index)
Sm["boycott"] = 0.0
o, fitm, Xm, Vm = m2_fit(Sm, K, extra=extra)
cols = list(Xm.columns)
R["boycott_monthly"] = {}
for t in months:
    i = cols.index(f"boy_{t}")
    R["boycott_monthly"][f"{2015 + t // 12}-{t % 12 + 1:02d}"] = [float(fitm.params[f"boy_{t}"]), float(np.sqrt(Vm[i, i]))]
peak_cols = [f"boy_{ym(2019, m)}" for m in (10, 11, 12)]          # full-blown boycott, pre-COVID
a = sum_vector(cols, peak_cols) / 3
bth = sum_vector(cols, [c for c in lag_cols(K) if c in cols])
pk, thm = a @ fitm.params.values, bth @ fitm.params.values
R["boycott_peak"] = {"gamma": float(pk), "pct": float(100 * (np.exp(pk) - 1)),
                     "breakeven": breakeven(thm, pk, bth @ Vm @ bth, a @ Vm @ a, bth @ Vm @ a)}

# ------------------------------------------------------------------ 3. a second social issue: Fukushima discharge (from 2023-09)
for label, end in [("fukushima_4m", ym(2023, 12)), ("fukushima_6m", ym(2024, 2))]:
    fk = ((S.country == "JP") & (S.t >= ym(2023, 9)) & (S.t <= end)).astype(float).rename("fukushima")
    _, gaf, _, vgf, *_ = theta_gamma(S, K, extra=fk.to_frame(), gamma_col="fukushima")
    R[label] = {"gamma": gaf, "se": np.sqrt(vgf), "pct": 100 * (np.exp(gaf) - 1)}

# ------------------------------------------------------------------ 4. what moved Japan in 2019H2? (money vs mind, same period)
JP = S[S.country == "JP"]
H = JP[(JP.t >= ym(2019, 7)) & (JP.t <= ym(2019, 12))]
Bm = JP[(JP.year == 2018) & JP.month.isin(range(7, 13))]
dv = H.set_index("month").ln_v - Bm.set_index("month").ln_v
lagmean = lambda df: df[lag_cols(K)].mean(axis=1).values
dfx = (lagmean(H) - lagmean(Bm)).mean()
R["h2_2019"] = {"actual_log": float(dv.mean()), "actual_pct": float(100 * (np.exp(dv.mean()) - 1)),
                "yen_change_pct": float(100 * (np.exp(dfx) - 1)),
                "money_log": float(th * dfx), "mind_log": float(ga)}

# ------------------------------------------------------------------ 5. naive version (Japan-only Model 1) for contrast
m1, X1 = m1_fit(K, japan_sample(P))
a1, b1 = sum_vector(X1.columns, lag_cols(K)), sum_vector(X1.columns, ["boycott"])
C1 = m1.cov_params().values
th1, ga1 = a1 @ m1.params.values, b1 @ m1.params.values
R["naive"] = {"theta": th1, "boycott": ga1, "breakeven": breakeven(th1, ga1, a1 @ C1 @ a1, b1 @ C1 @ b1, a1 @ C1 @ b1)}

# ------------------------------------------------------------------ 6. robustness of the break-even
rob = {}
for Kk in (0, 3, 6, 9, 12):
    t_, g_, a_, b_, c_, *_ = theta_gamma(S, Kk)
    rob[f"K={Kk}"] = breakeven(t_, g_, a_, b_, c_)["fall_pct"]
for c in ["VN", "TH", "PH", "TW"]:
    t_, g_, a_, b_, c_, *_ = theta_gamma(S[S.country != c], K)
    rob[f"drop_{c}"] = breakeven(t_, g_, a_, b_, c_)["fall_pct"]
t_, g_, a_, b_, c_, *_ = theta_gamma(S, K, use_event=False)
rob["calendarFE"] = breakeven(t_, g_, a_, b_, c_)["fall_pct"]
Sb = S[~((S.country == "JP") & (S.t >= ym(2020, 1)) & (S.t <= ym(2020, 2)))]   # drop Jan-Feb 2020 (COVID onset)
t_, g_, a_, b_, c_, *_ = theta_gamma(Sb, K)
rob["boycott_to_2019-12_only"] = breakeven(t_, g_, a_, b_, c_)["fall_pct"]
R["breakeven_robustness"] = rob

# ------------------------------------------------------------------ 7. "price-equivalent" table for marketing
R["discount_table"] = {f"{p}%": float(100 * (np.exp(th * np.log(1 - p / 100)) - 1)) for p in (5, 10, 20, 30, 50)}

json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
m = R["main"]
print(f"theta={m['theta']:.3f} (se {m['theta_se']:.3f})  boycott={m['boycott']:.3f} ({m['boycott_pct']:.0f}%)")
print(f"break-even yen fall = {m['breakeven']['fall_pct']:.1f}%  sim 95% CI {m['breakeven_sim_ci_fall_pct']}")
print(f"peak boycott {R['boycott_peak']['pct']:.0f}% -> break-even {R['boycott_peak']['breakeven']['fall_pct']:.1f}%")
print("break-even robustness:", {k: round(v, 1) for k, v in rob.items()})
print("wrote", OUT)
