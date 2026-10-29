"""Step 6. How does demand build up after the yen gets 10% cheaper than the comparison currencies?
(results/response_curves.json, figures/fig7_response_curves.png is drawn in figures.py)

1. Panel model, cumulative response h = 0..10 months after a sustained 10% relative yen fall.
   The unrestricted lag coefficients are collinear, so the path uses a quadratic Almon (polynomial
   distributed lag) restriction beta_k = a0 + a1 k + a2 k^2. The 10-month endpoint is the same as the
   main analysis; the shape in between is less certain than the endpoint.
2. The same response in market-share terms: in the panel, ln(share) = ln V - ln(all departures) and
   the denominator is common to every country in a month, so it is absorbed by the year-month FE and
   the share elasticity equals the count elasticity. Expressed in %p of Korean departures using
   Japan's 2018 share.
3. Japan-only share model (ln of Japan's share of all departures on relative-FX lags), several control
   sets: shows why a single series cannot pin down the FX effect (the yen-weak years coincide with the
   post-reopening period) while the boycott effect is stable.
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import openpyxl

from config import RESULTS, KTO_XLSX, K_MAIN as K, T0, T1, ym
from panel_models import load_full, load_panel, open_sample, lag_cols, m2_fit, nw_bw

warnings.filterwarnings("ignore")
L = lag_cols(K)
pct = lambda x: 100 * (np.exp(x * np.log(0.9)) - 1)        # % change for a 10% cheaper currency
R = {}

# ------------------------------------------------------------------ 1. panel response curve (Almon, degree 2)
S = open_sample(load_panel())


def almon(df, deg, KK=K, **kw):
    LL = lag_cols(KK)
    Z = pd.DataFrame({f"z{j}": sum((k ** j) * df[LL[k]] for k in range(KK + 1)) for j in range(deg + 1)}, index=df.index)
    D = df.copy()
    D[LL] = 0.0
    o, fit, X, V = m2_fit(D, KK, extra=Z, **kw)
    cols = list(X.columns)
    idx = [cols.index(f"z{j}") for j in range(deg + 1)]
    return fit.params.values[idx], V[np.ix_(idx, idx)], fit.aic


a, Va, aic2 = almon(S, 2)
aic_unres = m2_fit(S, K)[0]["aic"]
curve = []
for h in range(K + 1):
    w = np.array([sum(k ** j for k in range(h + 1)) for j in range(3)])
    c, s = float(w @ a), float(np.sqrt(w @ Va @ w))
    curve.append({"h": h, "cum_log": c, "se": s, "pct": pct(c), "lo": pct(c + 1.96 * s), "hi": pct(c - 1.96 * s)})
R["panel_curve"] = curve
R["panel_curve_aic"] = {"almon2": aic2, "unrestricted": aic_unres}

# the same curve under other specifications: the build-up over the first months is common to all,
# the peak-and-decline shape after that is not (the last lag tends to turn negative whatever K is)
alts = {"시차 12개월": (S, 12, {}), "시차 6개월": (S, 6, {}), "달력 월 고정효과만": (S, K, {"use_event": False}),
        "대만 제외": (S[S.country != "TW"], K, {}), "코로나 이전 기간만": (S[S.t <= 61], K, {"use_event": False})}
R["alt_curves"] = {}
for name, (df, kk, kw) in alts.items():
    aa, _, _ = almon(df, 2, kk, **kw)
    R["alt_curves"][name] = [pct(float(np.array([sum(k ** j for k in range(h + 1)) for j in range(3)]) @ aa)) for h in range(kk + 1)]
R["peak_ratio"] = {"peak_pct": max(p["pct"] for p in curve),
                   "boycott_over_peak": float(-json.load(open(RESULTS / "money_vs_mind.json"))["main"]["boycott"]
                                              / max(p["cum_log"] * np.log(0.9) for p in curve))}

# ------------------------------------------------------------------ 2. same response in %p of Korean departures
wb = openpyxl.load_workbook(KTO_XLSX, data_only=True)
ws = wb["Asia"]
rows, yr = [], None
for r in ws.iter_rows(min_row=4, values_only=True):
    if r[0]:
        yr = int(str(r[0])[:4])
    if r[1] is None or r[1] == "누계":
        continue
    rows.append(dict(year=yr, month=int(str(r[1]).replace("월", "")), tot=r[2]))
TOT = pd.DataFrame(rows)
F = load_full()
jp = F[F.country == "JP"].merge(TOT, on=["year", "month"])
share18 = float((jp[jp.year == 2018].visitors / jp[jp.year == 2018].tot).mean())
R["japan_share_2018"] = share18
R["share_curve_pp"] = [{"h": p["h"], "pp": 100 * share18 * p["pct"] / 100,
                        "lo": 100 * share18 * p["lo"] / 100, "hi": 100 * share18 * p["hi"] / 100} for p in curve]

# ------------------------------------------------------------------ 3. Japan-only share model (why it is not used for the FX effect)
W = F.pivot_table(index="t", columns="country", values="ln_fx").sort_index()
rel = (W.JP - W[["VN", "TH", "PH", "TW"]].mean(axis=1)).rename("rel")
RL = ["rel"] + [f"rel_l{k}" for k in range(1, K + 1)]
RF = pd.concat([rel] + [rel.shift(k).rename(f"rel_l{k}") for k in range(1, K + 1)], axis=1).reset_index()
J = jp.merge(RF, on="t")
J["ln_s"] = np.log(J.visitors / J.tot)
J = J[(J.t >= T0) & (J.t <= T1) & (J.closed == 0)].dropna(subset=["ln_s"] + RL).copy()
md = pd.get_dummies(pd.Categorical(J.month, categories=range(1, 13)), prefix="m", drop_first=True, dtype=float)
md.index = J.index
post = (J.t >= ym(2022, 10)).astype(float).rename("post")
eb = pd.get_dummies(np.minimum(J.event.fillna(-1), 18).astype(int), prefix="e", dtype=float).drop(columns=["e_-1", "e_18"], errors="ignore")
specs = {"month FE + boycott": None, "+ post-reopening level": post.to_frame(),
         "+ post + linear trend": pd.concat([post, (J.t / 12).rename("trend")], axis=1),
         "+ post + months-since-reopening": pd.concat([post, eb], axis=1)}
R["japan_only_share"] = {}
for name, extra in specs.items():
    X = sm.add_constant(pd.concat([J[RL], J[["boycott"]], md] + ([extra] if extra is not None else []), axis=1))
    m = sm.OLS(J.ln_s, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_bw(len(J))})
    av = np.array([1.0 if c in RL else 0.0 for c in X.columns])
    th, se = float(av @ m.params.values), float(np.sqrt(av @ m.cov_params().values @ av))
    R["japan_only_share"][name] = {"theta": th, "se": se, "yen10_pct": pct(th),
                                   "ci": [pct(th + 1.96 * se), pct(th - 1.96 * se)],
                                   "boycott_pct": float(100 * (np.exp(m.params["boycott"]) - 1)), "n": int(m.nobs)}

json.dump(R, open(RESULTS / "response_curves.json", "w"), ensure_ascii=False, indent=1, default=float)
print("AIC almon2 vs unrestricted:", round(aic2, 2), round(aic_unres, 2), "| boycott / peak yen effect:", round(R["peak_ratio"]["boycott_over_peak"], 1))
for p in curve:
    print(f"h={p['h']:2d}  {p['pct']:+6.1f}%  [{p['lo']:+6.1f}, {p['hi']:+6.1f}]")
print("Japan share 2018:", round(share18, 3))
for k, v in R["japan_only_share"].items():
    print(f"{k:34s} 10% yen -> {v['yen10_pct']:+5.1f}% [{v['ci'][0]:+5.1f}, {v['ci'][1]:+5.1f}]  boycott {v['boycott_pct']:.0f}%")
