"""Step 2. Diagnostics, Model 1 vs Model 2, robustness, decomposition (results/main_models.json).

Q: how big is the yen ('money') effect once revenge travel is taken out?
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import RESULTS, COUNTRIES
from panel_models import (load_full, load_panel, open_sample, japan_sample, lag_cols,
                          m1_fit, m2_fit, dk_cov, sum_vector)

warnings.filterwarnings("ignore")
OUT = RESULTS / "main_models.json"
R = {}
P = load_panel()

# ============================================================== 0. diagnostics (Japan, 2015-01..2019-06)
D = load_full()
pre = D[(D.country == "JP") & (D.t <= 53)].copy()
diag = {}
for var in ["ln_v", "ln_fx"]:
    y = pre[var].values
    # remove calendar-month means before unit-root tests (strong seasonality)
    y_sa = y - pre.groupby("month")[var].transform("mean").values + y.mean()
    diag[var] = {"adf_p_level_trend": float(adfuller(y_sa, regression="ct", autolag="AIC")[1]),
                 "kpss_p_level_trend": float(kpss(y_sa, regression="ct", nlags="auto")[1]),
                 "adf_p_diff": float(adfuller(np.diff(y_sa), regression="c", autolag="AIC")[1])}
X = pd.get_dummies(pre.month, prefix="m", drop_first=True, dtype=float)
X["trend"] = pre.t.values
f = sm.OLS(pre.ln_v.values, sm.add_constant(X)).fit()
diag["seasonality_F_p"] = float(f.f_test([f"{c} = 0" for c in X.columns if c.startswith("m_")]).pvalue)
yv = pre.ln_v.values - pre.groupby("month").ln_v.transform("mean").values
diag["engle_granger_p"] = float(coint(yv, pre.ln_fx.values, trend="ct")[1])
ann = D[D.year.between(2015, 2019)].groupby(["country", "year"]).visitors.sum().unstack()
diag["growth_2015_2018_pct"] = {c: float(100 * (ann.loc[c, 2018] / ann.loc[c, 2015] - 1)) for c in COUNTRIES}
R["diagnostics"] = diag

# ============================================================== 1. Model 1: Japan only
J = japan_sample(P)
aic1 = {K: m1_fit(K, J, cov=False)[0].aic for K in range(13)}
bic1 = {K: m1_fit(K, J, cov=False)[0].bic for K in range(13)}
K1 = min(min(aic1, key=aic1.get), min(bic1, key=bic1.get))   # smaller of the AIC/BIC picks
m1, X1 = m1_fit(K1, J)
lc = lag_cols(K1)
a = sum_vector(X1.columns, lc)
R["model1"] = {"K": K1, "aic_K": int(min(aic1, key=aic1.get)), "bic_K": int(min(bic1, key=bic1.get)),
               "theta": float(a @ m1.params.values), "se": float(np.sqrt(a @ m1.cov_params().values @ a)),
               "n": int(m1.nobs), "r2_adj": float(m1.rsquared_adj),
               "boycott": float(m1.params["boycott"]), "boycott_se": float(m1.bse["boycott"]),
               "betas": {c: [float(m1.params[c]), float(m1.bse[c])] for c in lc},
               "ljungbox12_p": float(acorr_ljungbox(m1.resid, lags=[12], return_df=True)["lb_pvalue"].iloc[0]),
               "aic": {int(k): float(v) for k, v in aic1.items()}}

# ============================================================== 2. Model 2: panel
S = open_sample(P)
aic2 = {K: m2_fit(S, K)[0]["aic"] for K in range(13)}
K = int(min(aic2, key=aic2.get))
R["model2_aic"] = {int(k): v for k, v in aic2.items()}
R["K2"] = K

main, fit_main, X_main, V_main = m2_fit(S, K)
R["model2_event"] = main
R["model2_calendar"] = m2_fit(S, K, use_event=False)[0]
m1b, X1b = m1_fit(K, J)          # Model 1 with the same K (apples-to-apples)
a = sum_vector(X1b.columns, lag_cols(K))
R["model1_sameK"] = {"theta": float(a @ m1b.params.values),
                     "se": float(np.sqrt(a @ m1b.cov_params().values @ a)), "K": K}

# ============================================================== 3. robustness
rob = {}
for c in COUNTRIES[1:]:
    rob[f"drop_{c}"] = m2_fit(S[S.country != c], K)[0]
Sa = P[(P.closed_alt == 0)].dropna(subset=["ln_v"]).copy()
rob["alt_reopen"] = m2_fit(Sa, K, event_col="event_alt")[0]
cut = 93 + 12   # 2022-10 + 12 months = 2023-10: keep only months well past reopening
rob["drop_transition_calendarFE"] = m2_fit(S[(S.t < 62) | (S.t >= cut)], K, use_event=False)[0]
rob["country_trends"] = m2_fit(S, K, trends=True)[0]
rob["drop_first3"] = m2_fit(S[~(S.event < 3)], K)[0]
pl, plfit, plX, plV = m2_fit(S.dropna(subset=["ln_fx_f3"]), K, leads=3)     # placebo: future FX
a = sum_vector(plX.columns, [f"ln_fx_f{k}" for k in range(1, 4)])
rob["placebo_leads"] = {"lead_sum": float(a @ plfit.params.values), "se": float(np.sqrt(a @ plV @ a)),
                        "theta_lags": pl["theta"], "theta_lags_se": pl["se"]}
rob["drop_TH_PH_2025"] = m2_fit(S[~(S.country.isin(["TH", "PH"]) & (S.year == 2025))], K)[0]
R["robustness"] = rob

# ============================================================== 4. decomposition for Japan (vs 2018)
params = fit_main.params
JP = S[S.country == "JP"].copy()
Xjp = X_main.loc[JP.index]
lc = [c for c in lag_cols(K) if c in X_main.columns]
tcols = [c for c in X_main.columns if c.startswith("t_")]
ecols = [c for c in X_main.columns if c.startswith("e_")]
JP["fx_part"] = Xjp[lc].values @ params[lc].values
JP["time_part"] = Xjp[tcols].values @ params[tcols].values
JP["event_part"] = Xjp[ecols].values @ params[ecols].values
JP["boy_part"] = Xjp["boycott"].values * params["boycott"]
JP["resid"] = fit_main.resid.loc[JP.index]
B = JP[JP.year == 2018]


def mean_matched(df):
    return df.groupby("month")[["ln_v", "fx_part", "time_part", "event_part", "boy_part", "resid"]].mean().mean()


def decompose(years):
    Pp = JP[JP.year.isin(years)]
    d = mean_matched(Pp) - mean_matched(B)
    total = float(d.ln_v)
    dec = {"years": years, "total_log": total, "total_pct": 100 * (np.exp(total) - 1),
           "fx_log": float(d.fx_part), "common_log": float(d.time_part + d.event_part),
           "calendar_log": float(d.time_part), "revenge_curve_log": float(d.event_part),
           "japan_specific_log": float(d.resid + d.boy_part)}
    for k in ["fx", "common", "japan_specific"]:
        dec[f"{k}_share"] = dec[f"{k}_log"] / total
    # yen change measured on the same lag window the model uses (mean of lags 0..K)
    lagm = lambda df: df[lc].mean(axis=1).groupby(df.month).mean().mean()
    dec["yen_change_pct_lagwindow"] = float(100 * (np.exp(lagm(Pp) - lagm(B)) - 1))
    dec["yen_change_pct_contemp"] = float(100 * (np.exp(Pp.groupby("month").ln_fx.mean().mean()
                                                       - B.groupby("month").ln_fx.mean().mean()) - 1))
    b1 = np.array([m1b.params[c] for c in lc])
    naive = float((Pp[lc].values @ b1).mean() - (B[lc].values @ b1).mean())
    dec["naive_fx_log"] = naive
    dec["naive_fx_share"] = naive / total
    return dec


R["decomposition"] = decompose([2024, 2025])
R["decomposition_2023"] = decompose([2023])
R["model2_cum_betas"] = list(np.cumsum([fit_main.params[c] for c in lc]))
R["model1_cum_betas_sameK"] = list(np.cumsum([m1b.params[c] for c in lc]))

# ============================================================== 5. year-on-year (seasonal difference) panel
Y = P.sort_values(["country", "t"]).copy()
Y["d_lnv"] = Y.ln_v - Y.groupby("country").ln_v.shift(12)
for c in ["ln_fx"] + [f"ln_fx_l{k}" for k in range(1, 13)]:
    Y["d_" + c] = Y[c] - Y.groupby("country")[c].shift(12)
Y["e_lag12"] = Y.groupby("country").event.shift(12)
ok_pre = (Y.t <= 61)                                         # both t and t-12 pre-pandemic
ok_post = (Y.event >= 12) & (Y.e_lag12 >= 6)                 # both well past reopening
Yd = Y[(ok_pre | ok_post) & Y.d_lnv.notna()].dropna(subset=["d_ln_fx_l" + str(K)]).copy()
Xy = pd.concat([Yd[["d_" + c for c in lag_cols(K)]],
                ((Yd.country == "JP") & Yd.t.between(54, 61)).astype(float).rename("boy_yoy"),
                pd.get_dummies(Yd.t, prefix="t", dtype=float).iloc[:, 1:]], axis=1)
Xy = sm.add_constant(Xy)
fy = sm.OLS(Yd.d_lnv, Xy).fit()
Vy = dk_cov(Yd.d_lnv, Xy, Yd)
a = np.array([1.0 if c.startswith("d_ln_fx") else 0.0 for c in Xy.columns])
R["robustness"]["yoy_panel"] = {"theta": float(a @ fy.params.values), "se": float(np.sqrt(a @ Vy @ a)),
                                "n": int(fy.nobs)}

R["event_curve"] = {c: float(params[c]) for c in ecols}       # revenge-travel curve
R["model2_resid"] = {"jp_ljungbox12_p": float(acorr_ljungbox(fit_main.resid.loc[JP.index].values, lags=[12],
                                                            return_df=True)["lb_pvalue"].iloc[0])}

json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1, default=float)
for k in ["model1", "model1_sameK", "model2_calendar", "model2_event"]:
    v = R[k]
    print(k, {kk: v[kk] for kk in v if kk in ("K", "theta", "se", "n", "boycott")})
print("K2 =", K)
for k, v in R["robustness"].items():
    print("  rob", k, {kk: round(vv, 3) for kk, vv in v.items() if kk in ("theta", "se", "lead_sum")})
print("decomposition 2024-25 vs 2018:", {k: round(v, 3) for k, v in R["decomposition"].items() if k != "years"})
print("wrote", OUT)
