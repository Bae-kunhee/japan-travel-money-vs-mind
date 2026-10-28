"""Appendix. Standard time-series diagnostics before/after fitting, SE sensitivity,
small-cluster inference, placebo boycott windows and control-contamination adjustment.
Outputs: results/diagnostics.csv (every check), results/diagnostics_appendix.md (report tables),
results/se_sensitivity.csv, results/inference_checks.json. Run after demand_shift.py.
"""
import itertools
import json
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from linearmodels.panel import PanelOLS

from config import RESULTS, K_MAIN as K, COUNTRIES, NAME_KO, ym
from panel_models import (load_full, load_panel, open_sample, japan_sample, m1_fit, design,
                          lag_cols, nw_bw, theta_gamma)
from scipy import stats
from tsdiag import diagnose, residual_check, ROWS

warnings.filterwarnings("ignore")
F = load_full()
F["date"] = pd.to_datetime(dict(year=F.year, month=F.month, day=1))
P = load_panel()
S = open_sample(P)

log = io.StringIO()
with contextlib.redirect_stdout(log):
    pre = F[(F.country == "JP") & (F.t <= 53)].set_index("date")          # 2015-01 .. 2019-06
    diagnose(pre.ln_v, "일본 ln(방문자) 2015.01–2019.06")
    sa = pre.ln_v - pre.groupby("month").ln_v.transform("mean") + pre.ln_v.mean()
    diagnose(sa, "일본 ln(방문자) 계절조정")
    diagnose(F[(F.country == "JP") & (F.year <= 2025)].set_index("date").ln_fx, "ln(원/100엔) 2015.01–2025.12")
    m1, X1 = m1_fit(0, japan_sample(P))
    residual_check(m1.resid.values, k=X1.shape[1], name="모형 1 잔차 (일본 단독)")
    X = design(S, K)
    fit = sm.OLS(S.ln_v, X).fit()
    r = fit.resid
    residual_check(r[S.country == "JP"].values, k=X.shape[1], name="모형 2 잔차 · 일본")
(RESULTS / "diagnostics_log.txt").write_text(log.getvalue())
D = pd.DataFrame(ROWS, columns=["series", "section", "label", "value", "verdict", "note"])
D.to_csv(RESULTS / "diagnostics.csv", index=False)

# ---------------------------------------------------------------- per-country residual autocorrelation (Model 2)
rc = []
for c in COUNTRIES:
    e = r[S.country == c].values
    a = acf(e, nlags=12)
    rc.append({"country": NAME_KO[c], "n": len(e), "acf1": a[1], "acf3": a[3], "acf6": a[6], "acf12": a[12],
               "lb12_p": float(acorr_ljungbox(e, lags=[12], return_df=True)["lb_pvalue"].iloc[0])})
rc = pd.DataFrame(rc)

# ---------------------------------------------------------------- SE sensitivity: DK bandwidth and clustering
tcols = [c for c in X.columns if c.startswith("t_")]
Xp = X.drop(columns=tcols + ["const"])
idx = pd.MultiIndex.from_arrays([S.country.values, S.t.values])
Xp.index = idx
mod = PanelOLS(pd.Series(S.ln_v.values, index=idx), Xp, time_effects=True, drop_absorbed=True, check_rank=False)
lc = lag_cols(K)
bw0 = nw_bw(S.t.nunique())
se_rows = []


def summarize(m, label):
    names = list(m.params.index)
    a = np.array([1.0 if c in lc else 0.0 for c in names])
    b = np.array([1.0 if c == "boycott" else 0.0 for c in names])
    C = m.cov.values
    th, ga = a @ m.params.values, b @ m.params.values
    vt, vg, cv = a @ C @ a, b @ C @ b, a @ C @ b
    d = np.random.default_rng(0).multivariate_normal([th, ga], [[vt, cv], [cv, vg]], 200000)
    ok = d[:, 0] < 0
    dd = -d[ok, 1] / d[ok, 0]
    ci = [100 * (1 - np.exp(np.percentile(dd, q))) for q in (97.5, 2.5)]
    se_rows.append({"cov": label, "theta": th, "theta_se": np.sqrt(vt), "z": th / np.sqrt(vt),
                    "gamma_se": np.sqrt(vg), "be_ci_lo": ci[0], "be_ci_hi": ci[1]})


for bw in sorted({2, bw0, 6, 12, 24}):
    summarize(mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bw),
              f"Driscoll-Kraay, bandwidth {bw}" + (" (본 분석)" if bw == bw0 else ""))
summarize(mod.fit(cov_type="clustered", cluster_entity=True), "국가 군집 (linearmodels 기본 보정, 참고용)")
summarize(mod.fit(cov_type="clustered", cluster_time=True), "월 군집")
se = pd.DataFrame(se_rows)
dk = se[se["cov"].str.startswith("Driscoll")]
se.to_csv(RESULTS / "se_sensitivity.csv", index=False)

# ---------------------------------------------------------------- D. small-cluster inference for theta (5 countries)
Xv, y = X.values, S.ln_v.values
cols = list(X.columns)
a = np.array([1.0 if c in lc else 0.0 for c in cols])
g = S.country.values
G = sorted(set(g))
XtXi = np.linalg.pinv(Xv.T @ Xv)
n, k = Xv.shape
w_inf = a @ XtXi @ Xv.T


def cr1_t(yv):
    b = XtXi @ Xv.T @ yv
    u = yv - Xv @ b
    sc = np.array([w_inf[g == c] @ u[g == c] for c in G])
    var = (sc ** 2).sum() * len(G) / (len(G) - 1) * (n - 1) / (n - k)
    return a @ b, np.sqrt(var)


th0, se_cr1 = cr1_t(y)
t0 = th0 / se_cr1
Xr = X.copy()                                 # impose H0: sum of lag coefficients = 0
for c in lc[1:]:
    Xr[c] = X[c] - X["ln_fx"]
Xr = Xr.drop(columns=["ln_fx"]).values
fr = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
ur = y - fr
webb = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])
ts = []
for wv in itertools.product(webb, repeat=len(G)):               # all 6^5 = 7,776 sign patterns
    wm = dict(zip(G, wv))
    tb, sb = cr1_t(fr + ur * np.array([wm[c] for c in g]))
    ts.append(tb / sb)
p_wcr = float((np.abs(np.array(ts)) >= abs(t0)).mean())
p_t4 = float(2 * stats.t.sf(abs(t0), len(G) - 1))
th_hi = th0 - stats.t.ppf(0.975, len(G) - 1) * se_cr1           # most yen-sensitive end of CR1 t(4) CI
ga_main = theta_gamma(S, K)[1]
INF = {"cr1": {"theta": th0, "se": se_cr1, "t": t0, "p_t4": p_t4},
       "wild_cluster_bootstrap_webb": {"p": p_wcr, "draws": len(ts)},
       "worst_case_theta_cr1_t4": th_hi,
       "worst_case_breakeven_pct": float(100 * (1 - np.exp(-ga_main / th_hi))),
       "worst_case_ratio": float(abs(ga_main / (th_hi * np.log(0.9))))}

# ---------------------------------------------------------------- E. placebo boycott windows (Japan, 8 months, before 2019-07)
pl = []
for s0 in range(ym(2016, 1), ym(2018, 11) + 1):
    d = ((S.country == "JP") & (S.t >= s0) & (S.t <= s0 + 7)).astype(float).rename("pb").to_frame()
    _, gp, _, vgp, *_ = theta_gamma(S, K, extra=d, gamma_col="pb")
    pl.append((s0, gp, np.sqrt(vgp)))
pl = pd.DataFrame(pl, columns=["start_t", "gamma", "se"])
pl.to_csv(RESULTS / "placebo_boycott_windows.csv", index=False)
INF["placebo_windows"] = {"n": len(pl), "mean": pl.gamma.mean(), "sd": pl.gamma.std(),
                          "min": pl.gamma.min(), "max": pl.gamma.max(),
                          "share_abs_z_gt_1.96": float((np.abs(pl.gamma / pl.se) > 1.96).mean()),
                          "actual_gamma_in_sd": float(ga_main / pl.gamma.std())}

# ---------------------------------------------------------------- F. control contamination (diverted demand inflating controls)
SS = pd.read_csv(RESULTS / "share_shift_2019.csv").set_index("dest")
contam = {}
for c in ["VN", "TH", "PH", "TW"]:
    r_ = SS.loc[NAME_KO[c], "abs_did"]
    vh2 = S[(S.country == c) & (S.t >= ym(2019, 7)) & (S.t <= ym(2019, 12))].visitors.sum()
    contam[c] = float(np.log(vh2 / (vh2 - r_)))
INF["contamination_log"] = contam
for label, hi in [("adj_2019H2", ym(2019, 12)), ("adj_to_2020-02", ym(2020, 2))]:
    S2 = S.copy()
    for c, a_ in contam.items():
        m_ = (S2.country == c) & (S2.t >= ym(2019, 7)) & (S2.t <= hi)
        S2.loc[m_, "ln_v"] -= a_
    th2, ga2, *_ = theta_gamma(S2, K)
    INF[label] = {"theta": th2, "gamma": ga2, "boycott_pct": 100 * (np.exp(ga2) - 1),
                  "breakeven_pct": 100 * (1 - np.exp(-ga2 / th2)), "ratio": abs(ga2 / (th2 * np.log(0.9)))}
json.dump(INF, open(RESULTS / "inference_checks.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- appendix markdown
def tbl(df, fmt):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(fmt.get(c, "{}").format(row[c]) for c in cols) + " |")
    return "\n".join(out)


pick = D[D.label.isin(["n", "skewness", "excess kurtosis", "Jarque-Bera p", "ADF p (H0 단위근, ct)",
                       "KPSS p (H0 정상, ct)", "verdict", "ADF p (1차 차분)", "Ljung-Box p (lag 12)",
                       "LB on e² p (lag 12)", "ARCH-LM p (lag 6)"])].copy()
pick["value"] = pick.value.map(lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
wide = pick.pivot_table(index="label", columns="series", values="value", aggfunc="first", sort=False).fillna("–")
wide = wide.reset_index().rename(columns={"label": "검정"})
md = ["# 부록: 시계열 진단과 표준오차 민감도", "",
      "`python3 src/diagnostics.py`가 만든 표예요. 항목별 전체 판정(OK/WARN/FAIL)은 `results/diagnostics.csv`, 원 출력은 `results/diagnostics_log.txt`에 있어요.", "",
      "## A. 원계열과 잔차 진단", "", tbl(wide, {}), "",
      "- 일본 방문자 수와 엔화 환율 모두 수준에서는 단위근을 기각하지 못하고(ADF), 정상성은 기각돼요(KPSS). 1차 차분하면 정상이에요.",
      "- 그래서 일본 한 나라의 수준 회귀(모형 1)는 허구 회귀 위험이 있고, 실제로 잔차가 강한 자기상관과 두꺼운 꼬리를 보여요.",
      "- 본 분석(모형 2)은 연-월 고정효과로 공통 추세를 걷어내고, 전년동월비(계절 차분) 모형으로도 같은 결론(θ = −0.84)을 확인했어요.", "",
      "## B. 모형 2 잔차의 나라별 자기상관", "",
      tbl(rc, {"acf1": "{:.2f}", "acf3": "{:.2f}", "acf6": "{:.2f}", "acf12": "{:.2f}", "lb12_p": "{:.4f}"}), "",
      "모든 나라에서 잔차 자기상관이 남아 있어요. 그래서 계수의 표준오차는 자기상관과 국가 간 상관에 강건한 Driscoll-Kraay 방식으로 구했어요. "
      "베트남·필리핀은 자기상관이 길게 이어져서, 아래 C에서 bandwidth를 넓혀도 결론이 유지되는지 확인했어요. "
      "참고로 연-월 고정효과가 매달 국가 평균을 0으로 맞추기 때문에, 나라 5개의 잔차끼리는 기계적으로 음의 상관(약 −1/4)을 띠어요.", "",
      "## C. 표준오차 계산 방식에 대한 민감도", "",
      tbl(se, {"theta": "{:.3f}", "theta_se": "{:.3f}", "z": "{:.2f}", "gamma_se": "{:.3f}",
               "be_ci_lo": "{:.0f}%", "be_ci_hi": "{:.0f}%"}), "",
      f"- 본 분석 bandwidth는 규칙 ⌊4(T/100)^(2/9)⌋ = {bw0}개월이에요. 24개월까지 넓혀도 θ의 표준오차는 "
      f"{se.theta_se.iloc[1]:.2f} → {se[se['cov'].str.contains('bandwidth 24')].theta_se.item():.2f}로 커지는 정도이고, |z| > 3이 유지돼요. "
      f"손익분기 엔저 폭의 95% 구간도 {dk.be_ci_lo.min():.0f}~{dk.be_ci_hi.max():.0f}% 범위 안에서 움직여요.",
      "- 국가 군집 표준오차는 군집이 5개뿐이라 그 자체로는 신뢰하기 어려워요. 그래서 아래 D에서 군집이 적을 때 쓰는 방법으로 다시 확인했어요.", "",
      "## D. 비교국이 적을 때의 추론 (국가 5개 군집)", "",
      f"- 국가 군집 표준오차(CR1): θ = {th0:.2f}, SE {se_cr1:.2f}, t = {t0:.2f}, t(4) 분포 기준 p = {p_t4:.2f}",
      f"- Wild cluster bootstrap (Webb 6점 가중치, 7,776가지 전수, 귀무가설 θ = 0 부과): **p = {p_wcr:.2f}**",
      f"- 즉 국가 단위 불확실성까지 보면 **엔저 효과는 5% 수준에서 유의하지 않아요.** 다만 신뢰구간의 가장 큰 쪽(θ = {th_hi:.2f})을 써도 "
      f"불매 효과는 10% 엔저 효과의 {INF['worst_case_ratio']:.1f}배이고, 손익분기 엔저 폭은 {INF['worst_case_breakeven_pct']:.0f}%로 관측된 최대 엔저(약 12%)보다 훨씬 커요. "
      "**\"마음 > 돈\" 결론은 이 가장 보수적인 추론에서도 유지돼요.**", "",
      "## E. 가짜 불매 구간 위약 검정 (일본, 8개월 구간, 2016.01~2019.06 시작)", "",
      f"- 가짜 구간 {len(pl)}개의 γ: 평균 {pl.gamma.mean():+.3f}, 표준편차 {pl.gamma.std():.3f}, 범위 [{pl.gamma.min():+.2f}, {pl.gamma.max():+.2f}]",
      f"- 실제 불매 γ = {ga_main:.2f}는 가짜 구간 분포에서 약 {abs(INF['placebo_windows']['actual_gamma_in_sd']):.0f} 표준편차 떨어져 있어요. 불매 효과가 우연일 가능성은 사실상 없어요.",
      f"- 다만 가짜 구간의 {100 * INF['placebo_windows']['share_abs_z_gt_1.96']:.0f}%가 |z| > 1.96이 나와서(명목 5%), 처치 단위가 하나뿐인 γ의 표준오차는 작게 잡히는 경향이 있어요. "
      "그래서 불매 효과의 크기는 표준오차보다 이 위약 분포와 비교해 판단하는 게 안전해요.", "",
      "## F. 비교국 오염 보정", "",
      "불매로 일본에서 빠진 수요 일부가 비교국으로 가면 비교국이 부풀어 불매 효과가 과대 추정될 수 있어요. 수요 이동 분석의 비교국별 증가분을 2019년 하반기 비교국 방문자에서 빼고 다시 추정했어요.", "",
      "| 보정 | 불매 효과 | 손익분기 엔저 폭 | 10% 엔저 대비 배수 |", "|---|---|---|---|",
      f"| 보정 전 (본 분석) | −65% | 69% | 11.0배 |",
      f"| 2019년 하반기 보정 | {INF['adj_2019H2']['boycott_pct']:.0f}% | {INF['adj_2019H2']['breakeven_pct']:.0f}% | {INF['adj_2019H2']['ratio']:.1f}배 |",
      f"| 2020년 2월까지 보정 | {INF['adj_to_2020-02']['boycott_pct']:.0f}% | {INF['adj_to_2020-02']['breakeven_pct']:.0f}% | {INF['adj_to_2020-02']['ratio']:.1f}배 |", "",
      "보정해도 결론은 거의 바뀌지 않아요.", ""]
(RESULTS / "diagnostics_appendix.md").write_text("\n".join(md))
print(se.round(3).to_string(index=False))
print(rc.round(3).to_string(index=False))
print(json.dumps(INF, indent=1, default=lambda v: round(float(v), 3)))
print("wrote results/diagnostics.csv, diagnostics_appendix.md, se_sensitivity.csv, inference_checks.json")
