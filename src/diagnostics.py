"""Appendix. Standard time-series diagnostics before/after fitting + SE sensitivity.
Outputs: results/diagnostics.csv (every check), results/diagnostics_appendix.md (report tables).
"""
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

from config import RESULTS, K_MAIN as K, COUNTRIES, NAME_KO
from panel_models import (load_full, load_panel, open_sample, japan_sample, m1_fit, design,
                          lag_cols, nw_bw)
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
summarize(mod.fit(cov_type="clustered", cluster_entity=True), "국가 군집 (군집 5개, 참고용)")
summarize(mod.fit(cov_type="clustered", cluster_time=True), "월 군집")
se = pd.DataFrame(se_rows)
se.to_csv(RESULTS / "se_sensitivity.csv", index=False)

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
      f"손익분기 엔저 폭의 95% 구간도 {se.be_ci_lo.min():.0f}~{se.be_ci_hi.max():.0f}% 범위 안에서 움직여요.",
      "- 국가 군집 표준오차는 군집이 5개뿐이라 신뢰하기 어려워요(군집 수가 적으면 과소·과대 추정이 모두 심함). 그래도 이 기준에서는 θ의 |z|가 2 아래로 내려가니, "
      "\"비교국이 적다\"는 한계와 함께 정직하게 밝히는 게 좋아요.", ""]
(RESULTS / "diagnostics_appendix.md").write_text("\n".join(md))
print(se.round(3).to_string(index=False))
print(rc.round(3).to_string(index=False))
print("wrote results/diagnostics.csv, diagnostics_appendix.md, se_sensitivity.csv")
