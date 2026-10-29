"""Robustness: does a flexible functional form (XGBoost) change the money-vs-mind conclusion?
Design and decision rules are fixed in docs/xgb_prereg.md (written before results).
Outputs: results/xgb_robustness.json, results/xgb_oos_rolling.csv, results/xgb_pd_fx.csv,
         figures/fig7_xgb_linearity.png

FWL: remove Model 2 fixed effects (country x calendar month, year-month, months-since-reopening bins)
from y and from each regressor; fit g(residual X) -> residual y with OLS (= Model 2) or XGBoost.
Prediction for a new row: fe_y(row) + g(x_row - fe_x(row)).
"""
import sys, json, warnings
import numpy as np, pandas as pd, xgboost as xgb
from scipy import stats
warnings.filterwarnings("ignore")
from pathlib import Path
from panel_models import open_sample, load_panel, design, lag_cols
from config import ym, RESULTS, FIGURES
PARAMS = dict(max_depth=2, n_estimators=400, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=3, reg_lambda=1.0, random_state=2026, n_jobs=4)
LN09 = np.log(0.9)

S = open_sample(load_panel()).reset_index(drop=True)
X = design(S, 10)
XC = lag_cols(10) + ["boycott"]
FE = [c for c in X.columns if c not in XC]
F, XV, Y = X[FE].values, X[XC].values, S.ln_v.values


def fe_fit(rows):
    """FE projection coefficients estimated on `rows` for y and every regressor."""
    B = np.linalg.lstsq(F[rows], np.column_stack([Y[rows], XV[rows]]), rcond=None)[0]
    return B                                   # (n_FE, 1 + n_X)


def resid(B, rows):
    P = F[rows] @ B
    return Y[rows] - P[:, 0], XV[rows] - P[:, 1:], P[:, 0]


class Lin:
    def fit(self, R, r):
        self.b = np.linalg.lstsq(R, r, rcond=None)[0]; return self
    def predict(self, R):
        return R @ self.b


def make_xgb(**kw):
    p = dict(PARAMS); p.update(kw)
    return xgb.XGBRegressor(objective="reg:squarederror", **p)


# ---------------------------------------------------------------- 1. rolling-origin out-of-sample
origins = [ym(y, m) for y in range(2018, 2026) for m in (6, 12) if ym(2018, 6) <= ym(y, m) <= ym(2025, 6)]
rows = []
for o in origins:
    test_t = range(o + 1, o + 7)
    tr = ((S.t <= o) | ((S.country != "JP") & S.t.isin(test_t))).values
    te = ((S.country == "JP") & S.t.isin(test_t)).values
    if te.sum() == 0:
        continue
    B = fe_fit(tr)
    ry, R, _ = resid(B, tr)
    _, Rte, fey = resid(B, te)
    keep = R.std(0) > 1e-10                    # boycott column is all zero before 2019-07 -> unidentified
    models = {"linear": Lin().fit(R[:, keep], ry)}
    for d in (2, 3, 4):
        models[f"xgb_d{d}"] = make_xgb(max_depth=d).fit(R[:, keep], ry)
    for name, g in models.items():
        pred = fey + g.predict(Rte[:, keep])
        for t, p, a in zip(S.t[te], pred, Y[te]):
            rows.append({"origin": f"{2015 + o // 12}-{o % 12 + 1:02d}", "t": int(t), "model": name,
                         "pred": float(p), "actual": float(a), "err": float(p - a)})
O = pd.DataFrame(rows)
O["month"] = O.t.map(lambda t: f"{2015 + t // 12}-{t % 12 + 1:02d}")
O.to_csv(RESULTS / "xgb_oos_rolling.csv", index=False)
W = O.pivot_table(index="t", columns="model", values="err")


def dm(e1, e2, lag=3):
    """Diebold-Mariano (squared loss), Newey-West variance, HLN-style t reference. >0: model 1 worse."""
    d = (e1 ** 2 - e2 ** 2).values; n = len(d); dc = d - d.mean()
    v = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    st = d.mean() / np.sqrt(v / n)
    return float(st), float(2 * stats.t.sf(abs(st), n - 1)), n


onset = (W.index >= ym(2019, 7)) & (W.index <= ym(2019, 12))
acc = {}
for name in W.columns:
    for lab, msk in [("all", np.ones(len(W), bool)), ("excl_boycott_onset", ~onset)]:
        e = W.loc[msk, name]
        acc[f"{name}/{lab}"] = {"RMSE_%": 100 * np.sqrt((e ** 2).mean()), "MAE_%": 100 * e.abs().mean(), "n": int(len(e))}
for name in ["xgb_d2", "xgb_d3", "xgb_d4"]:
    for lab, msk in [("all", np.ones(len(W), bool)), ("excl_boycott_onset", ~onset)]:
        st, p, n = dm(W.loc[msk, name], W.loc[msk, "linear"])
        acc[f"DM {name} vs linear/{lab}"] = {"DM": st, "p": p, "n": n}
print("== rolling-origin out-of-sample (Japan, log errors) ==")
print(pd.DataFrame(acc).T.round(3).to_string())

# ---------------------------------------------------------------- 2. full-sample counterfactual effects
ALL = np.ones(len(S), bool)
B = fe_fit(ALL)
ry, R, _ = resid(B, ALL)
fx = slice(0, 11); bi = 11
lo, hi = np.percentile(R[:, fx], 1, axis=0), np.percentile(R[:, fx], 99, axis=0)
boy = (S.boycott == 1).values


def effects(g, R, boy_mask):
    h = -LN09 / 2                                              # 0.0527 each side, total ln 0.9
    Rm, Rp = R.copy(), R.copy()
    Rm[:, fx] -= h; Rp[:, fx] += h                             # Rm: yen cheaper by ~5%, Rp: dearer by ~5%
    ok = np.all((Rm[:, fx] >= lo) & (Rp[:, fx] <= hi), axis=1)
    e_fx = float(np.mean(g.predict(Rm[ok]) - g.predict(Rp[ok])))          # log change for a 10% cheaper yen
    Rb = R[boy_mask].copy(); R0 = Rb.copy(); R0[:, bi] -= 1.0
    e_b = float(np.mean(g.predict(Rb) - g.predict(R0)))                   # log effect of boycott (negative)
    Rz = Rb.copy(); Rz[:, bi] = np.median(R[~boy_mask, bi])               # sensitivity: set to typical non-boycott
    e_bz = float(np.mean(g.predict(Rb) - g.predict(Rz)))
    return {"fx10_log": e_fx, "fx10_pct": 100 * (np.exp(e_fx) - 1), "n_fx_used": int(ok.sum()),
            "boy_log": e_b, "boy_pct": 100 * (np.exp(e_b) - 1), "boy_log_alt": e_bz,
            "ratio": -e_b / e_fx, "ratio_alt": -e_bz / e_fx}


fits = {"linear": Lin().fit(R, ry)}
for d in (2, 3, 4):
    fits[f"xgb_d{d}"] = make_xgb(max_depth=d).fit(R, ry)
eff = {k: effects(g, R, boy) for k, g in fits.items()}
print("\n== counterfactual effects (full sample) ==")
print(pd.DataFrame(eff).T.round(3).to_string())

# block bootstrap over time (12-month blocks), FE projection held fixed
rng = np.random.default_rng(2026)
ts = np.sort(S.t.unique()); nb = int(np.ceil(len(ts) / 12))
boot = {k: [] for k in ["linear", "xgb_d2"]}
skipped = 0
for b in range(300):
    starts = rng.integers(0, len(ts) - 11, nb)
    tt = np.concatenate([ts[s:s + 12] for s in starts])[:len(ts)]
    idx = np.concatenate([np.where(S.t.values == t)[0] for t in tt])
    if boy[idx].sum() == 0:
        skipped += 1; continue
    for k in boot:
        g = Lin().fit(R[idx], ry[idx]) if k == "linear" else make_xgb(random_state=b).fit(R[idx], ry[idx])
        boot[k].append(effects(g, R, boy)["ratio"])
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 50)), float(np.percentile(v, 97.5)), len(v)]
      for k, v in boot.items()}
print("\nbootstrap ratio [2.5%, median, 97.5%, B] :", {k: np.round(v, 2).tolist() for k, v in ci.items()},
      "| skipped (no boycott rows):", skipped)

# ---------------------------------------------------------------- 3. SHAP + partial dependence
g = fits["xgb_d2"]
contrib = g.get_booster().predict(xgb.DMatrix(R), pred_contribs=True)
shap_fx = contrib[:, fx].sum(axis=1)
rbar = R[:, fx].mean(axis=1)
grid = np.linspace(-0.10, 0.10, 21)
pd_xgb, pd_lin = [], []
for s in grid:
    Rs = R.copy(); Rs[:, fx] += s
    pd_xgb.append(float(np.mean(g.predict(Rs) - g.predict(R))))
    pd_lin.append(float(np.mean(fits["linear"].predict(Rs) - fits["linear"].predict(R))))
pd.DataFrame({"shift": grid, "pd_xgb": pd_xgb, "pd_linear": pd_lin}).to_csv(RESULTS / "xgb_pd_fx.csv", index=False)
theta_lin = float(fits["linear"].b[fx].sum())

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # any Korean font works
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT); family = "Noto Sans CJK JP"
else:
    family = next((f.name for f in font_manager.fontManager.ttflist
                   if f.name in ("Malgun Gothic", "AppleGothic", "NanumGothic")), "sans-serif")
plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
pal = {"JP": "#1f3b57", "VN": "#9fb4c7", "TH": "#b7c9a8", "PH": "#d9c39a", "TW": "#c9a6a6"}
for c, gg in pd.DataFrame({"r": rbar, "s": shap_fx, "c": S.country}).groupby("c"):
    ax[0].scatter(gg.r, gg.s, s=12 if c != "JP" else 18, color=pal[c], alpha=.8, label={"JP": "일본", "VN": "베트남", "TH": "태국", "PH": "필리핀", "TW": "대만"}[c])
xx = np.linspace(rbar.min(), rbar.max(), 10)
ax[0].plot(xx, theta_lin * xx, color="#c0392b", lw=1.5, ls="--", label=f"선형 모형 (θ = {theta_lin:.2f})")
ax[0].set_xlabel("환율 잔차 (시차 0~10 평균, 고정효과 제거 후 ln)"); ax[0].set_ylabel("환율 변수 SHAP 합 (ln 방문객)")
ax[0].set_title("XGBoost가 배운 환율 효과 vs 선형 가정", fontsize=11, loc="left")
ax[0].legend(frameon=False, fontsize=8)
pct = lambda v: 100 * (np.exp(np.array(v)) - 1)
ax[1].plot(100 * (np.exp(grid) - 1), pct(pd_lin), color="#c0392b", ls="--", lw=1.5, label="선형 모형")
ax[1].plot(100 * (np.exp(grid) - 1), pct(pd_xgb), color="#1f3b57", lw=2, label="XGBoost")
ax[1].axvspan(-10, 10, color="none")
ax[1].axhline(0, color="#999", lw=.6); ax[1].axvline(0, color="#999", lw=.6)
ax[1].set_xlabel("모든 시차의 환율을 함께 바꾼 폭 (%, 음수 = 엔화가 싸짐)"); ax[1].set_ylabel("방문객 평균 변화 (%)")
ax[1].set_title("부분의존: 환율 폭에 따른 반응", fontsize=11, loc="left"); ax[1].legend(frameon=False, fontsize=8.5)
for a in ax:
    for s in ["top", "right"]: a.spines[s].set_visible(False)
fig.text(0.01, 0.005, "고정효과(국가×달력월, 연-월, 재개 후 경과 개월)를 먼저 제거한 잔차에 적합. 관측 범위 밖(잔차 ±10% 이상)은 트리 모형이 외삽하지 못해 평평해짐.", fontsize=7.8, color="#666")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIGURES / "fig7_xgb_linearity.png", dpi=160)

# ---------------------------------------------------------------- 4. post-hoc diagnostic (NOT pre-registered)
# Why the FWL counterfactual understates the boycott effect for trees: residualising the 0/1 dummy on the
# fixed effects leaves 0.68 on boycott months (0.32 is absorbed by Japan x month and year-month effects).
# The exact counterfactual (dummy 1 -> 0) moves the residual to -0.32, outside anything the trees saw,
# so a tree can only return its learned step (~0.68 of the contrast). Linear models extrapolate that line.
# Imputation estimator (standard for a binary event with a flexible outcome model): estimate the fixed
# effects and g on non-boycott months only, impute the no-boycott outcome for boycott months,
# boycott effect = mean(actual - imputed). FX effect from the same g.
nb = ~boy
Bn = np.linalg.lstsq(F[nb], np.column_stack([Y[nb], XV[nb, :11]]), rcond=None)[0]
Pn = F @ Bn; ryn = Y - Pn[:, 0]; Rn = XV[:, :11] - Pn[:, 1:]
lo_n, hi_n = np.percentile(Rn[nb], 1, axis=0), np.percentile(Rn[nb], 99, axis=0)


def fx10_imp(g, Rf):
    h = -LN09 / 2; Rm, Rp = Rf - h, Rf + h
    ok = np.all((Rm >= lo_n) & (Rp <= hi_n), axis=1)
    return float(np.mean(g.predict(Rm[ok]) - g.predict(Rp[ok])))


imp = {}
for name, mk in [("linear", lambda: Lin()), ("xgb_d2", lambda: make_xgb()), ("xgb_d3", lambda: make_xgb(max_depth=3)),
                 ("xgb_d4", lambda: make_xgb(max_depth=4))]:
    g4 = mk().fit(Rn[nb], ryn[nb]); gam = float(np.mean(ryn[boy] - g4.predict(Rn[boy]))); f4 = fx10_imp(g4, Rn[nb])
    imp[name] = {"fx10_pct": 100 * (np.exp(f4) - 1), "boy_log": gam, "boy_pct": 100 * (np.exp(gam) - 1), "ratio": -gam / f4}
rng = np.random.default_rng(7); tsn = np.sort(S.t[nb].unique()); rb = {"linear": [], "xgb_d2": []}
for b in range(300):
    st = rng.integers(0, len(tsn) - 11, int(np.ceil(len(tsn) / 12)))
    tt = np.concatenate([tsn[s:s + 12] for s in st])[:len(tsn)]
    idx = np.concatenate([np.where((S.t.values == t) & nb)[0] for t in tt])
    for k in rb:
        g4 = Lin().fit(Rn[idx], ryn[idx]) if k == "linear" else make_xgb(random_state=b).fit(Rn[idx], ryn[idx])
        rb[k].append(-np.mean(ryn[boy] - g4.predict(Rn[boy])) / fx10_imp(g4, Rn[idx]))
imp_ci = {k: np.percentile(v, [2.5, 50, 97.5]).tolist() for k, v in rb.items()}
gx = fits["xgb_d2"]; cb = gx.get_booster().predict(xgb.DMatrix(R[boy]), pred_contribs=True)
diag = {"resid_dummy_on_boycott_rows": float(R[boy, bi].mean()),
        "xgb_shap_boycott_on_boycott_rows": float(cb[:, bi].mean()),
        "linear_boycott_part_on_boycott_rows": float((R[boy, bi] * fits["linear"].b[bi]).mean())}
print("\n== post-hoc: imputation estimator ==")
print(pd.DataFrame(imp).T.round(3).to_string()); print("ratio CI", {k: np.round(v, 2).tolist() for k, v in imp_ci.items()})
print(diag)

json.dump({"oos": acc, "effects": eff, "bootstrap_ratio": ci, "boot_skipped": skipped,
           "posthoc_imputation": imp, "posthoc_imputation_ci": imp_ci, "posthoc_diag": diag,
           "theta_linear_fwl": theta_lin, "gamma_linear_fwl": float(fits["linear"].b[bi]),
           "origins": [f"{2015 + o // 12}-{o % 12 + 1:02d}" for o in origins], "params": PARAMS},
          open(RESULTS / "xgb_robustness.json", "w"), ensure_ascii=False, indent=1, default=float)
print("theta_lin", round(theta_lin, 3), "gamma_lin", round(float(fits["linear"].b[bi]), 3))
