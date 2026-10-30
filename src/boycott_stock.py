"""Boycott intensity and persistence: a decaying stock of Japan-boycott search volume.

Data: data/raw/naver_datalab_boycott_search.csv (Naver DataLab, monthly, one request -> common scale).
S_t   = '노재팬' + '일본 불매' search index minus its 2021-01..2026-09 median (floored at 0), so that the
        low background searching after 2020 does not pile up into a permanent 'boycott'.
        '불매운동' is not used: it also picks up non-Japan boycotts (e.g. 2016-05).
K_t   = S_t + delta * K_{t-1}  (adstock), K = 0 before 2016-01. Half-life = ln 0.5 / ln delta.
Model = Model 2 with the 0/1 boycott dummy replaced by JP x K_{t-L}(delta) / mean K over the boycott months
        (so the coefficient is the effect at the average boycott-period intensity), L in {0, 1}.
Money vs mind: boycott effect at average intensity / 10% yen effect from the same model.
delta and L are tuned on the single 2019 episode -> reported as a sensitivity curve, not one number.
Outputs: results/boycott_stock_grid.csv, results/boycott_stock.json, figures/fig8_boycott_stock.png
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from config import ROOT, RESULTS, FIGURES, ym
from panel_models import open_sample, load_panel, load_full, m2_fit, pct10

LN09 = np.log(0.9)
BOY = list(range(ym(2019, 7), ym(2020, 2) + 1))
DELTAS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95]

N = pd.read_csv(ROOT / "data" / "raw" / "naver_datalab_boycott_search.csv")
N["t"] = (N.month.str[:4].astype(int) - 2015) * 12 + N.month.str[5:7].astype(int) - 1
N = N.set_index("t").sort_index()
I_raw = N.no_japan + N.japan_boycott
I_exc = (I_raw - I_raw.loc[ym(2021, 1):].median()).clip(lower=0)
S = open_sample(load_panel())


def stock(I, d):
    k, out = 0.0, {}
    for t in I.index:
        k = I[t] + d * k
        out[t] = k
    return pd.Series(out)


def fit_stock(I, d, L):
    K = stock(I, d).shift(L)
    Kn = K / K.loc[BOY].mean()                                    # 1 = average boycott-period intensity
    x = pd.Series(np.where(S.country == "JP", S.t.map(Kn).fillna(0), 0.0), index=S.index, name="kstock")
    D = S.copy(); D["boycott"] = 0.0                               # dummy dropped (all-zero column is pruned)
    o, fit, X, V = m2_fit(D, 10, extra=x.to_frame())
    i = list(X.columns).index("kstock")
    jp = ((S.country == "JP") & S.t.isin(BOY)).values
    return o, fit, Kn, float(fit.params["kstock"]), float(np.sqrt(V[i, i])), float(np.sqrt(np.mean(fit.resid[jp] ** 2)))


rows = []
for var, I in [("excess", I_exc), ("raw", I_raw)]:
    for L in (1, 0):
        for d in DELTAS:
            o, fit, Kn, g, se, rmse_b = fit_stock(I, d, L)
            rows.append({"search": var, "lag": L, "delta": d,
                         "half_life_months": float(np.log(.5) / np.log(d)) if 0 < d < 1 else 0.0,
                         "theta": o["theta"], "yen10_pct": pct10(o["theta"]),
                         "gamma_avg": g, "gamma_se": se, "boycott_pct_avg": 100 * (np.exp(g) - 1),
                         "ratio_boycott_to_yen10": -g / (o["theta"] * LN09),
                         "aic": o["aic"], "rmse_boycott_months": rmse_b})
R = pd.DataFrame(rows)
R.to_csv(RESULTS / "boycott_stock_grid.csv", index=False)

o, fit, X, V = m2_fit(S, 10)
jp = ((S.country == "JP") & S.t.isin(BOY)).values
dummy = {"theta": o["theta"], "gamma": o["boycott"][0], "yen10_pct": pct10(o["theta"]),
         "boycott_pct": 100 * (np.exp(o["boycott"][0]) - 1),
         "ratio_boycott_to_yen10": -o["boycott"][0] / (o["theta"] * LN09), "aic": o["aic"],
         "rmse_boycott_months": float(np.sqrt(np.mean(fit.resid[jp] ** 2)))}
main = R[(R.search == "excess") & (R.lag == 1)]
best = main.loc[main.aic.idxmin()]
band = main[(main.delta >= 0.6) & (main.delta <= 0.9)].ratio_boycott_to_yen10
s19 = I_raw.loc[[ym(2019, m) for m in (7, 8, 9, 10)]].round(1).tolist()
out = {"dummy_reference": dummy, "best_excess_lag1": best.to_dict(),
       "best_by_spec": R.loc[R.groupby(["search", "lag"]).aic.idxmin()].to_dict("records"),
       "ratio_range_delta_0.6_0.9": [float(band.min()), float(band.max())],
       "search_2019_jul_oct": s19,
       "note": "delta and lag are tuned on one episode; add 4 to AIC for the two tuned values"}
json.dump(out, open(RESULTS / "boycott_stock.json", "w"), ensure_ascii=False, indent=1, default=float)
print(f"dummy: AIC {dummy['aic']:.1f}, ratio {dummy['ratio_boycott_to_yen10']:.1f}x, boycott-month RMSE {dummy['rmse_boycott_months']:.2f}")
print(f"stock (excess, lag 1) best delta {best.delta} (half-life {best.half_life_months:.1f} m): AIC {best.aic:.1f}, "
      f"boycott {best.boycott_pct_avg:.0f}%, yen10 {best.yen10_pct:.1f}%, ratio {best.ratio_boycott_to_yen10:.1f}x, "
      f"boycott-month RMSE {best.rmse_boycott_months:.2f}")
print(f"ratio over delta 0.6-0.9: {band.min():.1f} - {band.max():.1f}x | search Jul-Oct 2019: {s19}")

# ------------------------------------------------------------------ figure 8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # any Korean font works
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT); family = "Noto Sans CJK JP"
else:
    family = next((f.name for f in font_manager.fontManager.ttflist
                   if f.name in ("Malgun Gothic", "AppleGothic", "NanumGothic")), "sans-serif")
plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10})

_, _, Kn, g_best, _, _ = fit_stock(I_exc, float(best.delta), 1)
J = load_full().query("country == 'JP'").set_index("t")
ts = list(range(ym(2019, 4), ym(2020, 2) + 1)); lab = [f"{2015 + t // 12}.{t % 12 + 1}" for t in ts]
INK, INK2, GRID, MUTED, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29c", "#2a78d6", "#eb6834"
plt.rcParams.update({"axes.edgecolor": "#c9c8c2", "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def clean(a):
    a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    for s_ in ["top", "right"]:
        a.spines[s_].set_visible(False)
    a.tick_params(length=0)


# fig8a: one time axis, two panels (search index on top, visitors and model responses below)
x = np.arange(len(ts))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
a1.bar(x, 100 * I_exc.loc[ts] / I_exc.loc[ts].max(), color="#e3c9a6", width=.65)
a1.set_ylabel("불매 검색량\n(정점 = 100)"); a1.set_ylim(0, 110); clean(a1)
a2.plot(x, J.loc[ts, "idx_2018"], color=INK, lw=2.4, marker="o", ms=4)
a2.plot(x, 100 * np.exp(g_best * Kn.loc[ts].fillna(0)), color=ORANGE, lw=2, ls="--")
a2.plot(x, [100 * np.exp(dummy["gamma"]) if t in BOY else 100 for t in ts], color=MUTED, lw=1.6, ls=":")
a2.text(10.3, J.loc[ts[-1], "idx_2018"], "실제 방일 한국인", color=INK, fontsize=9.5, va="center")
a2.text(10.3, 100 * np.exp(g_best * Kn.loc[ts[-1]]), "스톡 모형", color=ORANGE, fontsize=9.5, va="center")
a2.text(10.3, 100 * np.exp(dummy["gamma"]) - 4, "0/1 더미", color=INK2, fontsize=9.5, va="center")
a2.annotate("코로나 영향 시작", xy=(10, J.loc[ts[-1], "idx_2018"]), xytext=(8.2, 8), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a2.set_ylabel("방문자 지수\n(2018년 같은 달 = 100)"); a2.set_ylim(0, 115); clean(a2)
a2.set_xticks(x); a2.set_xticklabels(lab, fontsize=9)
a2.set_xlim(-0.6, 12.2)
fig.tight_layout(); fig.savefig(FIGURES / "fig8a_boycott_stock_time.png", dpi=200); plt.close(fig)

# fig8b: one delta axis, two panels (AIC on top, ratio below)
fig, (b1, b2) = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True)
b1.plot(main.delta, main.aic, color=INK2, lw=2, marker="o", ms=4)
b1.axhline(dummy["aic"], color=MUTED, ls=":", lw=1.2)
b1.text(0.0, dummy["aic"] + 4, f"0/1 더미 AIC {dummy['aic']:.0f}", fontsize=9, color=INK2)
b1.set_ylabel("AIC (낮을수록 적합)"); clean(b1)
b2.plot(main.delta, main.ratio_boycott_to_yen10, color=INK, lw=2, marker="o", ms=4)
b2.axhline(dummy["ratio_boycott_to_yen10"], color=MUTED, ls=":", lw=1.2)
b2.text(0.0, dummy["ratio_boycott_to_yen10"] + 0.4, f"0/1 더미 {dummy['ratio_boycott_to_yen10']:.1f}배", fontsize=9, color=INK2)
b2.set_ylabel("불매 ÷ 10% 엔저 (배)"); b2.set_ylim(0, 13); clean(b2)
for b_ in (b1, b2):
    b_.axvline(best.delta, color=ORANGE, lw=1)
b1.text(best.delta - 0.015, b1.get_ylim()[1] * 0.92 if b1.get_ylim()[1] > 0 else 60,
        f"AIC 최소 δ = {best.delta}", fontsize=9, color=ORANGE, ha="right", va="top")
b2.set_xlabel("월 잔존율 δ")
fig.tight_layout(); fig.savefig(FIGURES / "fig8b_boycott_stock_delta.png", dpi=200); plt.close(fig)
print("wrote figures/fig8a_boycott_stock_time.png, fig8b_boycott_stock_delta.png")
