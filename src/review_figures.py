"""Figures added after the review: event study (fig10), Japan vs comparators trend (fig11),
both effects on one % axis (fig12). Run after review_checks.py.
Colour roles used in every figure: FX/yen = blue, boycott = orange, comparators/common = grey."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import RESULTS, FIGURES, ym

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT); family = "Noto Sans CJK JP"
else:
    family = next((f.name for f in font_manager.fontManager.ttflist
                   if f.name in ("Malgun Gothic", "AppleGothic", "NanumGothic")), "sans-serif")
plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10.5,
                     "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29c"
BLUE, ORANGE = "#2a78d6", "#eb6834"
R = json.load(open(RESULTS / "review_checks.json"))


def style(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"] + (["left"] if grid == "x" else []):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def lab(t):
    return f"{2015 + t // 12}.{t % 12 + 1}"


# ------------------------------------------------------------------ fig10 event study
E = pd.DataFrame(R["event_study"]["coefs"])
E["pct"] = 100 * (np.exp(E.coef) - 1)
E["lo"] = 100 * (np.exp(E.coef - 1.96 * E.se) - 1)
E["hi"] = 100 * (np.exp(E.coef + 1.96 * E.se) - 1)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = np.arange(len(E))
post = (E.t >= ym(2019, 7)).values
for xi, (_, r), p in zip(x, E.iterrows(), post):
    col = ORANGE if p else MUTED
    ax.plot([xi, xi], [r.lo, r.hi], color=col, lw=1.8, solid_capstyle="round")
    ax.scatter([xi], [r.pct], s=34, color=col, zorder=3, edgecolor="#fcfcfb", lw=1.2)
ax.axhline(0, color=INK2, lw=0.9)
ax.axvline(x[post][0] - 0.5, color=INK2, lw=0.9, ls=(0, (3, 3)))
pm = 100 * (np.exp(R["event_study"]["pre_mean"]) - 1)
ax.plot([x[0] - 0.3, x[~post][-1] + 0.3], [pm, pm], color=INK2, lw=1, ls=":")
ax.text(x[0] - 0.3, -86, f"점선: 불매 전 12개월 평균 ({pm:.0f}%)", fontsize=9, color=INK2)
ax.text(x[post][0] - 0.7, 8, "2019.7 수출규제·불매 시작", fontsize=9, color=INK2, ha="right")
ax.set_xticks(x); ax.set_xticklabels([lab(t) for t in E.t], rotation=45, fontsize=8.5)
ax.set_ylim(-90, 30)
ax.set_ylabel("비교국 대비 일본 (%)")
style(ax)
fig.subplots_adjust(top=0.97, bottom=0.18, left=0.1, right=0.98)
fig.savefig(FIGURES / "fig10_event_study.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ fig11 Japan vs comparators
T = pd.read_csv(RESULTS / "trend_japan_vs_comparators.csv")
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.axvspan(ym(2019, 7) - 0.5, ym(2020, 2) + 0.5, color=ORANGE, alpha=0.08, lw=0)
ax.plot(T.t, T.comparators, color=MUTED, lw=2)
ax.plot(T.t, T.japan, color=INK, lw=2.4)
ax.text(T.t.iloc[-1] + 0.4, T.comparators.iloc[-1], "비교국 4곳 평균", color=INK2, va="center", fontsize=9.5)
ax.text(T.t.iloc[-1] + 0.4, T.japan.iloc[-1], "일본", color=INK, va="center", fontsize=10.5, weight="bold")
ax.axhline(100, color=INK2, lw=0.9, ls=(0, (3, 3)))
ax.text(ym(2019, 7), 128, "불매 국면", color=ORANGE, fontsize=9.5)
ticks = [t for t in T.t if (t % 12) in (0, 6)]
ax.set_xticks(ticks); ax.set_xticklabels([lab(t) for t in ticks])
ax.set_xlim(T.t.iloc[0] - 0.5, T.t.iloc[-1] + 6)
ax.set_ylim(0, 135)
ax.set_ylabel("계절조정 지수 (불매 전 1년 = 100)")
style(ax)
fig.subplots_adjust(top=0.97, bottom=0.1, left=0.1, right=0.98)
fig.savefig(FIGURES / "fig11_trend_japan_vs_comparators.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ fig12 both effects on one % axis
Y = R["yen10_intervals"]; B = R["boycott_intervals"]
M = json.load(open(RESULTS / "money_vs_mind.json"))["main"]
rows = [("10% 엔저", Y["point"], Y["dk"], Y["cr1_t4"], BLUE),
        ("2019년 불매 국면", M["boycott_pct"], B["pct_dk"], None, ORANGE)]
fig, ax = plt.subplots(figsize=(8.6, 2.9))
for y, (name, pt, dk, cr, col) in zip([1, 0], rows):
    if cr is not None:
        ax.plot(cr, [y, y], color=col, lw=1.2, ls=(0, (2, 2)))
    ax.plot(dk, [y, y], color=col, lw=3.2, solid_capstyle="round")
    ax.scatter([pt], [y], s=70, color=col, zorder=3, edgecolor="#fcfcfb", lw=1.5)
    ax.text(pt, y + 0.22, f"{pt:+.0f}%", ha="center", fontsize=10.5, color=INK, weight="bold")
ax.axvline(0, color=INK2, lw=0.9)
ax.text(-84, -0.5, "굵은 선: Driscoll-Kraay 95%   점선: 국가 군집 95% (t(4))", ha="left", fontsize=8.5, color=INK2)
ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
ax.set_ylim(-0.6, 1.6); ax.set_xlim(-85, 45)
ax.set_xlabel("방일 한국인 변화 (%)")
style(ax, "x")
fig.subplots_adjust(top=0.97, bottom=0.2, left=0.17, right=0.97)
fig.savefig(FIGURES / "fig12_effects.png", dpi=200); plt.close(fig)
print("wrote fig10, fig11, fig12")
