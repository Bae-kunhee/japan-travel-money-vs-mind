"""fig13: 일본 방문 한국인(막대, 만 명)과 엔화(선, 원/100엔)를 한 그래프에 (x = 시간)."""
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from config import PROCESSED, FIGURES, ym

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
INK2, GRID, BAR = "#52514e", "#e6e5e0", "#bdbcb6"
BLUE, ORANGE = "#2a78d6", "#eb6834"

P = pd.read_csv(PROCESSED / "panel.csv")
J = P[(P.country == "JP") & (P.t >= ym(2016, 1)) & (P.t <= ym(2025, 12))].sort_values("t")
x = J.t.values

fig, ax = plt.subplots(figsize=(9.4, 4.6))
ax.axvspan(ym(2019, 7) - 0.5, ym(2020, 2) + 0.5, color=ORANGE, alpha=0.12, lw=0, zorder=0)
ax.bar(x, J.visitors / 1e4, width=0.85, color=BAR, zorder=2)
ax.set_ylabel("방일 한국인 (만 명)")
ax.set_ylim(0, 120)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)

ax2 = ax.twinx()
ax2.plot(x, J.fx, color=BLUE, lw=2.2, zorder=4)
ax2.set_ylabel("엔화 (원/100엔)", color=BLUE)
ax2.tick_params(axis="y", colors=BLUE, length=0)
ax2.set_ylim(800, 1200)
for s in ["top", "left"]:
    ax2.spines[s].set_visible(False)
ax2.spines["right"].set_color("#c9c8c2")

ticks = [t for t in x if (t - ym(2016, 1)) % 12 == 0]
ax.set_xticks(ticks); ax.set_xticklabels([2016 + i for i in range(len(ticks))])
ax.set_xlim(x[0] - 1, x[-1] + 1)
ax.text((ym(2019, 7) + ym(2020, 2)) / 2, 118, "불매 국면", color=ORANGE, fontsize=9.5, ha="center", va="top")
ax.text(ym(2020, 9), 118, "막대: 방일 한국인\n선: 엔화 (위로 갈수록 엔고)", color=INK2, fontsize=9, va="top", linespacing=1.5)
fig.subplots_adjust(top=0.97, bottom=0.09, left=0.09, right=0.91)
fig.savefig(FIGURES / "fig13_fx_visitors.png", dpi=200)
print(J.visitors.max()/1e4, J.fx.min(), J.fx.max())
