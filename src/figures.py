"""Step 6. All report figures (figures/fig1..fig6.png). Run after the analysis scripts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import RESULTS, FIGURES, NAME_KO
from panel_models import load_full, pct10

# ------------------------------------------------------------------ style
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # any Korean font works; change if needed
if Path(FONT).exists():
    font_manager.fontManager.addfont(FONT)
    family = "Noto Sans CJK JP"
else:
    family = next((f.name for f in font_manager.fontManager.ttflist
                   if f.name in ("Malgun Gothic", "AppleGothic", "NanumGothic")), "sans-serif")
plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10.5,
                     "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e0", "#a3a29c"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SRC = "자료: 한국관광 데이터랩 국민 해외관광객 주요 목적지별 통계(2025.12), JNTO"


def style(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"] + (["left"] if grid == "x" else []):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def titles(fig, title, sub, note=None, y=0.97, ys=0.915):
    fig.text(0.01, y, title, fontsize=13, weight="bold", color=INK, va="top")
    fig.text(0.01, ys, sub, fontsize=9.5, color=INK2, va="top")
    if note:
        fig.text(0.01, 0.01, note, fontsize=8.5, color=INK2)


P = load_full()
R = json.load(open(RESULTS / "main_models.json"))
MV = json.load(open(RESULTS / "money_vs_mind.json"))
YT = json.load(open(RESULTS / "yen_tiers.json"))
SS = pd.read_csv(RESULTS / "share_shift_2019.csv")

# ------------------------------------------------------------------ 1. event-time overlay
fig, ax = plt.subplots(figsize=(8.6, 4.9))
E = P[(P.event.notna()) & (P.event <= 44) & (P.year <= 2025)].sort_values(["country", "event"]).copy()
E["idx_s"] = E.groupby("country").idx_2018.transform(lambda s: s.rolling(3, center=True, min_periods=2).mean())
for c in ["VN", "TH", "PH", "TW"]:
    d = E[E.country == c]
    ax.plot(d.event, d.idx_s, color=MUTED, lw=1.6, zorder=2)
    ax.text(d.event.iloc[-1] + 0.6, d.idx_s.iloc[-1], NAME_KO[c], color=INK2, va="center", fontsize=9.5)
d = E[E.country == "JP"]
ax.plot(d.event, d.idx_s, color=BLUE, lw=2.4, zorder=3)
ax.text(d.event.iloc[-1] + 0.6, d.idx_s.iloc[-1], "일본", color=INK, va="center", fontsize=10.5, weight="bold")
ax.axhline(100, color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
ax.text(49.5, 103.5, "2018년 같은 달 = 100", color=INK2, fontsize=9, ha="right")
ax.set_xlim(0, 50); ax.set_ylim(0, 165)
ax.set_xlabel("입국 재개 후 경과 개월"); ax.set_ylabel("한국인 방문자 수 (2018년 같은 달 = 100)")
style(ax)
titles(fig, "같은 회복 단계에서 일본은 첫 1년 반 동안 가장 빨리 회복했다",
       "재개 시점에 맞춰 정렬한 나라별 회복 곡선, 3개월 이동평균. 일본·대만 재개 2022년 10월, 동남아 3국 2022년 4월", SRC)
fig.subplots_adjust(top=0.83, bottom=0.14, left=0.09, right=0.97)
fig.savefig(FIGURES / "fig1_recovery_by_event_time.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 2. coefficient comparison
rows = [("모형 1: 일본 단독 (통념)", R["model1"]["theta"], R["model1"]["se"], "main1"),
        ("모형 2: 달력 월 고정효과", R["model2_calendar"]["theta"], R["model2_calendar"]["se"], "alt"),
        ("모형 2: 재개 후 경과 고정효과 (본 분석)", R["model2_event"]["theta"], R["model2_event"]["se"], "main2")]
rlab = {"drop_VN": "베트남 제외", "drop_TH": "태국 제외", "drop_PH": "필리핀 제외", "drop_TW": "대만 제외",
        "alt_reopen": "재개 시점 다른 기준", "drop_transition_calendarFE": "회복 전환기 제외",
        "country_trends": "나라별 추세 추가", "drop_first3": "재개 직후 3개월 제외",
        "drop_TH_PH_2025": "태국·필리핀 2025년 제외", "drop_JP_2025H2": "일본 2025년 하반기 제외",
        "yoy_panel": "전년동월비 모형"}
for k, lab in rlab.items():
    rows.append(("강건성 · " + lab, R["robustness"][k]["theta"], R["robustness"][k]["se"], "rob"))
rows.append(("강건성 · 코로나 이전 기간만", R["pre_covid"]["model2_K10"]["theta"], R["pre_covid"]["model2_K10"]["se"], "rob"))
fig, ax = plt.subplots(figsize=(8.6, 6.6))
ys = np.arange(len(rows))[::-1].astype(float)
ys[:3] += 0.6
for y, (lab, th, se, kind) in zip(ys, rows):
    lo, hi = pct10(th + 1.96 * se), pct10(th - 1.96 * se)
    col = {"main1": ORANGE, "main2": BLUE, "alt": INK2, "rob": MUTED}[kind]
    ax.plot([lo, hi], [y, y], color=col, lw=2 if kind != "rob" else 1.6, solid_capstyle="round")
    ax.scatter([pct10(th)], [y], s=58 if kind != "rob" else 34, color=col, zorder=3, edgecolor="#fcfcfb", lw=1.5)
    ax.text(hi + 1.2, y, f"{pct10(th):+.0f}%", va="center", fontsize=9.5, color=INK if kind != "rob" else INK2)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
for t, (_, _, _, kind) in zip(ax.get_yticklabels(), rows):
    t.set_color(INK if kind != "rob" else INK2)
ax.axvline(0, color=INK2, lw=0.9)
ax.set_xlim(-2, 52)
ax.set_xlabel("엔화(해당 통화)가 10% 싸질 때 방문자 수 변화 (%), 점 = 추정치, 선 = 95% 신뢰구간")
style(ax, "x")
titles(fig, "공통 수요를 걷어내면 엔저 효과는 통념의 절반 이하로 줄어든다",
       "비교국과 공통인 해외여행 수요(코로나 전 여행 붐, 코로나 후 보복 여행)를 통제. 시차 0~10개월, Driscoll-Kraay 표준오차(모형 1은 Newey-West)",
       ys=0.925)
fig.subplots_adjust(top=0.88, bottom=0.09, left=0.35, right=0.97)
fig.savefig(FIGURES / "fig2_elasticity_comparison.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 3. decomposition waterfall
dec = R["decomposition"]
fx = dec["fx_theta_log"] * 100          # theta x (lag-window FX change): stable vs single betas
common = dec["common_log"] * 100
total = dec["total_log"] * 100
jp = dec["japan_specific_theta_log"] * 100
naive = dec["naive_theta_log"] * 100
steps = [("공통 요인\n(보복 여행·해외여행\n수요 전반)", common), ("엔저", fx), ("일본 고유 요인", jp)]
fig, ax = plt.subplots(figsize=(8.6, 4.6))
run = 0.0
for x, (lab, v) in enumerate(steps):
    col = BLUE if lab == "엔저" else ("#c9c8c2" if v < 0 else MUTED)
    ax.bar(x, abs(v), bottom=run if v >= 0 else run + v, color=col, width=0.62)
    ax.text(x, run + v + (1.0 if v >= 0 else -1.0), f"{v:+.1f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=10.5, color=INK)
    run += v
ax.bar(3, total, color=INK2, width=0.62)
ax.text(3, total + 1.0, f"{total:+.1f}", ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")
ax.axhline(0, color=INK2, lw=0.9)
ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in steps] + ["실제 증가\n(2018 → 2024~25)"], fontsize=9.5)
ax.set_ylabel("로그 차이 × 100 (≈ %)"); ax.set_ylim(-10, 30)
style(ax)
titles(fig, f"2018년 대비 일본 증가분(약 +{dec['total_pct']:.0f}%) 중 엔저 몫은 약 {round(100 * fx / total, -1):.0f}%",
       f"방일 한국인 수 2018년 vs 2024~25년 (같은 달끼리 비교). 같은 기간 엔화 약 {abs(dec['yen_change_pct_lagwindow']):.0f}% 하락. "
       f"통념 모형(모형 1)이라면 엔저 몫 {naive:.0f}%p, 즉 증가분 전부")
fig.subplots_adjust(top=0.84, bottom=0.2, left=0.09, right=0.97)
fig.savefig(FIGURES / "fig3_decomposition.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 4. break-even curve
th, se = MV["main"]["theta"], MV["main"]["theta_se"]
d = np.linspace(0, 90, 300); x = np.log(1 - d / 100)
fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.fill_between(d, 100 * (th - 1.96 * se) * x, 100 * (th + 1.96 * se) * x, color=BLUE, alpha=0.13, lw=0)
ax.plot(d, 100 * th * x, color=BLUE, lw=2.4)
ax.axvspan(0, 12.3, color=GRID, alpha=0.6, lw=0, zorder=0)
ax.text(1, 198, "실제 관측된\n최대 하락\n(−12%)", fontsize=8.8, color=INK2, va="top")
for lab, g, be, col in [("불매운동 평균 (2019.7~2020.2)", MV["main"]["boycott"], MV["main"]["breakeven"]["fall_pct"], ORANGE),
                        ("불매운동 정점 (2019.10~12)", MV["boycott_peak"]["gamma"], MV["boycott_peak"]["breakeven"]["fall_pct"], INK2)]:
    y = -100 * g
    ax.axhline(y, color=col, lw=1.2, ls=(0, (4, 3)))
    ax.text(14, y + 3, f"{lab} 효과 크기: {y:.0f}", ha="left", va="bottom", fontsize=9.5, color=INK)
    ax.plot([be, be], [0, y], color=col, lw=1.0, ls=":")
    ax.scatter([be], [y], s=60, color=col, zorder=4, edgecolor="#fcfcfb", lw=1.5)
    ax.text(be + 1.2, 8 if col == ORANGE else 22, f"{be:.0f}%", color=col, fontsize=10.5, weight="bold")
ax.set_xlim(0, 90); ax.set_ylim(0, 200)
ax.set_xlabel("엔화 하락 폭 (%, 2018년 대비)"); ax.set_ylabel("엔저로 늘어나는 방일 수요 (로그 차이 × 100)")
style(ax)
titles(fig, f"불매운동 하나를 상쇄하려면 엔화가 약 {round(MV['main']['breakeven']['fall_pct'], -1):.0f}% 떨어져야 한다",
       f"파란 선 = 엔저 효과(본 분석 탄력성 {th:.2f}), 띠 = 95% 신뢰구간. 점선(불매 효과 크기)과 만나는 지점이 \"돈이 마음을 이기는\" 엔화 하락 폭",
       "주: 회색 구간 밖은 관측 범위를 벗어난 외삽이므로 예측이 아니라 크기 비교를 위한 환산값이다")
fig.subplots_adjust(top=0.85, bottom=0.15, left=0.09, right=0.97)
fig.savefig(FIGURES / "fig4_breakeven.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 5. share shift 2019
o = SS[SS.v18H2 > 100000].sort_values("did")
fig, ax = plt.subplots(figsize=(8.6, 5.4))
cols = [ORANGE if dd == "일본" else (MUTED if v < 0 else BLUE) for dd, v in zip(o.dest, o.did)]
ax.barh(o.dest, o.did, color=cols, height=0.62)
for y, v in enumerate(o.did):
    ax.text(v + (0.15 if v >= 0 else -0.15), y, f"{v:+.1f}%p", va="center", ha="left" if v >= 0 else "right",
            fontsize=9.5, color=INK)
ax.axvline(0, color=INK2, lw=0.9)
ax.set_xlim(-11.5, 4.5)
ax.tick_params(axis="y", colors=INK)
ax.set_xlabel("한국인 전체 출국자 대비 비중 변화 (%p), 2019년 하반기 변화 − 2019년 상반기 변화")
style(ax, "x")
titles(fig, "일본 비중이 줄어든 같은 기간, 베트남·미국·필리핀·대만 등의 비중이 올랐다",
       "전년 같은 반기 대비 비중 변화에서 상반기(불매 전) 변화를 뺀 값. 2018년 하반기 방문자 10만 명 이상 목적지",
       "자료: 한국관광 데이터랩 국민 해외관광객 주요 목적지별 통계(2025.12). 홍콩 감소는 2019년 시위 시기와 겹친다", ys=0.92)
fig.subplots_adjust(top=0.86, bottom=0.15, left=0.14, right=0.97)
fig.savefig(FIGURES / "fig5_share_shift.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 6. yen tiers
order = [("p10_plus", "10% 이상\n비싸짐"), ("p5_10", "5~10%\n비싸짐"), ("ref_0_5", "0~5%\n비싸짐\n(기준)"),
         ("m5_0", "0~5%\n싸짐"), ("yen_lt_m5", "5% 이상\n싸짐")]
mid = {"p10_plus": 0.13, "p5_10": 0.075, "ref_0_5": 0.025, "m5_0": -0.025, "yen_lt_m5": -0.08}
b = YT["linear"]["rel_lin"][0]
fig, ax = plt.subplots(figsize=(8.6, 4.9))
xs = np.arange(len(order))
lin = [100 * (np.exp(b * (mid[k] - mid["ref_0_5"])) - 1) for k, _ in order]
ax.plot(xs, lin, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
ax.text(4.08, lin[-1], "일정 탄력성\n가정 시", color=INK2, fontsize=9, va="center")
for x, (k, lab) in zip(xs, order):
    if k == "ref_0_5":
        ax.scatter([x], [0], s=50, color=INK2, zorder=3); continue
    c, s = YT["tiers"][k]
    y, lo, hi = [100 * (np.exp(v) - 1) for v in (c, c - 1.96 * s, c + 1.96 * s)]
    ax.plot([x, x], [lo, hi], color=BLUE, lw=2, solid_capstyle="round")
    ax.scatter([x], [y], s=60, color=BLUE, zorder=3, edgecolor="#fcfcfb", lw=1.5)
    ax.text(x + 0.1, y, f"{y:+.0f}%", va="center", fontsize=10, color=INK)
ax.axhline(0, color=INK2, lw=0.9)
ax.set_xticks(xs); ax.set_xticklabels([l for _, l in order], fontsize=9.5)
ax.tick_params(axis="x", colors=INK)
ax.set_xlim(-0.5, 4.9)
ax.set_ylabel("방문자 수 차이 (%), 기준 구간 대비")
style(ax)
titles(fig, "엔저 폭별 효과는 일정 탄력성 가정과 통계적으로 구분되지 않는다",
       "이슈 없는 기간, 통화 가치의 10개월 평균을 자국 2018년 수준과 비교한 구간별 효과. 점 = 추정치, 선 = 95% 신뢰구간",
       "주: '5% 이상 싸짐' 구간은 일본에서만, 2016년 1월과 2023년 9월~2025년 8월 25개월 관측. 사실상 일본 그 시기 더미라 다른 요인과 구분되지 않는다")
fig.subplots_adjust(top=0.85, bottom=0.2, left=0.1, right=0.93)
fig.savefig(FIGURES / "fig6_yen_tiers.png", dpi=200); plt.close(fig)

print("wrote", sorted(p.name for p in FIGURES.glob("fig*.png")))
