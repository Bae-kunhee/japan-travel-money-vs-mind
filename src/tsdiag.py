"""Standard time-series diagnostic checklist ([1]-[6] before fitting, [7]-[8] on residuals).
Every check prints `label | value | verdict | note` and is also collected in ROWS for tables."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

warnings.filterwarnings("ignore")
W = 78
_MARK = {"OK": "OK  ", "WARN": "WARN", "FAIL": "FAIL", "": "    "}
ROWS = []          # (series, section, label, value, verdict, note)
_CUR = {"name": "", "sec": ""}


def _head(no, title):
    _CUR["sec"] = f"[{no}] {title}"
    print("\n" + "=" * W); print(f"[{no}] {title}  —  {_CUR['name']}"); print("=" * W)


def _row(label, value, verdict="", note=""):
    ROWS.append((_CUR["name"], _CUR["sec"], label, value, verdict, note))
    if isinstance(value, (float, np.floating)):
        value = f"{value:,.4f}" if abs(value) < 1e5 else f"{value:,.3e}"
    print(f"  {label:<26} {str(value):>14}  {_MARK[verdict]}  {note}")


def _gph(x):
    x = np.asarray(x) - np.mean(x)
    n = len(x); m = int(np.floor(np.sqrt(n)))
    I = np.abs(np.fft.fft(x)) ** 2 / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    X = -2 * np.log(2 * np.sin(lam / 2))
    y = np.log(I[1:m + 1])
    b, a, r, p, se = stats.linregress(X, y)
    return b, se, m


def diagnose(series, name, lags=24, freq="MS"):
    _CUR["name"] = name
    s = pd.Series(series).copy()
    # [1] integrity
    _head(1, "DATA INTEGRITY")
    n_nan = int(s.isna().sum()); n_inf = int(np.isinf(s.astype(float)).sum())
    _row("missing", n_nan, "OK" if n_nan == 0 else "WARN", "결측 없음" if n_nan == 0 else "결측 제거 후 진행")
    _row("infinite", n_inf, "OK" if n_inf == 0 else "FAIL", "")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    _row("non-positive", int((s <= 0).sum()), "OK" if (s > 0).all() else "", "로그값 시리즈면 해당 없음")
    if isinstance(s.index, pd.DatetimeIndex):
        _row("period", f"{s.index[0]:%Y-%m}~{s.index[-1]:%Y-%m}", "", "")
        exp = pd.date_range(s.index[0], s.index[-1], freq=freq)
        gaps = len(exp) - len(s)
        _row("gaps (missing months)", gaps, "OK" if gaps == 0 else "WARN", "등간격" if gaps == 0 else "봉쇄 등으로 빠진 달")
        dup = int(s.index.duplicated().sum())
        _row("duplicate index", dup, "OK" if dup == 0 else "FAIL", "")
    # [2] descriptive
    _head(2, "DESCRIPTIVE")
    sk, ku = float(stats.skew(s)), float(stats.kurtosis(s))
    for lab, v in [("n", len(s)), ("mean", float(s.mean())), ("std", float(s.std())),
                   ("min", float(s.min())), ("p01", float(s.quantile(.01))), ("q1", float(s.quantile(.25))),
                   ("median", float(s.median())), ("q3", float(s.quantile(.75))), ("p99", float(s.quantile(.99))),
                   ("max", float(s.max()))]:
        _row(lab, v)
    _row("skewness", sk, "WARN" if abs(sk) > 1 else "OK", "")
    _row("excess kurtosis", ku, "WARN" if abs(ku) > 3 else "OK", "")
    # [3] distribution
    _head(3, "DISTRIBUTION & TRANSFORMATION")
    jb = stats.jarque_bera(s)
    _row("Jarque-Bera p", float(jb.pvalue), "OK" if jb.pvalue > .05 else "WARN",
         "정규성 기각 못 함" if jb.pvalue > .05 else "정규성 기각")
    if (s > 0).all():
        ls = np.log(s)
        lsk, lku = float(stats.skew(ls)), float(stats.kurtosis(ls))
        imp = 1 - (abs(lsk) + abs(lku)) / max(abs(sk) + abs(ku), 1e-12)
        _row("log skew / kurt", f"{lsk:.2f} / {lku:.2f}", "", "")
        _row("log improvement", float(imp), "", "log 변환 강력 권장" if imp >= .3 else "log 변환 이득 작음")
    # [4] stationarity
    _head(4, "STATIONARITY")
    adf_p = float(adfuller(s, regression="ct", autolag="AIC")[1])
    kpss_p = float(kpss(s, regression="ct", nlags="auto")[1])
    _row("ADF p (H0 단위근, ct)", adf_p, "OK" if adf_p < .05 else "WARN", "")
    _row("KPSS p (H0 정상, ct)", kpss_p, "OK" if kpss_p > .05 else "WARN", "KPSS p는 0.01~0.1로 잘림")
    stat = adf_p < .05 and kpss_p > .05
    unit = adf_p >= .05 and kpss_p <= .05
    _row("verdict", "추세정상" if stat else ("단위근" if unit else "불일치"),
         "OK" if stat else "WARN", "" if stat else "차분 또는 패널 고정효과로 처리")
    d = np.diff(s)
    _row("ADF p (1차 차분)", float(adfuller(d, regression="c", autolag="AIC")[1]),
         "OK" if adfuller(d, regression="c", autolag="AIC")[1] < .05 else "WARN", "")
    # [5] dependence
    _head(5, "DEPENDENCE")
    L = min(lags, len(s) // 2 - 1)
    band = 1.96 / np.sqrt(len(s))
    a = acf(s, nlags=L, fft=True); pa = pacf(s, nlags=L, method="ywm")
    _row("95% band", float(band))
    for k in (1, 6, 12):
        if k <= L:
            _row(f"ACF lag {k}", float(a[k]), "", "계절 랙" if k == 12 else "")
    _row("ACF > band (count)", int((np.abs(a[1:]) > band).sum()), "", f"lag 1~{L}")
    _row("PACF > band (count)", int((np.abs(pa[1:]) > band).sum()), "", "")
    lb = float(acorr_ljungbox(s, lags=[min(12, L)], return_df=True)["lb_pvalue"].iloc[0])
    _row("Ljung-Box p (lag 12)", lb, "WARN" if lb < .05 else "OK", "자기상관 있음" if lb < .05 else "")
    # [6] long memory
    _head(6, "LONG MEMORY")
    dgph, se, m = _gph(s)
    _row("GPH d (se)", f"{dgph:.3f} ({se:.3f})", "WARN" if dgph >= .5 else "",
         f"m={m}. n이 작아 참고용" if len(s) < 200 else ("비정상" if dgph >= .5 else ""))
    last = int(np.max(np.where(np.abs(a[1:]) > band)[0]) + 1) if (np.abs(a[1:]) > band).any() else 0
    _row("last lag above band", last, "", "")
    return s


def residual_check(resid, k, name, lags=12):
    _CUR["name"] = name
    e = pd.Series(resid).dropna()
    _head(7, "RESIDUAL BASICS")
    sk, ku = float(stats.skew(e)), float(stats.kurtosis(e))
    _row("n", len(e)); _row("mean", float(e.mean()), "OK" if abs(e.mean()) < 1e-6 + .1 * e.std() else "WARN", "")
    _row("std", float(e.std()))
    _row("skewness", sk, "WARN" if abs(sk) > 1 else "OK", "")
    _row("excess kurtosis", ku, "WARN" if abs(ku) > 3 else "OK",
         "자기상관 검정 무력화 위험" if ku > 20 else "")
    jb = stats.jarque_bera(e)
    _row("Jarque-Bera p", float(jb.pvalue), "OK" if jb.pvalue > .05 else "WARN", "")
    _head(8, "RESIDUAL DYNAMICS")
    L = min(lags, len(e) // 2 - 1)
    lb = acorr_ljungbox(e, lags=[L], return_df=True)["lb_pvalue"].iloc[0]
    _row(f"Ljung-Box p (lag {L})", float(lb), "WARN" if lb < .05 else "OK",
         "잔차 자기상관 → HAC/DK 표준오차 필요" if lb < .05 else "")
    if lb > .999:
        _row("note", "p≈1", "WARN", "검정 무력화 의심 (첨도 확인)")
    lb2 = acorr_ljungbox(e ** 2, lags=[L], return_df=True)["lb_pvalue"].iloc[0]
    _row(f"LB on e² p (lag {L})", float(lb2), "WARN" if lb2 < .05 else "OK", "이분산 군집" if lb2 < .05 else "")
    arch_p = het_arch(e, nlags=min(6, L))[1]
    _row("ARCH-LM p (lag 6)", float(arch_p), "WARN" if arch_p < .05 else "OK", "")
    return e
