"""Shared model code.

Model 1 (Japan only, the 'naive' view)
    ln V_t = a + sum_k beta_k ln FX_{t-k} + gamma * boycott_t + month FE + e_t        (Newey-West SE)

Model 2 (5-country panel, main specification)
    ln V_ct = sum_k beta_k ln FX_{c,t-k} + gamma * boycott_ct
              + mu_{c,m}     country x calendar-month FE (seasonality of each destination)
              + lambda_t     year-month FE (anything common to all destinations: pandemic, Korean demand)
              + rho_e        months-since-reopening FE (revenge travel, aligned on each country's reopening)
              + e_ct                                                                   (Driscoll-Kraay SE)

theta = sum_k beta_k is the cumulative FX elasticity ('money'); gamma is the boycott effect ('mind').
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import PANEL_CSV, COUNTRIES, T0, T1


# ---------------------------------------------------------------- data
def load_full():
    return pd.read_csv(PANEL_CSV)


def load_panel():
    P = load_full()
    return P[(P.t >= T0) & (P.t <= T1)].copy()


def open_sample(P=None):
    """Months with borders open (closure dropped) and visitors observed."""
    P = load_panel() if P is None else P
    return P[(P.closed == 0)].dropna(subset=["ln_v"]).copy()


def japan_sample(P=None):
    P = load_panel() if P is None else P
    return P[(P.country == "JP") & (P.closed == 0)].copy().sort_values("t")


# ---------------------------------------------------------------- helpers
def lag_cols(K):
    return ["ln_fx"] + [f"ln_fx_l{k}" for k in range(1, K + 1)]


def nw_bw(T):
    """Newey-West rule-of-thumb bandwidth floor(4 (T/100)^(2/9))."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def sum_vector(cols, names):
    return np.array([1.0 if c in names else 0.0 for c in cols])


# ---------------------------------------------------------------- Model 1
def m1_fit(K, data, cov=True):
    mdum = pd.get_dummies(pd.Categorical(data.month, categories=range(1, 13)),
                          prefix="m", drop_first=True, dtype=float)
    mdum.index = data.index
    X = sm.add_constant(pd.concat([data[lag_cols(K)], data[["boycott"]], mdum], axis=1))
    if cov:
        return sm.OLS(data.ln_v, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_bw(len(data))}), X
    return sm.OLS(data.ln_v, X).fit(), X


# ---------------------------------------------------------------- Model 2
def design(df, K, event_col="event", use_event=True, extra=None, trends=False, leads=0):
    """Dummy-variable design: country x calendar-month FE, year-month FE, event-time FE."""
    df = df.copy()
    parts = [df[lag_cols(K)]]
    if leads:
        parts.append(df[[f"ln_fx_f{k}" for k in range(1, leads + 1)]])
    parts.append(df[["boycott"]])
    cm = pd.get_dummies(df.country + "_" + df.month.astype(str), prefix="cm", dtype=float)
    parts.append(cm.iloc[:, 1:])
    tt = pd.get_dummies(df.t, prefix="t", dtype=float)
    parts.append(tt.iloc[:, 1:])
    if use_event:
        e = df[event_col]
        # bins 0..17 and 18+; -1 = pre-pandemic. e_-1 is dropped here; with year-month FE the 18+ bin
        # is also collinear (post-reopening is a function of t) and is pruned below. So the event-time
        # coefficients are measured relative to the long-run (18+) level, and the pre-pandemic vs 18+
        # difference is absorbed by lambda_t.
        eb = np.where(e.isna(), -1, np.minimum(e, 18)).astype(int)
        ed = pd.get_dummies(pd.Series(eb, index=df.index), prefix="e", dtype=float)
        ed = ed.drop(columns=["e_-1"], errors="ignore")
        parts.append(ed)
    if trends:
        for c in COUNTRIES[1:]:
            parts.append(pd.Series((df.country == c) * df.t / 12.0, name=f"trend_{c}", index=df.index))
    if extra is not None:
        parts.append(extra)
    X = sm.add_constant(pd.concat(parts, axis=1))
    # drop exactly collinear columns (keeps the first column of each collinear set)
    keep, rank = [], 0
    for c in X.columns:
        r = np.linalg.matrix_rank(X[keep + [c]].values)
        if r > rank:
            keep.append(c)
            rank = r
    return X[keep]


def dk_cov(y, X, df):
    """Driscoll-Kraay covariance via linearmodels (year-month dummies -> time_effects).
    Returns a matrix aligned with X.columns (zeros for absorbed columns)."""
    from linearmodels.panel import PanelOLS
    tcols = [c for c in X.columns if c.startswith("t_")]
    Xp = X.drop(columns=tcols + ["const"]).copy()
    idx = pd.MultiIndex.from_arrays([df.country.values, df.t.values])
    Xp.index = idx
    yy = pd.Series(y.values, index=idx)
    m = PanelOLS(yy, Xp, time_effects=True, drop_absorbed=True, check_rank=False).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=nw_bw(df.t.nunique()))
    V = pd.DataFrame(0.0, index=X.columns, columns=X.columns)
    V.loc[m.cov.index, m.cov.columns] = m.cov.values
    return V.values


def m2_fit(df, K, **kw):
    """Fit Model 2. Returns (summary dict, statsmodels fit, design X, DK covariance V)."""
    X = design(df, K, **kw)
    fit = sm.OLS(df.ln_v, X).fit()
    V = dk_cov(df.ln_v, X, df)
    cols = list(X.columns)
    lc = [c for c in lag_cols(K) if c in cols]
    a = sum_vector(cols, lc)
    out = {"theta": float(a @ fit.params.values), "se": float(np.sqrt(a @ V @ a)),
           "n": int(fit.nobs), "k": int(X.shape[1]), "aic": float(fit.aic),
           "betas": {c: [float(fit.params[c]), float(np.sqrt(V[cols.index(c), cols.index(c)]))] for c in lc}}
    if "boycott" in cols:
        i = cols.index("boycott")
        out["boycott"] = [float(fit.params["boycott"]), float(np.sqrt(V[i, i]))]
    return out, fit, X, V


def theta_gamma(df, K, extra=None, gamma_col="boycott", **kw):
    """theta (sum of FX lags), gamma (a social-issue dummy) and their (co)variances."""
    out, fit, X, V = m2_fit(df, K, extra=extra, **kw)
    cols = list(X.columns)
    a = sum_vector(cols, [c for c in lag_cols(K) if c in cols])
    b = sum_vector(cols, [gamma_col])
    th, ga = a @ fit.params.values, b @ fit.params.values
    return th, ga, a @ V @ a, b @ V @ b, a @ V @ b, fit, X, V


def breakeven(th, ga, vt, vg, cov):
    """Yen fall (%) whose effect exactly offsets the social-issue shock: theta * ln(r) = -gamma.
    Delta-method CI on d = -gamma/theta (see money_vs_mind.py for the simulation CI)."""
    d = -ga / th
    gd = np.array([ga / th ** 2, -1.0 / th])
    sd = np.sqrt(gd @ np.array([[vt, cov], [cov, vg]]) @ gd)
    fall = lambda x: 100 * (1 - np.exp(x))
    return {"dlnfx": float(d), "se": float(sd), "fall_pct": float(fall(d)),
            "ci_fall_pct": [float(fall(d + 1.96 * sd)), float(fall(d - 1.96 * sd))]}


def pct10(theta):
    """Visitors % change when the destination currency gets 10% cheaper."""
    return 100 * (np.exp(theta * np.log(0.9)) - 1)
