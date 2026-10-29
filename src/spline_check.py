"""Appendix: natural cubic spline linearity check (docs/spline_prereg.md, written before results).

Tried as a replacement for the XGBoost effect-size check. Not adopted: the pre-specified main index (A,
11-month simple mean) was unstable across knot counts (ratio 4.7-10.9x, nonlinearity Wald p < 0.001 with
5 knots), so the pre-specified replacement condition failed. Output: results/spline_check.json
"""
import json, warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
from config import ym, RESULTS
from panel_models import open_sample, load_panel, load_full, m2_fit, design, lag_cols

LN09 = np.log(0.9)
rng = np.random.default_rng(2026)
S = open_sample(load_panel()).reset_index(drop=True)
full = load_full()
base18 = full[full.year == 2018].groupby("country").ln_fx.mean()
boy = (S.boycott == 1).values
nb = ~boy

# ------------------------------------------------------------------ indices
LC = lag_cols(10)
o10, fit10, X10, V10 = m2_fit(S, 10)
w = np.array([o10["betas"][c][0] for c in LC]) / o10["theta"]
Z_A = S[LC].mean(1) - S.country.map(base18)
Z_B = S[LC].values @ w - S.country.map(base18) * w.sum()          # w sums to 1


def ns_basis(x, knots):
    """Natural cubic spline, ESL (5.4)-(5.5): returns nonlinear columns only (linear term = x itself)."""
    x = np.asarray(x, float); k = np.asarray(knots, float); K = len(k)
    d = lambda j: (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])
    return np.column_stack([d(j) - d(K - 2) for j in range(K - 2)])


def fit_spline(Z, qs, label):
    knots = np.quantile(Z[nb], qs)
    NLraw = ns_basis(Z, knots)
    sc = NLraw[nb].std(0)
    NL = pd.DataFrame(NLraw / sc, columns=[f"ns{j + 1}" for j in range(NLraw.shape[1])], index=S.index)
    D = S.copy(); D["ln_fx"] = Z.values
    o, fit, X, V = m2_fit(D, 0, extra=NL)
    cols = list(X.columns)
    ix = [cols.index("ln_fx")] + [cols.index(c) for c in NL.columns]
    ib = cols.index("boycott")
    b = fit.params.values

    def f_basis(z):
        return np.column_stack([z, ns_basis(z, knots) / sc])

    # Wald: nonlinear terms = 0
    inl = ix[1:]
    bn, Vn = b[inl], V[np.ix_(inl, inl)]
    W = float(bn @ np.linalg.solve(Vn, bn)); pW = float(stats.chi2.sf(W, len(inl)))
    # 10% yen effect, averaged over obs whose two points stay inside the boundary knots
    z = Z.values[nb]; h = LN09 / 2
    ok = (z + h >= knots[0]) & (z - h <= knots[-1])
    a = (f_basis(z[ok] + h) - f_basis(z[ok] - h)).mean(0)          # cheaper minus dearer
    fx10 = float(a @ b[ix]); g = float(b[ib])
    # simulation CI of the ratio
    sel = ix + [ib]
    dr = rng.multivariate_normal(b[sel], V[np.ix_(sel, sel)], 10000)
    rat = -dr[:, -1] / (dr[:, :-1] @ a)
    # local 10% effects at Japan's p10 / p50 / p90 of the index
    zj = Z.values[(S.country == "JP").values & nb]
    loc = {f"p{int(q * 100)}": float((f_basis(np.array([np.quantile(zj, q) + h])) -
                                      f_basis(np.array([np.quantile(zj, q) - h])))[0] @ b[ix])
           for q in (0.1, 0.5, 0.9)}
    zb = Z.values[boy]
    res = {"index": label, "knots_q": list(qs), "knots": knots.tolist(), "wald": W, "wald_p": pW, "df_nl": len(inl),
           "fx10_log": fx10, "fx10_pct": 100 * (np.exp(fx10) - 1), "gamma": g, "boycott_pct": 100 * (np.exp(g) - 1),
           "ratio": -g / fx10, "ratio_ci": np.percentile(rat, [2.5, 97.5]).tolist(),
           "local_fx10_pct": {k: 100 * (np.exp(v) - 1) for k, v in loc.items()},
           "boycott_index_pct_position": [float((Z.values[nb] < v).mean() * 100) for v in zb],
           "n_fx_used": int(ok.sum()), "aic": o["aic"]}
    return res, dict(X=X, b=b, V=V, ix=ix, ib=ib, knots=knots, sc=sc, f_basis=f_basis, NL=NL, D=D)


def fit_linear_index(Z, label):
    D = S.copy(); D["ln_fx"] = Z.values
    o, fit, X, V = m2_fit(D, 0)
    th, g = o["theta"], o["boycott"][0]
    return {"index": label, "theta": th, "fx10_pct": 100 * (np.exp(th * LN09) - 1), "gamma": g,
            "ratio": -g / (th * LN09), "aic": o["aic"]}, X


# ------------------------------------------------------------------ main fits
H4, H3, H5 = (0.05, 0.35, 0.65, 0.95), (0.10, 0.50, 0.90), (0.05, 0.275, 0.50, 0.725, 0.95)
results, store = {}, {}
for lab, Z in [("A", Z_A), ("B", Z_B)]:
    results[f"{lab}_linear"], _ = fit_linear_index(Z, lab)
    for kn, qs in [("k4", H4), ("k3", H3), ("k5", H5)]:
        if lab == "B" and kn != "k4":
            continue
        r, st = fit_spline(Z, qs, lab)
        results[f"{lab}_spline_{kn}"] = r; store[f"{lab}_{kn}"] = st
results["model2_K10"] = {"theta": o10["theta"], "gamma": o10["boycott"][0],
                         "ratio": -o10["boycott"][0] / (o10["theta"] * LN09)}

# ------------------------------------------------------------------ imputation (boycott-free fit), A k4
st = store["A_k4"]
Xi = st["X"].drop(columns=["boycott"]).values
Y = S.ln_v.values
cols_i = [c for c in st["X"].columns if c != "boycott"]
ix_i = [cols_i.index("ln_fx")] + [cols_i.index(c) for c in st["NL"].columns]
z = Z_A.values[nb]; h = LN09 / 2
ok = (z + h >= st["knots"][0]) & (z - h <= st["knots"][-1])
a_i = (st["f_basis"](z[ok] + h) - st["f_basis"](z[ok] - h)).mean(0)


def impute(rows):
    bb = np.linalg.lstsq(Xi[rows], Y[rows], rcond=None)[0]
    gam = float(np.mean(Y[boy] - Xi[boy] @ bb))
    return gam, float(a_i @ bb[ix_i])


gi, fi = impute(np.where(nb)[0])
tsn = np.sort(S.t[nb].unique()); rb = []
for _ in range(300):
    stt = rng.integers(0, len(tsn) - 11, int(np.ceil(len(tsn) / 12)))
    tt = np.concatenate([tsn[s:s + 12] for s in stt])[:len(tsn)]
    idx = np.concatenate([np.where((S.t.values == t) & nb)[0] for t in tt])
    gb, fb = impute(idx); rb.append(-gb / fb)
results["A_spline_k4_imputation"] = {"gamma": gi, "boycott_pct": 100 * (np.exp(gi) - 1), "fx10_pct": 100 * (np.exp(fi) - 1),
                                     "ratio": -gi / fi, "ratio_ci_block_bootstrap": np.percentile(rb, [2.5, 97.5]).tolist()}

# ------------------------------------------------------------------ rolling-origin out of sample (A index)
XA_lin = fit_linear_index(Z_A, "A")[1]
XA_sp = st["X"]
models = {"linear_K10": X10, "linear_index_A": XA_lin, "spline_index_A": XA_sp}
origins = [ym(y, m) for y in range(2018, 2026) for m in (6, 12) if ym(2018, 6) <= ym(y, m) <= ym(2025, 6)]
rows = []
for o_ in origins:
    test_t = range(o_ + 1, o_ + 7)
    tr = ((S.t <= o_) | ((S.country != "JP") & S.t.isin(test_t))).values
    te = ((S.country == "JP") & S.t.isin(test_t)).values
    if te.sum() == 0:
        continue
    for name, X in models.items():
        bb = np.linalg.lstsq(X.values[tr], Y[tr], rcond=None)[0]
        for t, p, yv in zip(S.t[te], X.values[te] @ bb, Y[te]):
            rows.append({"t": int(t), "model": name, "err": float(p - yv)})
O = pd.DataFrame(rows); Wd = O.pivot_table(index="t", columns="model", values="err")


def dm(e1, e2, lag=3):
    d = (e1 ** 2 - e2 ** 2).values; n = len(d); dc = d - d.mean()
    v = dc @ dc / n + 2 * sum((1 - l / (lag + 1)) * (dc[l:] @ dc[:-l]) / n for l in range(1, lag + 1))
    tdm = d.mean() / np.sqrt(v / n)
    return float(tdm), float(2 * stats.t.sf(abs(tdm), n - 1))


rmse = {m: float(100 * np.sqrt((Wd[m] ** 2).mean())) for m in Wd}
results["oos"] = {"rmse_pct": rmse, "n": int(len(Wd)),
                  "DM_spline_vs_linear_index": dm(Wd.spline_index_A, Wd.linear_index_A),
                  "DM_spline_vs_K10": dm(Wd.spline_index_A, Wd.linear_K10)}
json.dump(results, open(RESULTS / "spline_check.json", "w"), ensure_ascii=False, indent=1, default=float)
for k, v in results.items():
    print(k, json.dumps(v, ensure_ascii=False, default=lambda x: round(float(x), 3))[:600])
