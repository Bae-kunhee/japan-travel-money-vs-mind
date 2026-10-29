"""One-off: build data/raw/ecb_krw_cross_monthly.csv from the ECB euro foreign exchange reference rates.

Source: European Central Bank euro reference rates (daily, eurofxref-hist), as bundled in the PyPI package
`currencyconverter` 0.18.22 (file currency_converter/eurofxref-hist.zip, data to 2026-09-14).
  pip download currencyconverter==0.18.22 --no-deps && unzip the wheel, then
  python3 fetch_ecb_fx.py path/to/eurofxref-hist.zip
KRW per unit of X = (KRW per EUR) / (X per EUR), daily, then monthly mean (JPY and IDR per 100 units).
Used only for the comparison countries ECOS was not downloaded for (SGD, MYR, IDR, HKD, USD).
JPY, THB and PHP are kept to check against ECOS (mean |log difference| 0.13%).
"""
import sys
import zipfile
import pandas as pd
from config import RAW

with zipfile.ZipFile(sys.argv[1]) as f:
    d = pd.read_csv(f.open(f.namelist()[0]))
d["Date"] = pd.to_datetime(d.Date)
d = d.set_index("Date").sort_index().loc["2014-12":"2026-09"]
units = {"JPY": 100, "THB": 1, "PHP": 1, "SGD": 1, "MYR": 1, "IDR": 100, "HKD": 1, "USD": 1}
out = pd.DataFrame({f"{c}{'' if u == 1 else u}": (d.KRW / d[c] * u).resample("MS").mean() for c, u in units.items()})
out.index = out.index.strftime("%Y-%m"); out.index.name = "month"
out.round(4).to_csv(RAW / "ecb_krw_cross_monthly.csv")
print(out.tail(3))
