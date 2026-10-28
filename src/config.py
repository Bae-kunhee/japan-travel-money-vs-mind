"""Paths and constants shared by every script (all paths are relative to the repo root)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

KTO_XLSX = RAW / "kto_outbound_by_destination_202512.xlsx"   # 한국관광 데이터랩
FX_CSV = RAW / "ecos_fx_monthly_2015_2026.csv"                # 한국은행 ECOS
PANEL_CSV = PROCESSED / "panel.csv"

COUNTRIES = ["JP", "VN", "TH", "PH", "TW"]
NAME_KO = {"JP": "일본", "VN": "베트남", "TH": "태국", "PH": "필리핀", "TW": "대만"}

# estimation window: 2016-01 .. 2025-12 (t = 0 is 2015-01; 12 FX lags available from 2016-01)
T0, T1 = 12, 131
K_MAIN = 10          # lag length chosen by AIC in main_models.py (0..10 months)


def ym(y, m):
    """Month index used everywhere: 2015-01 -> 0."""
    return (y - 2015) * 12 + (m - 1)


for d in (PROCESSED, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)
