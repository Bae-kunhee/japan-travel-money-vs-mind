#!/usr/bin/env bash
# Reproduce every number and figure: bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")/src"
python3 build_panel.py      # data/processed/panel.csv
python3 main_models.py      # results/main_models.json      (money effect, robustness, decomposition)
python3 money_vs_mind.py    # results/money_vs_mind.json    (boycott effect, break-even yen fall)
python3 demand_shift.py     # results/share_shift_2019.csv  (where the Japan demand went)
python3 yen_tiers.py        # results/yen_tiers.json        (effect by size of yen weakness)
python3 lag_sensitivity.py  # results/lag_sensitivity.csv   (appendix)
python3 diagnostics.py      # results/diagnostics_appendix.md (appendix: time-series diagnostics, SE sensitivity)
python3 figures.py          # figures/fig1..fig6.png
python3 xgb_robustness.py   # results/xgb_robustness.json  (XGBoost out-of-sample comparison; ~2 min)
python3 spline_check.py     # results/spline_check.json    (appendix: natural spline check, not adopted)
python3 boycott_stock.py    # results/boycott_stock.json   (search-volume boycott stock, fig8a/8b)
python3 comparison_expansion.py  # results/comparison_expansion.csv (more comparison countries, boycott window by destination, fig9a/9b)
python3 review_checks.py    # results/review_checks.json   (sample table, D test, event study, Japan-specific theta, drop 2020-01..02)
python3 review_figures.py   # figures/fig10..fig12.png     (event study, Japan vs comparators, both effects)
