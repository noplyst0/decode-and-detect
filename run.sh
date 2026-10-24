#!/bin/sh
# Reproduce the submitted predictions (Cloudee, Decode & Detect 2026). Python 3.13, CPU only, ~10 min, ~1 GB RAM.
# macOS needs `brew install libomp` for LightGBM. Data: organiser files in data/train and data/test.
# Final = rank-average of up to three model configurations, then 5-minute burst smoothing. Set FINAL to choose.
set -e
FINAL="${FINAL:-v5 v5 v6c}"   # leaderboard-chosen; see NOTES.md
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -r requirements.txt; }
PY=.venv/bin/python
B="--train data/train --test data/test --stack 0 --seeds 5 --xgb --drop memo,probe --seed 42"
mkdir -p runs
files=""
rm -f runs/final_*.csv
for m in $FINAL; do
  files="$files runs/final_$m.csv"
  [ -f runs/final_$m.csv ] && continue   # a model listed twice counts double in the average but is built once
  case $m in
    v5)  USD_MODE=nominal              $PY src/pipeline.py $B --out runs/final_v5.csv ;;   # LB 63.9 alone
    v6a) USD_MODE=true WEIGHTS=hard0   $PY src/pipeline.py $B --out runs/final_v6a.csv ;;  # LB 62.6 alone
    v6c) USD_MODE=true                 $PY src/pipeline.py $B --out runs/final_v6c.csv ;;  # LB 62.8 alone
    v5s7)  USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_v5s7.csv --seed 7 ;;    # same as v5, seed 7
    v6cs7) USD_MODE=true    $PY src/pipeline.py $B --out runs/final_v6cs7.csv --seed 7 ;;   # same as v6c, seed 7
  esac
done
if [ "$(echo $FINAL | wc -w)" -gt 1 ]; then $PY src/rank_ensemble.py --out runs/final_ens.csv $files; else cp $files runs/final_ens.csv; fi
$PY src/smooth_sub.py --test data/test --sub runs/final_ens.csv --out submission.csv
$PY src/check_submission.py submission.csv data/test
