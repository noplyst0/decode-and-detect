#!/bin/sh
# Reproduce the submitted predictions (Cloudee, Decode & Detect 2026). Python 3.13, CPU only, ~30 min, ~1 GB RAM.
# macOS needs `brew install libomp` for LightGBM. Data: organiser files in data/train and data/test.
# Final = rank-average of six models (nominal and true-USD values x seeds 42, 7, 11, each with the joint helper), then
# 7.5-minute burst smoothing.
set -e
FINAL="${FINAL:-v5ja v5jas7 v5jas11 v6cja v6cjas7 v6cjas11}"   # leaderboard-chosen (66.9); see NOTES.md
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
    v5rn)    RANKNORM=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv ;;                # week-normalised
    v6crn)   RANKNORM=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v5rns*)  RANKNORM=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5rns} ;;
    v6crns*) RANKNORM=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6crns} ;;
    v5ja)    JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv ;;              # + joint family/difficulty helper
    v6cja)   JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v5jas*)  JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5jas} ;;
    v6cjas*) JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6cjas} ;;
    v5s*)  USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5s} ;;    # v5 with another seed
    v6cs*) USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6cs} ;;   # v6c with another seed
  esac
done
if [ "$(echo $FINAL | wc -w)" -gt 1 ]; then $PY src/rank_ensemble.py --out runs/final_ens.csv $files; else cp $files runs/final_ens.csv; fi
$PY src/smooth_sub.py --test data/test --sub runs/final_ens.csv --out submission.csv --w "${SMOOTH_W:-450}"
$PY src/check_submission.py submission.csv data/test
