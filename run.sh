#!/bin/sh
# Reproduce the submitted predictions (Cloudee, Decode & Detect 2026). Python 3.13, CPU only, ~45 min, ~1.5 GB RAM.
# macOS needs `brew install libomp` for LightGBM. Data: organiser files in data/train and data/test.
# Final = rank-average of six models (nominal and true-USD values x seeds 42, 7, 11, each with the joint and the
# pattern x difficulty helpers), test-week self-training (src/selftrain.py), then 7.5-minute burst smoothing.
set -e
FINAL="${FINAL:-v5fd v5fds7 v5fds11 v6cfd v6cfds7 v6cfds11}"   # leaderboard-chosen (67.3 with self-training); see NOTES.md
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
    v5fd)    FAMDIFF=1 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv ;;    # + pattern x difficulty helper
    v6cfd)   FAMDIFF=1 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v5fds*)  FAMDIFF=1 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5fds} ;;
    v6cfds*) FAMDIFF=1 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6cfds} ;;
    v5fm)    WEIGHTS=med3 FAMDIFF=1 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v6cfm)   WEIGHTS=med3 FAMDIFF=1 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v5fms*)  WEIGHTS=med3 FAMDIFF=1 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5fms} ;;
    v6cfms*) WEIGHTS=med3 FAMDIFF=1 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6cfms} ;;
    v5fa)    FAMAUX_AVG=1 FAMDIFF=1 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v6cfa)   FAMAUX_AVG=1 FAMDIFF=1 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v5fas*)  FAMAUX_AVG=1 FAMDIFF=1 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5fas} ;;
    v6cfas*) FAMAUX_AVG=1 FAMDIFF=1 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6cfas} ;;
    v5jm)    WEIGHTS=med3 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv ;;   # joint helper, medium x3
    v6cjm)   WEIGHTS=med3 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv ;;
    v5jms*)  WEIGHTS=med3 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5jms} ;;
    v6cjms*) WEIGHTS=med3 JOINT_AUX=1 USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6cjms} ;;
    v5s*)  USD_MODE=nominal $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v5s} ;;    # v5 with another seed
    v6cs*) USD_MODE=true    $PY src/pipeline.py $B --out runs/final_$m.csv --seed ${m#v6cs} ;;   # v6c with another seed
  esac
done
if [ "$(echo $FINAL | wc -w)" -gt 1 ]; then $PY src/rank_ensemble.py --out runs/final_ens.csv $files; else cp $files runs/final_ens.csv; fi
if [ "${SELFTRAIN:-1}" = 1 ]; then   # test-week self-training (src/selftrain.py) on the joint-helper model's test features
  if [ "${SELFTRAIN_FD:-0}" = 1 ]; then   # pseudo-model on the pattern x difficulty model's features instead
    FAMDIFF=1 JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --dump --out runs/feats_model.csv
  else
    JOINT_AUX=1 USD_MODE=nominal $PY src/pipeline.py $B --dump --out runs/feats_model.csv   # writes runs/test_feats.parquet
  fi
  $PY src/selftrain.py --sub runs/final_ens.csv --feats runs/test_feats.parquet --out runs/final_ens_st.csv \
    --K "${SELFTRAIN_K:-600}" --lam "${SELFTRAIN_L:-0.25}" --bag "${SELFTRAIN_BAG:-1}" --N "${SELFTRAIN_N:-4000}" \
    --leaves "${SELFTRAIN_LEAVES:-15}" --rounds "${SELFTRAIN_ROUNDS:-200}" ${SELFTRAIN_MIXED:+--train-feats runs/oof.parquet}
  mv runs/final_ens_st.csv runs/final_ens.csv
fi
$PY src/smooth_sub.py --test data/test --sub runs/final_ens.csv --out submission.csv --w "${SMOOTH_W:-450}" --a "${SMOOTH_A:-0.5}"
$PY -c "import pandas as pd; d = pd.read_csv('submission.csv'); d.sort_values('TRANSACTION_HASH').to_csv('submission.csv', index=False)"   # rows sorted by hash (scores unchanged)
$PY src/check_submission.py submission.csv data/test
