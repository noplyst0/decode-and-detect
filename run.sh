#!/bin/sh
# Reproduce the submitted predictions. Python 3.13; macOS needs `brew install libomp` for LightGBM.
set -e
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/pipeline.py --train data/train --test data/test --out submission.csv --seed 42 --seeds 5 --stack 0 --drop memo,probe
