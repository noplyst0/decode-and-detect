#!/bin/sh
# Reproduce the submitted predictions. Python 3.13; macOS needs `brew install libomp` for LightGBM.
set -e
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/pipeline.py --train data/train --test data/test --out submission.csv --seed 42
