#!/bin/sh
# Code ZIP for upload: code, docs, final predictions + validation results. No data/venv/secrets. Limit 8 MiB.
set -e
rm -f code.zip
git ls-files | grep -vE '^(data|mock|runs)/|^CLAUDE.md$' | zip -q code.zip -@
[ -f submission.csv ] && zip -q code.zip submission.csv
[ -f runs/log.csv ] && zip -q code.zip runs/log.csv
ls -lh code.zip
