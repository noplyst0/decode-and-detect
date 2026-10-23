#!/bin/sh
# Code ZIP for upload: code + docs only, no data/venv/secrets. Limit 8 MiB.
set -e
rm -f code.zip
git ls-files | grep -vE '^(data|mock|runs)/|^CLAUDE.md$' | zip -q code.zip -@
ls -lh code.zip
