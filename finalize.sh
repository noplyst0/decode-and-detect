#!/bin/sh
# Package the final entry: checks the predictions CSV, builds the code ZIP around it, copies both to ~/Desktop/FINAL.
#   ./finalize.sh path/to/final_predictions.csv
set -e
CSV="$1"; [ -f "$CSV" ] || { echo "usage: ./finalize.sh predictions.csv"; exit 1; }
.venv/bin/python src/check_submission.py "$CSV" data/test
cp "$CSV" submission.csv
H=$(git rev-parse --short HEAD)
sed -i '' "s/^Submitted commit: \`[a-z0-9]*\`/Submitted commit: \`$H\`/" README.md
./make_zip.sh
unzip -l code.zip | grep -qE " submission.csv$" && unzip -l code.zip | grep -qE " run.sh$" && unzip -l code.zip | grep -qE " METHOD.md$"
mkdir -p ~/Desktop/FINAL && cp submission.csv ~/Desktop/FINAL/FINAL_predictions.csv && cp code.zip ~/Desktop/FINAL/FINAL_code.zip
ls -lh ~/Desktop/FINAL
echo "Upload FINAL_predictions.csv + FINAL_code.zip, then: git add -A && git commit -m final && git push"
