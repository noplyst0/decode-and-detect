# Decode & Detect: XRPL AI Hack entry

Team name: Cloudee. Captain: Faizan Arif. Members: Faizan Arif, Ibrahim Zubair.
Submitted commit: `abd00fa`. Leaderboard AP: `67.0`.

## Reproduce
`./run.sh` with the organiser data in `data/train` and `data/test` (seeds 42, 7 and 11, set in run.sh). It builds a venv from
`requirements.txt` and writes `submission.csv`. Check the output before upload:
`.venv/bin/python src/check_submission.py submission.csv data/test`.

Computing: Python 3.13, CPU only, about 45 minutes and ~1.5 GB peak RAM on an Apple M4 laptop. macOS needs
`brew install libomp` for LightGBM. No external services, APIs or network access at run time.

## Third-party components
Python libraries pinned in `requirements.txt`: LightGBM (MIT), XGBoost (Apache-2.0), pandas, NumPy, scikit-learn,
NetworkX (BSD-3-Clause), PyArrow (Apache-2.0), Matplotlib (Matplotlib licence, BSD-style), tqdm (MIT/MPL-2.0). Data: the
organisers' supplied files only (not redistributed here). AI assistance is listed in `AI_DISCLOSURE.md`.

## Contents
`submission.csv` (submitted predictions), `runs/log.csv` (validation results per run), `NOTES.md` (experiment
log), `METHOD.md` (method report), `AI_DISCLOSURE.md` (AI models, prompts in `docs/`). Licence: MIT.
