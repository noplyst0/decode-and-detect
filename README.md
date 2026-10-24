# Decode & Detect: XRPL AI Hack entry

Team name: Cloudee. Captain: Faizan Arif. Members: Faizan Arif, Ibrahim Zubair.
Submitted commit: `c91d87d`. Leaderboard AP: `67.3`.

## Reproduce
`./run.sh` with the organiser data in `data/train` and `data/test` (seeds 42, 7 and 11, set in run.sh). It builds a venv from
`requirements.txt` and writes `submission.csv`. Check the output before upload:
`.venv/bin/python src/check_submission.py submission.csv data/test`.

Computing: Python 3.13, CPU only, about 50 minutes and ~1.5 GB peak RAM on an Apple M4 laptop. macOS needs
`brew install libomp` for LightGBM. No external services, APIs or network access at run time.

## Other leaderboard runs and how to rebuild each
The submitted file (67.3) is the `./run.sh` default. Other runs that also showed 67.3 rebuild with the same code:
- self-training weight 0.20: `SELFTRAIN_L=0.2 ./run.sh`; three averaged pseudo-models: `SELFTRAIN_BAG=3 ./run.sh`
- medium-weighted models: `FINAL="v5fm v5fms7 v5fms11 v6cfm v6cfms7 v6cfms11" ./run.sh`
- both model sets: `FINAL="v5fd v5fds7 v5fds11 v6cfd v6cfds7 v6cfds11 v5fm v5fms7 v5fms11 v6cfm v6cfms7 v6cfms11" ./run.sh`
Runs at 67.2:
- negatives below rank 6,000: `SELFTRAIN_N=6000 ./run.sh` (also with `SELFTRAIN_L=0.15` or `0.2`)
- medium-weighted models: `SELFTRAIN_N=6000 FINAL="v5fm v5fms7 v5fms11 v6cfm v6cfms7 v6cfms11" ./run.sh`
Without self-training (67.0): `SELFTRAIN=0 ./run.sh`; other 67.0 runs: `SELFTRAIN=0 SMOOTH_W=500 ./run.sh`,
`SELFTRAIN=0 SMOOTH_A=0.45 ./run.sh`, `SELFTRAIN=0 FINAL="v5fd v5fds7 v5fds11 v6cfd v6cfds7 v6cfds11 v5jm v5jms7 v6cjm v6cjms7" ./run.sh`.
Earlier test uploads used a placeholder code ZIP; the final entry is this commit.

## Third-party components
Python libraries pinned in `requirements.txt`: LightGBM (MIT), XGBoost (Apache-2.0), pandas, NumPy, scikit-learn,
NetworkX (BSD-3-Clause), PyArrow (Apache-2.0), Matplotlib (Matplotlib licence, BSD-style), tqdm (MIT/MPL-2.0). Data: the
organisers' supplied files only (not redistributed here). AI assistance is listed in `AI_DISCLOSURE.md`.

## Contents
`submission.csv` (submitted predictions), `runs/log.csv` (validation results per run), `NOTES.md` (experiment
log), `METHOD.md` (method report), `AI_DISCLOSURE.md` (AI models, prompts in `docs/`). Licence: MIT.
