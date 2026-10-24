Second opinion, round 2: review our FINAL hackathon submission package before we upload it (deadline 16:30 London today).

Context: XRPL laundering-detection hackathon, metric = 100 x average precision on a hidden test week. Our final is
six LightGBM+XGBoost models (two USD modes x three seeds) rank-averaged, then 7.5-minute burst smoothing: leaderboard 66.1.
Attached: METHOD.md (the judged write-up, max 4 pages), README.md, AI_DISCLOSURE.md, NOTES.md (full leaderboard log),
run.sh, pipeline.py, rank_ensemble.py, smooth_sub.py (as .txt).

Since your last review we tested your suggestions on the leaderboard:
- RANKNORM=1: +0.4 with two models (65.8 vs 65.4) but 65.7 vs 66.1 with six. Six week-normalised + six normal = 66.1 (tie).
- Covariate-shift weights (DOMAIN_W): the train-vs-test domain classifier has AUC 1.000 and 94% of weights clip at 0.25.
- usd_rates now selects XRP explicitly (verified byte-identical output). The family-helper CV leak is noted in METHOD.md 7.
- JOINT_AUX is building now (six models).

Please answer only these, briefly:
1. Anything in METHOD.md or README.md that is wrong, inconsistent with the code or NOTES.md, or overclaims? Quote the line.
2. Would a judge be able to reproduce the CSV from run.sh as written? Any missing step, seed or dependency?
3. A domain-classifier AUC of 1.000 means some feature identifies the week. Which features in pipeline.py are most
   likely to do that (e.g. absolute time or amount levels), and is that a sign of a leak or overfitting we should fix in
   the last 2 hours, or is it harmless for ranking within one week?
Do not suggest new feature families or bigger ensembles.
