"""Rank-average several prediction CSVs (ours, a teammate's, another seed) into one submission.

    python src/ensemble.py sub_lgbm.csv sub_teammate.csv --weights 2 1 --out submission.csv
Ranks, not raw scores, so models on different scales mix fairly. AP only cares about order.
"""
import argparse, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+"); ap.add_argument("--weights", nargs="*", type=float)
ap.add_argument("--out", default="submission.csv")
a = ap.parse_args()
w = a.weights or [1.0] * len(a.files)
subs = [pd.read_csv(f).set_index("TRANSACTION_HASH")["score"] for f in a.files]
idx = subs[0].index
assert all(set(s.index) == set(idx) for s in subs), "files cover different transactions"
score = sum(wi * s.reindex(idx).rank(pct=True) for wi, s in zip(w, subs)) / sum(w)
pd.DataFrame({"TRANSACTION_HASH": idx, "score": score.clip(0, 1)}).to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(idx)} rows from {len(subs)} files, {score.nunique()} distinct scores")
