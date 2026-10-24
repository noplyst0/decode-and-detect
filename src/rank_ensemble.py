"""Rank-average prediction CSVs (each score -> percentile rank, mean, rescaled to 0-1).
    python src/rank_ensemble.py --out ens.csv a.csv b.csv ..."""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("files", nargs="+")
a = ap.parse_args()
base = pd.read_csv(a.files[0]).set_index("TRANSACTION_HASH").score
cols = [base] + [pd.read_csv(f).set_index("TRANSACTION_HASH").score.reindex(base.index) for f in a.files[1:]]
assert all(c.notna().all() for c in cols), "files do not cover the same transactions"
s = sum(c.rank(pct=True) for c in cols) / len(cols)
s = (s - s.min()) / (s.max() - s.min())
pd.DataFrame({"TRANSACTION_HASH": base.index, "score": s.values}).to_csv(a.out, index=False)
print(f"wrote {a.out} from {len(cols)} files")
