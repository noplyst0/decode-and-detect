"""Mock data only: break a submission's AP down by difficulty and shape (each group's positives vs all negatives).

    python src/eval_mock.py mock/sub.csv
"""
import sys, pandas as pd
from sklearn.metrics import average_precision_score as aps

sub = pd.read_csv(sys.argv[1]).set_index("TRANSACTION_HASH")["score"]
lab = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "mock/test_labels_hidden.csv").set_index("TRANSACTION_HASH")
df = lab.join(sub, how="inner")
neg = df[df.LABEL == 0]
print(f"overall AP {100 * aps(df.LABEL, df.score):6.2f}   ({int(df.LABEL.sum())} positives)")
top = df.sort_values("score", ascending=False).head(max(1, len(df) // 100))
print(f"top 1%: precision {top.LABEL.mean():.3f}, recall {top.LABEL.sum() / df.LABEL.sum():.3f}")
for col in ("DIFFICULTY", "SHAPE"):
    for k, g in df[df.LABEL == 1].groupby(col):
        part = pd.concat([g, neg])
        print(f"  {col.lower():10s} {k:15s} AP {100 * aps(part.LABEL, part.score):6.2f}  (n={len(g)})")
