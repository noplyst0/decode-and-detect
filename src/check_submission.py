"""Check a predictions CSV before upload: python src/check_submission.py sub.csv data/test
Every test tx exactly once, header TRANSACTION_HASH,score, scores finite in [0,1], file <= 12 MiB, few ties."""
import glob, os, sys
import numpy as np, pandas as pd

csv, test_dir = sys.argv[1], sys.argv[2]
ok = True
def fail(msg):
    global ok; ok = False; print("FAIL:", msg)

size = os.path.getsize(csv)
if size > 12 * 2**20: fail(f"file is {size / 2**20:.1f} MiB (limit 12)")
with open(csv) as f: header = f.readline().strip()
if header != "TRANSACTION_HASH,score": fail(f"header is {header!r}")

sub = pd.read_csv(csv, dtype={"TRANSACTION_HASH": str})
tx_file = [p for p in glob.glob(os.path.join(test_dir, "*.parquet")) if "transaction" in os.path.basename(p).lower()]
tx = pd.read_parquet(tx_file[0])
hcol = "TRANSACTION_HASH" if "TRANSACTION_HASH" in tx.columns else next(c for c in tx.columns if "TX" in c.upper() and "HASH" in c.upper())
want = set(tx[hcol].astype(str))
got = sub.TRANSACTION_HASH

if got.duplicated().any(): fail(f"{got.duplicated().sum()} duplicated hashes")
missing, extra = want - set(got), set(got) - want
if missing: fail(f"{len(missing)} test tx missing")
if extra: fail(f"{len(extra)} hashes not in the test set")
s = sub.score
if not np.isfinite(s).all(): fail(f"{(~np.isfinite(s)).sum()} non-finite scores")
if (s < 0).any() or (s > 1).any(): fail("scores outside [0,1]")

# Tied scores are ranked as one block, which blurs the order: report how much of the top is tied.
top = s.sort_values(ascending=False).head(402)
print(f"rows {len(sub)} / test {len(want)} | {size / 2**20:.2f} MiB | distinct scores {s.nunique()} "
      f"| top-402 distinct {top.nunique()}")
print("OK" if ok else "NOT READY")
sys.exit(0 if ok else 1)
