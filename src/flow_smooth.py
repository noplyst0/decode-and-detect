"""Flow smoothing: link payment A -> B when B's sender is A's receiver, B leaves within --w seconds of A, and B moves
85-102% of A's USD value (money being passed on). Each payment's score is blended with the best score in its flow group.
Unlike wallet-sharing groups, flow groups stay small at long windows, so slow (medium) flows can be linked too.
    python src/flow_smooth.py --test data/test --sub in.csv --out out.csv [--w 1800 --b 0.3]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["USD_MODE"] = "true"  # true per-asset USD so values can be compared across conversions
import numpy as np, pandas as pd
import pipeline as P
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True); ap.add_argument("--sub", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--w", type=int, default=1800); ap.add_argument("--b", type=float, default=0.3)
ap.add_argument("--lo", type=float, default=0.85); ap.add_argument("--hi", type=float, default=1.02)
a = ap.parse_args()

e = P.build_edges(*P.load_split(a.test))
s = pd.read_csv(a.sub).set_index("TRANSACTION_HASH").score.reindex(e.index).to_numpy()
n = len(e)
c = pd.DataFrame({"i": np.arange(n), "s": e.SRC.astype(str).to_numpy(), "d": e.DST.astype(str).to_numpy(),
                  "t": e.TS.to_numpy(float), "ui": e.USD_RECV.to_numpy(float), "uo": e.USD_SENT.to_numpy(float)})
m = c.merge(c, left_on="d", right_on="s", suffixes=("", "_n"))
r = m.uo_n / m.ui
m = m[(m.t_n > m.t) & (m.t_n - m.t <= a.w) & (r >= a.lo) & (r <= a.hi) & (m.i != m.i_n)]
g = coo_matrix((np.ones(len(m)), (m.i.to_numpy(), m.i_n.to_numpy())), shape=(n, n))
lab = connected_components(g, directed=False)[1]
size = pd.Series(lab).map(pd.Series(lab).value_counts()).to_numpy()
top = pd.Series(s).groupby(lab).transform("max").to_numpy()
rk = lambda x: pd.Series(x).rank(pct=True).to_numpy()
out = (1 - a.b) * rk(s) + a.b * rk(top) + 1e-9 * s
out = (out - out.min()) / (out.max() - out.min())
pd.DataFrame({"TRANSACTION_HASH": e.index, "score": out}).to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(m)} flow links, groups>1: {(pd.Series(lab).value_counts() > 1).sum()}, "
      f"largest group {size.max()}, spearman vs input {pd.Series(out).corr(pd.Series(s), method='spearman'):.3f}")
