"""(Written by the independent review session.) Apply 5-min burst smoothing to an existing predictions CSV (no retraining).
    .venv/bin/python src/smooth_sub.py --test data/test --sub sub_v6.csv --out sub_v6s.csv [--w 300 --a 0.5]"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import pipeline as P


def burst_smooth(e, score, w=300, a=0.5):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n, rows = len(e), np.arange(len(e))
    ev = pd.DataFrame({"w": np.r_[e.SRC.astype(str).to_numpy(), e.DST.astype(str).to_numpy()],
                       "t": np.r_[e.TS.to_numpy(float), e.TS.to_numpy(float)], "r": np.r_[rows, rows]})
    ev = ev[ev.w != "nan"].sort_values(["w", "t"])
    k = (ev.w.to_numpy()[1:] == ev.w.to_numpy()[:-1]) & (np.diff(ev.t.to_numpy()) <= w)
    g = coo_matrix((np.ones(k.sum()), (ev.r.to_numpy()[:-1][k], ev.r.to_numpy()[1:][k])), shape=(n, n))
    lab = connected_components(g, directed=False)[1]
    top = pd.Series(score).groupby(lab).transform("max").to_numpy()
    r = lambda x: pd.Series(x).rank(pct=True).to_numpy()
    return ((1 - a) * r(score) + a * r(top) + 1e-9 * score) / (1 + 1e-9)


ap = argparse.ArgumentParser()
ap.add_argument("--test", required=True); ap.add_argument("--sub", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--w", type=int, default=300); ap.add_argument("--a", type=float, default=0.5)
a = ap.parse_args()
e = P.build_edges(*P.load_split(a.test))
s = pd.read_csv(a.sub).set_index("TRANSACTION_HASH").score.reindex(e.index)
assert s.notna().all(), "submission does not cover every test tx"
out = pd.DataFrame({"TRANSACTION_HASH": e.index, "score": np.clip(burst_smooth(e, s.to_numpy(), a.w, a.a), 0, 1)})
assert out.TRANSACTION_HASH.is_unique and len(out) == len(e)
out.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(out)} rows, {out.score.nunique()} distinct, range {out.score.min():.4f}-{out.score.max():.4f}, "
      f"spearman vs input {pd.Series(out.score.values).corr(pd.Series(s.values), method='spearman'):.3f}")
