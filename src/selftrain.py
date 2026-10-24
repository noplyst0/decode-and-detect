"""Self-training on the test week (no labels): the ensemble's top K test payments are pseudo-labelled suspicious and
those ranked below N innocent; a small LightGBM is cross-fitted on the test week's own features (from the joint-helper
model, `pipeline.py --dump`) and its rank is blended in: score = rank(ensemble) + lam * rank(pseudo-model).
It adapts the ranking to the test week's shift and lifts other payments of confidently flagged flows.
    python src/selftrain.py --sub ens.csv --feats runs/test_feats.parquet --out out.csv --K 600 --lam 0.25"""
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold


def selftrain(X, score, K, N=6000, seed=0):
    r = pd.Series(score, index=X.index).rank(ascending=False).to_numpy()
    lab = np.where(r <= K, 1, np.where(r > N, 0, -1))
    pred = np.zeros(len(X))
    p = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=20, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=seed, force_col_wise=True,
             deterministic=True)
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        t = tr[lab[tr] >= 0]
        m = lgb.train(p, lgb.Dataset(X.iloc[t], lab[t]), 200)
        pred[va] = m.predict(X.iloc[va])
    return pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True); ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=600); ap.add_argument("--lam", type=float, default=0.25)
    ap.add_argument("--bag", type=int, default=1); ap.add_argument("--N", type=int, default=6000)
    a = ap.parse_args()
    ens = pd.read_csv(a.sub).set_index("TRANSACTION_HASH").score
    X = pd.read_parquet(a.feats).reindex(ens.index)
    rk = lambda v: pd.Series(np.asarray(v, float), index=ens.index).rank(pct=True)
    pr = sum(rk(selftrain(X, ens.to_numpy(), a.K, N=a.N, seed=s)) for s in range(a.bag)) / a.bag if a.bag > 1 \
        else rk(selftrain(X, ens.to_numpy(), a.K, N=a.N))
    s = rk(ens) + a.lam * rk(pr)
    s = (s - s.min()) / (s.max() - s.min())
    pd.DataFrame({"TRANSACTION_HASH": ens.index, "score": s.values}).to_csv(a.out, index=False)
    print(f"wrote {a.out}: K={a.K} lam={a.lam} bag={a.bag}, spearman vs input {ens.corr(s, method='spearman'):.3f}")
