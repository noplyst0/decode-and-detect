"""Pattern x difficulty helper (FAMDIFF=1, on top of JOINT_AUX=1). One multiclass model predicts family__difficulty for
planted payments (e.g. stack__medium vs stack__easy) and family__0 for innocent ones, out of fold on the main CV groups.
The joint helper merges easy and medium positives of a pattern into one class; this gives the main model an explicit
per-pattern medium signal, aimed at the medium cases that rank below their innocent look-alikes."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from gemini_modelling import _make_folds


def helper_family_difficulty(F, Ft, y, aux, groups, seed=42, min_rows=15):
    fam = aux["family"].fillna("background").astype(str).str.lower().to_numpy()
    diff = aux["difficulty"].fillna("normal").astype(str).str.lower().to_numpy()
    yb = y.to_numpy().astype(int)
    lab = np.array([f"{f}__{d}" if t == 1 else f"{f}__0" for f, d, t in zip(fam, diff, yb)])
    counts = pd.Series(lab).value_counts()
    rare = set(counts[counts < min_rows].index)
    lab = np.array([f"other__{s.split('__')[1]}" if s in rare else s for s in lab])
    classes, yk = np.unique(lab, return_inverse=True)
    params = {"objective": "multiclass", "num_class": len(classes), "metric": "multi_logloss", "learning_rate": 0.05,
              "num_leaves": 15, "min_child_samples": 18, "feature_fraction": 0.65, "bagging_fraction": 0.8,
              "bagging_freq": 1, "reg_lambda": 2.0, "verbosity": -1, "n_jobs": -1, "seed": seed + 7,
              "force_col_wise": True, "deterministic": True}
    oof = np.zeros((len(F), len(classes)))
    test = np.zeros((len(Ft), len(classes)))
    folds = _make_folds(groups, n_splits=5)
    for tr, va in folds:
        m = lgb.train(params, lgb.Dataset(F.iloc[tr], label=yk[tr]), num_boost_round=350,
                      valid_sets=[lgb.Dataset(F.iloc[va], label=yk[va])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(F.iloc[va], num_iteration=m.best_iteration)
        test += m.predict(Ft, num_iteration=m.best_iteration) / len(folds)
    cols = [f"hd_{c}" for c in classes]
    a = pd.DataFrame(oof, index=F.index, columns=cols)
    b = pd.DataFrame(test, index=Ft.index, columns=cols)
    for d in ("easy", "medium", "hard"):
        idx = [i for i, c in enumerate(classes) if c.endswith("__" + d)]
        a[f"hd_{d}_total"], b[f"hd_{d}_total"] = oof[:, idx].sum(1), test[:, idx].sum(1)
    return a, b
