"""Decode & Detect: score every test transaction for laundering-style behaviour.

    python src/pipeline.py --train data/train --test data/test --out submission.csv --seed 42

Stages: load (SonarX schema) -> one edge per transaction in USD -> features (tx, account, timing,
graph) -> LightGBM, 5-fold grouped CV scored by Average Precision -> optional stacking on
account-level OOF scores -> predictions CSV (TRANSACTION_HASH,score).
Features are computed inside each split, since train and test are separate 7-day windows.
"""
import argparse, json, os, subprocess, time
import numpy as np, pandas as pd, networkx as nx, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

XRP_TOKEN = "0x1111222211112222111122221111222211112222"
HOUR, DAY = 3600, 86400


# ---------- load: adjust here if the real README names things differently ----------
def to_unix(x):
    """Seconds since 1970. Unit-safe: pandas 3 keeps ms/us resolution, so astype(int64) is not seconds."""
    return (pd.to_datetime(x, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def upper(df):
    df.columns = [c.upper() for c in df.columns]
    return df


def find_col(df, *keys, exclude=()):
    for c in df.columns:
        if any(k in c for k in keys) and c not in exclude:
            return c
    raise KeyError(f"none of {keys} in {list(df.columns)}")


def load_split(d):
    tx = upper(pd.read_parquet(f"{d}/transactions.parquet"))
    tr = upper(pd.read_parquet(f"{d}/transfers.parquet"))
    rates = upper(pd.read_csv(f"{d}/market_rates.csv"))
    for df in (tx, tr):
        df["TS"] = to_unix(df["DATETIME"])
    return tx, tr, rates


def load_labels(d):
    lab = upper(pd.read_csv(f"{d}/labels.csv"))
    y = find_col(lab, "LABEL", "SUSP", "TARGET", "IS_", exclude=("TRANSACTION_HASH",))
    return lab.set_index("TRANSACTION_HASH")[y].astype(int)


def usd_rates(rates):
    """market_rates.csv -> lookup keyed by (hour, asset key)."""
    tcol = find_col(rates, "HOUR", "TIME", "DATE", "TS")
    pcol = find_col(rates, "USD", "PRICE", "RATE", exclude=(tcol,))
    r = rates.copy()
    r["H"] = to_unix(r[tcol]) // HOUR
    key = [k for k in ("TOKEN_ADDRESS", "SYMBOL") if k in r.columns]
    return r[["H", *key, pcol]].rename(columns={pcol: "USD_PRICE"}), key


def native_value(tr):
    if "VALUE" in tr.columns:
        return tr["VALUE"].astype(float)
    xrp = tr["SYMBOL"].eq("XRP")
    return np.where(xrp, tr["SOURCE_VALUE"].astype(float) / 1e6, tr["SOURCE_VALUE"].astype(float))


def build_edges(tx, tr, rates):
    """One row per transaction: who paid whom, what asset, how much in USD, when."""
    tr = tr.copy()
    tr["V"] = native_value(tr)
    r, key = usd_rates(rates)
    tr["H"] = tr["TS"] // HOUR
    tr = tr.merge(r, on=["H", *key], how="left")
    if tr["USD_PRICE"].isna().any() and "SYMBOL" in r.columns:  # fall back to nearest hour, same symbol
        r2 = r.groupby(["SYMBOL"])["USD_PRICE"].median().rename("P2")
        tr = tr.join(r2, on="SYMBOL")
        tr["USD_PRICE"] = tr["USD_PRICE"].fillna(tr["P2"])
    tr["USD"] = tr["V"] * tr["USD_PRICE"]
    tr["ASSET"] = tr["TOKEN_ADDRESS"].astype(str) + ":" + tr["SYMBOL"].astype(str)
    tr = tr.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    g = tr.groupby("TRANSACTION_HASH", sort=False)
    first, last = g.first(), g.last()
    legs = pd.DataFrame({
        "N_LEGS": g.size(), "SRC_LEG": first["FROM_ADDRESS"], "DST": last["TO_ADDRESS"],
        "ASSET_SENT": first["ASSET"], "ASSET_RECV": last["ASSET"],
        "USD_SENT": first["USD"], "USD_RECV": last["USD"], "V_SENT": first["V"],
        "N_ASSETS_TX": g["ASSET"].nunique()})
    e = tx.set_index("TRANSACTION_HASH")[["ACCOUNT", "TS", "TRANSACTION_TYPE", "FEE"]].join(legs)
    extra = tx.set_index("TRANSACTION_HASH")["EXTRA_TRANSACTION_FIELDS"].map(
        lambda s: json.loads(s) if isinstance(s, str) and s.startswith("{") else {})
    e["DST"] = e["DST"].fillna(extra.map(lambda x: x.get("Destination")))
    e["HAS_DEST_TAG"] = extra.map(lambda x: "DestinationTag" in x).astype(int)
    e["HAS_SENDMAX"] = extra.map(lambda x: "SendMax" in x).astype(int)
    e["HAS_MEMO"] = tx.set_index("TRANSACTION_HASH")["MEMOS"].notna().astype(int) if "MEMOS" in tx else 0
    e["SUCCESS"] = tx.set_index("TRANSACTION_HASH")["TRANSACTION_RESULT"].eq("tesSUCCESS").astype(int)
    e = e.rename(columns={"ACCOUNT": "SRC"})
    return e


# ---------- features ----------
def tx_features(e):
    f = pd.DataFrame(index=e.index)
    f["log_usd_sent"] = np.log1p(e.USD_SENT)
    f["log_usd_recv"] = np.log1p(e.USD_RECV)
    f["recv_over_sent"] = e.USD_RECV / e.USD_SENT.replace(0, np.nan)
    f["n_legs"] = e.N_LEGS
    f["is_conversion"] = (e.ASSET_SENT != e.ASSET_RECV).astype(int)
    f["sent_is_xrp"] = e.ASSET_SENT.astype(str).str.endswith(":XRP").astype(int)
    f["fee_drops"] = e.FEE * 1e6
    f["hour"] = (e.TS // HOUR) % 24
    f["dow"] = (e.TS // DAY) % 7
    f["t_rel"] = (e.TS - e.TS.min()) / DAY
    f["round_native"] = (np.round(e.V_SENT, 2) % 1 == 0).astype(int)
    f["round_usd_100"] = (np.round(e.USD_SENT) % 100 == 0).astype(int)
    f["self_pay"] = (e.SRC == e.DST).astype(int)
    for c in ("HAS_DEST_TAG", "HAS_SENDMAX", "HAS_MEMO", "SUCCESS"):
        f[c.lower()] = e[c]
    f["tx_type"] = e.TRANSACTION_TYPE.astype("category").cat.codes
    return f


def account_table(e):
    out = e.groupby("SRC").agg(n_out=("TS", "size"), usd_out=("USD_SENT", "sum"), cp_out=("DST", "nunique"),
                               out_mean=("USD_SENT", "mean"), out_std=("USD_SENT", "std"),
                               first_out=("TS", "min"), last_out=("TS", "max"), conv_out=("N_LEGS", lambda s: (s > 1).mean()))
    inc = e.groupby("DST").agg(n_in=("TS", "size"), usd_in=("USD_RECV", "sum"), cp_in=("SRC", "nunique"),
                               first_in=("TS", "min"), last_in=("TS", "max"))
    a = out.join(inc, how="outer")
    for c in ("n_out", "n_in", "usd_out", "usd_in", "cp_out", "cp_in"):
        a[c] = a[c].fillna(0)
    lo, hi = a[["usd_in", "usd_out"]].min(axis=1), a[["usd_in", "usd_out"]].max(axis=1)
    a["pass_through"] = (lo / hi.replace(0, np.nan)).fillna(0)
    a["net_flow"] = (a.usd_in - a.usd_out) / (a.usd_in + a.usd_out).replace(0, np.nan)
    a["out_cv"] = a.out_std / a.out_mean
    a["first_seen"] = a[["first_out", "first_in"]].min(axis=1)
    a["span_h"] = (a[["last_out", "last_in"]].max(axis=1) - a.first_seen) / HOUR
    a["in_to_out_h"] = (a.first_out - a.first_in) / HOUR  # funded, then spends
    a["degree"] = a.cp_in + a.cp_out
    a["n_tx"] = a.n_in + a.n_out
    keep = ["n_out", "n_in", "cp_out", "cp_in", "degree", "n_tx", "pass_through", "net_flow", "out_cv",
            "span_h", "in_to_out_h", "conv_out", "first_seen"]
    a = a[keep].copy()
    a["log_usd_out"], a["log_usd_in"] = np.log1p(out.usd_out.reindex(a.index)), np.log1p(inc.usd_in.reindex(a.index))
    return a


def window_count(e, key, w):
    """For each tx: how many txs share its `key` account within +-w seconds (burst / fan size)."""
    res = pd.Series(0, index=e.index, dtype=float)
    for _, grp in e.groupby(key):
        t = grp.TS.to_numpy()
        order = np.argsort(t); ts = t[order]
        cnt = np.searchsorted(ts, ts + w, "right") - np.searchsorted(ts, ts - w, "left")
        res.loc[grp.index[order]] = cnt
    return res


def timing_features(e):
    f = pd.DataFrame(index=e.index)
    s = e.reset_index().sort_values("TS")
    incoming = s[["DST", "TS", "USD_RECV"]].rename(columns={"DST": "ACC", "TS": "T_IN", "USD_RECV": "USD_IN"}).sort_values("T_IN")
    outgoing = s[["SRC", "TS", "USD_SENT"]].rename(columns={"SRC": "ACC", "TS": "T_OUT", "USD_SENT": "USD_OUT"}).sort_values("T_OUT")
    # sender: how long since it was last funded, and is this payment ~ that funding (pass-through)?
    m = pd.merge_asof(s[["TRANSACTION_HASH", "TS", "SRC", "USD_SENT"]], incoming.dropna(subset=["ACC"]),
                      left_on="TS", right_on="T_IN", left_by="SRC", right_by="ACC", direction="backward",
                      allow_exact_matches=False).set_index("TRANSACTION_HASH")
    f["src_since_funded_h"] = (m.TS - m.T_IN) / HOUR
    f["src_fwd_ratio"] = m.USD_SENT / m.USD_IN
    # receiver: how soon does it pay out again, and how much of this does it forward?
    m = pd.merge_asof(s[["TRANSACTION_HASH", "TS", "DST", "USD_RECV"]].dropna(subset=["DST"]), outgoing,
                      left_on="TS", right_on="T_OUT", left_by="DST", right_by="ACC", direction="forward",
                      allow_exact_matches=False).set_index("TRANSACTION_HASH")
    f["dst_next_out_h"] = (m.T_OUT - m.TS) / HOUR
    f["dst_fwd_ratio"] = m.USD_OUT / m.USD_RECV
    f["fwd_both"] = ((f.src_fwd_ratio.between(0.9, 1.0)) & (f.dst_fwd_ratio.between(0.9, 1.0))).astype(int)
    for w, tag in ((HOUR, "1h"), (6 * HOUR, "6h")):
        f[f"src_burst_{tag}"] = window_count(e, "SRC", w)
        f[f"dst_fanin_{tag}"] = window_count(e.dropna(subset=["DST"]), "DST", w).reindex(e.index)
    return f


def graph_features(e, acct, hub_q=0.995):
    """Pair, reciprocity, cycle (same SCC) and scatter-gather features, with hub accounts removed."""
    f = pd.DataFrame(index=e.index)
    ed = e.dropna(subset=["DST"])
    pair = ed.groupby(["SRC", "DST"]).size().rename("pair_n")
    f["pair_n"] = ed.join(pair, on=["SRC", "DST"]).pair_n
    rev = set(zip(ed.DST, ed.SRC))
    f["reciprocal"] = [int((s, d) in rev) for s, d in zip(e.SRC, e.DST)]
    hubs = set(acct.index[acct.degree > max(acct.degree.quantile(hub_q), 20)])
    f["src_is_hub"], f["dst_is_hub"] = e.SRC.isin(hubs).astype(int), e.DST.isin(hubs).astype(int)
    core = ed[~ed.SRC.isin(hubs) & ~ed.DST.isin(hubs) & (ed.SRC != ed.DST)]
    G = nx.DiGraph(); G.add_edges_from(zip(core.SRC, core.DST))
    scc = pd.Series({n: i for i, comp in enumerate(nx.strongly_connected_components(G)) for n in comp})
    wcc = pd.Series({n: i for i, comp in enumerate(nx.weakly_connected_components(G)) for n in comp})
    scc_size, wcc_size = scc.map(scc.value_counts()), wcc.map(wcc.value_counts())
    f["same_scc"] = [int(s in scc and d in scc and scc[s] == scc[d]) for s, d in zip(e.SRC, e.DST)]
    f["scc_size"] = e.SRC.map(scc_size)
    f["wcc_size"] = e.SRC.map(wcc_size)
    # scatter-gather: for s->m, how many distinct middlemen link s to the same far-end d?
    p = core[["SRC", "DST"]].drop_duplicates()
    two = p.merge(p, left_on="DST", right_on="SRC", suffixes=("", "_2"))
    two = two[two.SRC != two.DST_2]
    sg = two.groupby(["SRC", "DST_2"]).DST.nunique().rename("sg")
    two = two.join(sg, on=["SRC", "DST_2"])
    first_hop = two.groupby(["SRC", "DST"]).sg.max()
    second_hop = two.groupby(["DST", "DST_2"]).sg.max().rename_axis(["SRC", "DST"])
    f["sg_first_hop"] = e.join(first_hop.rename("x"), on=["SRC", "DST"]).x
    f["sg_second_hop"] = e.join(second_hop.rename("x"), on=["SRC", "DST"]).x
    groups = e.SRC.map(wcc).fillna(-1)
    return f, groups


def features(e):
    acct = account_table(e)
    s = acct.add_prefix("s_").reindex(e.SRC.values).set_axis(e.index)
    d = acct.add_prefix("d_").reindex(e.DST.values).set_axis(e.index)
    s["s_age_h"] = (e.TS - s.s_first_seen) / HOUR
    d["d_age_h"] = (e.TS - d.d_first_seen) / HOUR
    s, d = s.drop(columns="s_first_seen"), d.drop(columns="d_first_seen")
    g, _ = graph_features(e, acct)
    groups = (e.TS - e.TS.min()) // DAY  # day blocks: one giant graph component breaks component-grouped folds
    X = pd.concat([tx_features(e), s, d, timing_features(e), g], axis=1)
    return X.astype(float), groups


def stack_features(e, score):
    """Account-level suspicion from stage-1 scores, leaving each tx's own score out."""
    p = pd.Series(score, index=e.index)
    long = pd.concat([pd.DataFrame({"ACC": e.SRC, "p": p}), pd.DataFrame({"ACC": e.DST, "p": p})])
    long = long[long.ACC.notna()].rename_axis("H").reset_index().sort_values("p", ascending=False)
    long["rank"] = long.groupby("ACC").cumcount()
    agg = long.groupby("ACC").agg(tot=("p", "sum"), n=("p", "size"))
    top = long[long["rank"] == 0].set_index("ACC")
    agg["max1"], agg["max1_h"] = top.p, top.H
    agg["max2"] = long[long["rank"] == 1].set_index("ACC").p
    out = pd.DataFrame(index=e.index)
    for role in ("SRC", "DST"):
        a = agg.reindex(e[role].values).set_axis(e.index)
        out[f"stk_{role.lower()}_mean"] = (a.tot - p) / (a.n - 1).replace(0, np.nan)
        out[f"stk_{role.lower()}_max"] = np.where(a.max1_h == e.index, a.max2, a.max1)
        out[f"stk_{role.lower()}_n"] = a.n
    return out


# ---------- model ----------
PARAMS = dict(objective="binary", metric="None", learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)


def cv_oof(X, y, groups, seed, rounds=2000, folds=5):
    oof, best = np.zeros(len(y)), []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        m = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X.iloc[tr], y.iloc[tr]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])], feval=ap_eval,
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        best.append(m.best_iteration)
    return oof, int(np.mean(best) * 1.1) + 1


def ap_eval(pred, data):
    return "ap", average_precision_score(data.get_label(), pred), True


def fit_predict(X, y, Xt, seed, rounds, n_seeds=3):
    """Average a few seeds: steadier, and continuous scores (ties cost AP)."""
    preds = []
    for s in range(seed, seed + n_seeds):
        m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X, y), max(rounds, 50))
        preds.append(m.predict(Xt))
    return np.mean(preds, axis=0), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="submission.csv"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-stack", action="store_true"); ap.add_argument("--test-labels")
    a = ap.parse_args()
    t0 = time.time()
    e_tr, e_te = build_edges(*load_split(a.train)), build_edges(*load_split(a.test))
    y = load_labels(a.train).reindex(e_tr.index).fillna(0).astype(int)
    print(f"train {len(e_tr)} tx ({y.mean():.2%} suspicious), test {len(e_te)} tx  [{time.time()-t0:.0f}s]")
    X, groups = features(e_tr)
    Xt, _ = features(e_te)
    Xt = Xt[X.columns]
    oof, rounds = cv_oof(X, y, groups, a.seed)
    print(f"stage 1  CV AP {100*average_precision_score(y, oof):.2f}  AUC {roc_auc_score(y, oof):.4f}  rounds {rounds}")
    pred, model = fit_predict(X, y, Xt, a.seed, rounds)
    if not a.no_stack:
        X2 = pd.concat([X, stack_features(e_tr, oof)], axis=1)
        Xt2 = pd.concat([Xt, stack_features(e_te, pred)], axis=1)
        oof2, rounds2 = cv_oof(X2, y, groups, a.seed)
        print(f"stage 2  CV AP {100*average_precision_score(y, oof2):.2f}  AUC {roc_auc_score(y, oof2):.4f}  rounds {rounds2}")
        pred, model = fit_predict(X2, y, Xt2, a.seed, rounds2)
        oof = oof2
    imp = pd.Series(model.feature_importance("gain"), model.feature_name()).sort_values(ascending=False)
    print("top features:", ", ".join(imp.index[:12]))
    sub = pd.DataFrame({"TRANSACTION_HASH": e_te.index, "score": np.clip(pred, 0, 1)})
    assert sub.TRANSACTION_HASH.is_unique and len(sub) == len(e_te)
    sub.to_csv(a.out, index=False)
    cv_ap = 100 * average_precision_score(y, oof)
    msg = f"wrote {a.out}: {len(sub)} rows, {sub.score.nunique()} distinct scores"
    if a.test_labels:  # mock data only
        yt = upper(pd.read_csv(a.test_labels)).set_index("TRANSACTION_HASH").iloc[:, 0].reindex(sub.TRANSACTION_HASH)
        msg += f"  | TEST AP {100*average_precision_score(yt, sub.score):.2f}"
    print(msg, f"[{time.time()-t0:.0f}s]")
    os.makedirs("runs", exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open("runs/log.csv", "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')},{commit},{a.seed},{len(X.columns)},{cv_ap:.2f},{a.out}\n")
    imp.to_csv(f"runs/importance_{os.path.basename(a.out)}.csv")


if __name__ == "__main__":
    main()
