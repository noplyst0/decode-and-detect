"""Decode & Detect: score every test transaction for laundering-style behaviour.

    python src/pipeline.py --train data/train --test data/test --out sub.csv
    python src/pipeline.py ... --fast              # 1 seed, 1 stacking round: quick experiments
    python src/pipeline.py ... --drop cycles,sg    # ablation: leave feature blocks out
    python src/pipeline.py ... --xgb               # also train XGBoost and blend if OOF says so

1. load: SonarX parquet -> one edge per transaction (who paid whom, USD, when, legs, conversion).
2. features, computed inside each split (train and test are separate weeks):
   tx       amount, conversion, legs, fee, flags, structuring (just under a round threshold)
   account  each wallet's week: counts, counterparties, totals, pass-through, lifetime (sender + receiver)
   history  this payment vs the wallet's usual behaviour; first/last use of a wallet
   timing   time since the sender was funded, how fast and how much the receiver forwards, bursts,
            amount similarity in time windows, collector exits (one exit ~ sum of recent deposits)
   chains   how many forwarding hops lead into and out of this payment
   match    follow the money by value (same amount, less a fee, up to 7 days later): chains, loops, echoes
   cycles   does the money come back to the sender, in time order, within 3 days
   sg       scatter-gather: parallel 2-hop routes between the same two wallets
   graph    pair counts, reciprocity, hubs, strongly connected parts
   iforest  unsupervised oddness of each wallet (for patterns never seen in training)
   conv     currency behaviour: issuer swaps (USD->USD at another issuer), symbol changes, 3-leg paths,
            each wallet's conversion habits
   memo     memo text (hex-decoded): pattern frequency, and out-of-fold laundering rate per memo pattern
   probe    failed attempts just before a payment, and failures followed by a success
   subset   split-aware matching: 1-3 recent inflows summing to this payment, 1-3 next outflows summing to it
   dormancy idle time before a payment, and how idle the receiver's other payers were (sleepers waking)
   subset/dormancy are OFF by default (--add); so are --vtags (split busy receivers per destination tag)
   and --local-stack (+-24h neighbour scores). All four were Gemini's review ideas; none helped on either
   practice dataset (2026-10-24), but the real data may differ, so they stay one flag away.
3. LightGBM, 5-fold CV where each laundering case stays in one fold, early stopping on AP.
4. stacking: each payment learns from its neighbours' scores (same wallet, the payment that funded the
   sender, the receiver's next payment, same pair), so a whole flow lights up together. 2 rounds.
5. seed average (+ optional XGBoost), continuous scores in [0, 1], CSV + run log.
"""
import argparse, json, os, re, subprocess, time
import numpy as np, pandas as pd, networkx as nx, lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score as aps, roc_auc_score
from sklearn.model_selection import GroupKFold

XRP_TOKEN = "0x1111222211112222111122221111222211112222"
HOUR, DAY, BIG = 3600, 86400, 10**8
LINK_WINDOW, CYCLE_WINDOW, CYCLE_MAX_LEN, CYCLE_CAP = 48 * HOUR, 72 * HOUR, 6, 2000
T0 = time.time()
VIRTUAL_TAGS = False  # (--vtags) busy receivers are split per destination tag (one exchange address = many customers)


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ---------- 1. load: adjust here if the organisers' README names things differently ----------
def to_unix(x):
    """Seconds since 1970 from datetimes, ISO strings, or numbers in s / ms / Ripple time (s since 2000).
    Unit-safe: pandas 3 keeps ms/us resolution, so astype(int64) is not seconds."""
    x = pd.Series(x)
    if pd.api.types.is_numeric_dtype(x):
        v = x.astype(float)
        med = v.median()
        v = v / 1000 if med > 1e11 else (v + 946684800 if med < 1.2e9 else v)
        return v.round().astype("int64")
    return (pd.to_datetime(x, utc=True) - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)


def upper(df):
    df.columns = [c.upper() for c in df.columns]
    return df


def find_col(df, *keys, exclude=()):
    for c in df.columns:
        if any(k in c for k in keys) and c not in exclude:
            return c
    raise KeyError(f"none of {keys} in {list(df.columns)}")


ALIASES = {"HASH": "TRANSACTION_HASH", "TX_HASH": "TRANSACTION_HASH", "CURRENCY": "SYMBOL", "ISSUER": "TOKEN_ADDRESS",
           "FROM": "FROM_ADDRESS", "TO": "TO_ADDRESS", "SENDER": "ACCOUNT", "CLOSE_TIME": "DATETIME", "TIME": "DATETIME"}


def read_any(d, *words, required=True):
    """Find the file in d whose name contains one of `words` (parquet or csv) and load it with upper-case names."""
    for f in sorted(os.listdir(d)):
        low = f.lower()
        if any(w in low for w in words) and low.endswith((".parquet", ".csv")):
            df = pd.read_parquet(os.path.join(d, f)) if low.endswith(".parquet") else pd.read_csv(os.path.join(d, f))
            df = upper(df)
            return df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})
    if required:
        raise FileNotFoundError(f"no file matching {words} in {d}: {os.listdir(d)}")
    return None


def time_col(df):
    for c in ("DATETIME", "CLOSE_TIME_ISO", "TIMESTAMP", "CLOSE_TIME", "DATE"):
        if c in df.columns:
            return c
    return None


def load_split(d):
    tx, tr = read_any(d, "transaction"), read_any(d, "transfer")
    rates = read_any(d, "rate", "price", "market")
    tx["TS"] = to_unix(tx[time_col(tx)]).to_numpy()
    if time_col(tr):
        tr["TS"] = to_unix(tr[time_col(tr)]).to_numpy()
    else:  # transfers without their own time: borrow the transaction's
        tr["TS"] = tr["TRANSACTION_HASH"].map(tx.set_index("TRANSACTION_HASH")["TS"])
    return tx, tr, rates


def load_labels(d):
    lab = read_any(d, "label")
    try:
        y = find_col(lab, "LABEL", "SUSP", "TARGET", "IS_", "FLAG", "CLASS", exclude=("TRANSACTION_HASH",))
        return lab.set_index("TRANSACTION_HASH")[y].astype(int)
    except KeyError:  # a file that only lists the suspicious hashes
        return pd.Series(1, index=lab["TRANSACTION_HASH"])


def usd_rates(rates):
    """market_rates.csv -> (hour, asset key) -> USD price."""
    tcol = find_col(rates, "HOUR", "TIME", "DATE", "TS")
    pcol = find_col(rates, "USD", "PRICE", "RATE", exclude=(tcol,))
    r = rates.copy()
    r["H"] = to_unix(r[tcol]) // HOUR
    key = [k for k in ("TOKEN_ADDRESS", "SYMBOL") if k in r.columns]
    return r[["H", *key, pcol]].rename(columns={pcol: "USD_PRICE"}).drop_duplicates(["H", *key]), key


def native_value(tr):
    if "VALUE" in tr.columns:
        return tr["VALUE"].astype(float)
    xrp = tr["SYMBOL"].eq("XRP")
    return np.where(xrp, tr["SOURCE_VALUE"].astype(float) / 1e6, tr["SOURCE_VALUE"].astype(float))


def parse_json(s):
    if isinstance(s, str) and s.startswith("{"):
        try:
            return json.loads(s)
        except ValueError:
            return {}
    return {}


def memo_text(s):
    """Plain memo text. Real XRPL memos are JSON with hex-encoded MemoData/MemoType; decode those."""
    if not isinstance(s, str) or not s.strip():
        return None
    if s.lstrip()[:1] not in "[{":
        return s
    try:
        obj = json.loads(s)
    except ValueError:
        return s
    out = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("MemoData", "MemoType", "MemoFormat") and isinstance(v, str):
                    try:
                        out.append(bytes.fromhex(v).decode("utf-8", "ignore"))
                    except ValueError:
                        out.append(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(obj)
    return " ".join(out).strip() or s


def build_edges(tx, tr, rates):
    """One row per transaction: who paid whom, which assets, how much in USD, when."""
    tr = tr.copy()
    tr["V"] = native_value(tr)
    r, key = usd_rates(rates)
    tr["H"] = tr["TS"] // HOUR
    tr = tr.merge(r, on=["H", *key], how="left")
    if tr["USD_PRICE"].isna().any():  # missing hour: fall back to that asset's median price
        med = r.groupby(key)["USD_PRICE"].median().rename("P2")
        tr = tr.join(med, on=key)
        tr["USD_PRICE"] = tr["USD_PRICE"].fillna(tr["P2"])
    tr["USD"] = tr["V"] * tr["USD_PRICE"]
    tr["ASSET"] = tr["TOKEN_ADDRESS"].astype(str) + ":" + tr["SYMBOL"].astype(str)
    tr = tr.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"] if "TRANSFER_INDEX" in tr else ["TRANSACTION_HASH"], kind="stable")
    g = tr.groupby("TRANSACTION_HASH", sort=False)
    first, last = g.first(), g.last()
    legs = pd.DataFrame({"N_LEGS": g.size(), "DST_LEG": last["TO_ADDRESS"], "ASSET_SENT": first["ASSET"],
                         "ASSET_RECV": last["ASSET"], "USD_SENT": first["USD"], "USD_RECV": last["USD"],
                         "V_SENT": first["V"], "N_ASSETS_TX": g["ASSET"].nunique()})
    t = tx.set_index("TRANSACTION_HASH")
    for c, default in (("TRANSACTION_TYPE", "Payment"), ("FEE", np.nan), ("EXTRA_TRANSACTION_FIELDS", None)):
        if c not in t:
            t[c] = default
    if "ACCOUNT" not in t:  # no sender column: take it from the first money leg
        t["ACCOUNT"] = first["FROM_ADDRESS"].reindex(t.index)
    e = t[["ACCOUNT", "TS", "TRANSACTION_TYPE", "FEE"]].rename(columns={"ACCOUNT": "SRC"}).join(legs)
    extra = t["EXTRA_TRANSACTION_FIELDS"].map(parse_json)
    e["DST"] = extra.map(lambda x: x.get("Destination")).fillna(e["DST_LEG"])
    e["DEST_TAG"] = extra.map(lambda x: x.get("DestinationTag"))
    e["HAS_SENDMAX"] = extra.map(lambda x: "SendMax" in x).astype(int)
    e["MEMO_TEXT"] = t["MEMOS"].map(memo_text) if "MEMOS" in t else None
    e["HAS_MEMO"] = e["MEMO_TEXT"].notna().astype(int)
    e["SUCCESS"] = (t["TRANSACTION_RESULT"].eq("tesSUCCESS") if "TRANSACTION_RESULT" in t
                    else t["SUCCESS"].astype(bool) if "SUCCESS" in t else pd.Series(True, index=t.index)).astype(int)
    e["N_LEGS"] = e["N_LEGS"].fillna(0)
    # failed payments have no legs: price what they tried to deliver (Amount) so they are not blank
    miss = e.USD_SENT.isna() & extra.map(lambda x: "Amount" in x)
    if miss.any():
        xrp_tok = tr.loc[tr.SYMBOL.eq("XRP"), "TOKEN_ADDRESS"].mode()
        xrp_tok = xrp_tok.iloc[0] if len(xrp_tok) else XRP_TOKEN
        amt = extra[miss].map(lambda x: x["Amount"])
        intent = pd.DataFrame({
            "V": amt.map(lambda a: float(a) / 1e6 if isinstance(a, (str, int, float)) else float(a.get("value", "nan"))),
            "SYMBOL": amt.map(lambda a: "XRP" if not isinstance(a, dict) else a.get("currency")),
            "TOKEN_ADDRESS": amt.map(lambda a: xrp_tok if not isinstance(a, dict) else a.get("issuer")),
            "H": e.loc[miss, "TS"] // HOUR}, index=amt.index)
        intent = intent.reset_index().merge(r, on=["H", *key], how="left").set_index("TRANSACTION_HASH")
        med = r.groupby(key)["USD_PRICE"].median()
        fill = intent.join(med.rename("P2"), on=key)
        usd = fill.V * fill.USD_PRICE.fillna(fill.P2)
        e.loc[usd.index, "USD_SENT"] = usd
        e.loc[usd.index, "USD_RECV"] = usd
    if VIRTUAL_TAGS:
        indeg = e.groupby("DST").SRC.nunique()
        busy = set(indeg.index[indeg > 50])
        tag = pd.to_numeric(e.DEST_TAG, errors="coerce")
        virt = tag.notna() & e.DST.isin(busy)
        e.loc[virt, "DST"] = e.loc[virt, "DST"] + ":" + tag[virt].astype("int64").astype(str)
    e["IS_CONV"] = (e.ASSET_SENT != e.ASSET_RECV).astype(int) & e.ASSET_SENT.notna()
    e["TR"] = e.TS - e.TS.min()  # seconds since the split's first transaction
    return e.drop(columns="DST_LEG")


# ---------- 2. features ----------
def links(e):
    """For each tx: the sender's latest funding before it, and the receiver's next payment after it."""
    s = e.reset_index()[["TRANSACTION_HASH", "SRC", "DST", "TS", "SUCCESS"]].sort_values("TS")
    ok = e.SUCCESS.eq(1) & e.USD_SENT.notna()
    good = e[ok].reset_index().sort_values("TS")
    inc = good[["TRANSACTION_HASH", "DST", "TS", "USD_RECV"]].rename(
        columns={"TRANSACTION_HASH": "PREV_H", "DST": "ACC", "TS": "PREV_TS", "USD_RECV": "PREV_USD"}).dropna(subset=["ACC"])
    out = good[["TRANSACTION_HASH", "SRC", "TS", "USD_SENT"]].rename(
        columns={"TRANSACTION_HASH": "NEXT_H", "SRC": "ACC", "TS": "NEXT_TS", "USD_SENT": "NEXT_USD"})
    a = pd.merge_asof(s, inc, left_on="TS", right_on="PREV_TS", left_by="SRC", right_by="ACC",
                      direction="backward", allow_exact_matches=False)
    b = pd.merge_asof(s.dropna(subset=["DST"]), out, left_on="TS", right_on="NEXT_TS", left_by="DST", right_by="ACC",
                      direction="forward", allow_exact_matches=False)
    L = a.set_index("TRANSACTION_HASH")[["PREV_H", "PREV_TS", "PREV_USD"]].join(
        b.set_index("TRANSACTION_HASH")[["NEXT_H", "NEXT_TS", "NEXT_USD"]])
    return L.reindex(e.index)


def win_stats(acc, tr, val, w):
    """Per row: count, sum and coefficient of variation of `val` over the same account's rows within +-w s."""
    codes = pd.factorize(acc)[0].astype(np.int64)
    k = codes * BIG + tr.astype(np.int64)
    order = np.argsort(k, kind="stable")
    ks, v = k[order], np.nan_to_num(val[order].astype(float))
    cs, cs2 = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(v * v)]
    lo, hi = np.searchsorted(ks, ks - w, "left"), np.searchsorted(ks, ks + w, "right")
    n, s, s2 = hi - lo, cs[hi] - cs[lo], cs2[hi] - cs2[lo]
    mean = s / n
    cv = np.sqrt(np.maximum(s2 / n - mean ** 2, 0)) / np.where(mean > 0, mean, np.nan)
    out = np.full((len(k), 3), np.nan)
    out[order] = np.c_[n, s, cv]
    out[codes < 0] = np.nan
    return out


def window_sum(ev_acc, ev_tr, ev_val, q_acc, q_tr, w, codes):
    """Sum of event values for account q_acc in (q_tr - w, q_tr]."""
    ek = codes.get_indexer(ev_acc).astype(np.int64) * BIG + ev_tr.astype(np.int64)
    order = np.argsort(ek)
    ek, cs = ek[order], np.r_[0, np.cumsum(np.nan_to_num(ev_val[order]))]
    qc = codes.get_indexer(q_acc).astype(np.int64)
    qk = qc * BIG + np.nan_to_num(q_tr).astype(np.int64)
    res = cs[np.searchsorted(ek, qk, "right")] - cs[np.searchsorted(ek, qk - w, "right")]
    return np.where((qc >= 0) & ~np.isnan(q_tr), res, np.nan)


def tx_features(e):
    f = pd.DataFrame(index=e.index)
    f["log_usd_sent"] = np.log1p(e.USD_SENT)
    f["log_usd_recv"] = np.log1p(e.USD_RECV)
    f["usd_pct"] = e.USD_SENT.rank(pct=True)
    f["recv_over_sent"] = e.USD_RECV / e.USD_SENT.replace(0, np.nan)
    f["n_legs"], f["n_assets_tx"], f["is_conversion"] = e.N_LEGS, e.N_ASSETS_TX, e.IS_CONV
    f["sent_is_xrp"] = e.ASSET_SENT.astype(str).str.endswith(":XRP").astype(int)
    f["recv_is_xrp"] = e.ASSET_RECV.astype(str).str.endswith(":XRP").astype(int)
    f["fee_drops"] = e.FEE * 1e6
    f["hour"] = (e.TS // HOUR) % 24
    f["under_threshold"] = e.USD_SENT / 10 ** np.ceil(np.log10(e.USD_SENT.clip(lower=1e-9)))
    f["round_native"] = (np.round(e.V_SENT, 2) % 1 == 0).astype(int)
    f["round_usd_100"] = (np.round(e.USD_SENT) % 100 == 0).astype(int)
    f["self_pay"] = (e.SRC == e.DST).astype(int)
    f["has_dest_tag"] = e.DEST_TAG.notna().astype(int)
    f["has_sendmax"], f["has_memo"], f["success"] = e.HAS_SENDMAX, e.HAS_MEMO, e.SUCCESS
    f["tx_type"] = e.TRANSACTION_TYPE.astype("category").cat.codes
    return f


def account_table(e):
    g = e[e.SUCCESS.eq(1)]
    out = g.groupby("SRC").agg(n_out=("TS", "size"), usd_out=("USD_SENT", "sum"), cp_out=("DST", "nunique"),
                               out_med=("USD_SENT", "median"), out_mean=("USD_SENT", "mean"), out_std=("USD_SENT", "std"),
                               first_out=("TS", "min"), last_out=("TS", "max"), conv_out=("IS_CONV", "mean"))
    inc = g.dropna(subset=["DST"]).groupby("DST").agg(n_in=("TS", "size"), usd_in=("USD_RECV", "sum"), cp_in=("SRC", "nunique"),
                                                      first_in=("TS", "min"), last_in=("TS", "max"), tags_in=("DEST_TAG", "nunique"))
    fails = e[e.SUCCESS.eq(0)].groupby("SRC").size().rename("n_failed")
    a = out.join(inc, how="outer").join(fails, how="left")
    for c in ("n_out", "n_in", "usd_out", "usd_in", "cp_out", "cp_in", "n_failed", "tags_in"):
        a[c] = a[c].fillna(0)
    lo, hi = a[["usd_in", "usd_out"]].min(axis=1), a[["usd_in", "usd_out"]].max(axis=1)
    a["pass_through"] = (lo / hi.replace(0, np.nan)).fillna(0)
    a["net_flow"] = (a.usd_in - a.usd_out) / (a.usd_in + a.usd_out).replace(0, np.nan)
    a["out_cv"] = a.out_std / a.out_mean
    a["first_seen"] = a[["first_out", "first_in"]].min(axis=1)
    a["span_h"] = (a[["last_out", "last_in"]].max(axis=1) - a.first_seen) / HOUR
    a["in_to_out_h"] = (a.first_out - a.first_in) / HOUR
    a["last_in_to_last_out_h"] = (a.last_out - a.last_in) / HOUR
    a["degree"], a["n_tx"] = a.cp_in + a.cp_out, a.n_in + a.n_out
    a["fail_share"] = a.n_failed / (a.n_out + a.n_failed).replace(0, np.nan)
    a["log_usd_out"], a["log_usd_in"] = np.log1p(a.usd_out), np.log1p(a.usd_in)
    a["cp_ratio"] = a.cp_out / (a.cp_in + 1)
    keep = ["n_out", "n_in", "cp_out", "cp_in", "degree", "n_tx", "pass_through", "net_flow", "out_cv", "span_h",
            "in_to_out_h", "last_in_to_last_out_h", "conv_out", "tags_in", "fail_share", "log_usd_out", "log_usd_in",
            "cp_ratio", "first_seen", "out_med"]
    return a[keep]


def account_features(e, acct):
    s = acct.add_prefix("s_").reindex(e.SRC.values).set_axis(e.index)
    d = acct.add_prefix("d_").reindex(e.DST.values).set_axis(e.index)
    s["s_age_h"], d["d_age_h"] = (e.TS - s.s_first_seen) / HOUR, (e.TS - d.d_first_seen) / HOUR
    return pd.concat([s.drop(columns=["s_first_seen", "s_out_med"]), d.drop(columns=["d_first_seen", "d_out_med"])], axis=1)


def history_features(e, acct, L):
    f = pd.DataFrame(index=e.index)
    f["amt_vs_src_usual"] = np.log((e.USD_SENT + 1e-9) / (acct.out_med.reindex(e.SRC.values).to_numpy() + 1e-9))
    f["src_out_pos"] = e.groupby("SRC").TS.rank(pct=True)
    f["src_first_out"] = (e.groupby("SRC").TS.rank(method="min") == 1).astype(int)
    f["src_last_out"] = (e.groupby("SRC").TS.rank(method="max", ascending=False) == 1).astype(int)
    f["dst_first_in"] = (e.groupby("DST").TS.rank(method="min") == 1).astype(int)
    f["dst_last_in"] = (e.groupby("DST").TS.rank(method="max", ascending=False) == 1).astype(int)
    f["src_was_funded"] = L.PREV_H.notna().astype(int)
    f["dst_pays_on"] = L.NEXT_H.notna().astype(int)
    return f


def timing_features(e, L):
    f = pd.DataFrame(index=e.index)
    f["src_since_funded_h"] = (e.TS - L.PREV_TS) / HOUR
    f["src_fwd_ratio"] = e.USD_SENT / L.PREV_USD
    f["dst_next_out_h"] = (L.NEXT_TS - e.TS) / HOUR
    f["dst_fwd_ratio"] = L.NEXT_USD / e.USD_RECV
    f["fwd_both"] = (f.src_fwd_ratio.between(0.9, 1.01) & f.dst_fwd_ratio.between(0.9, 1.01)).astype(int)
    ok = e.SUCCESS.eq(1).to_numpy()
    for w, tag in ((300, "5m"), (HOUR, "1h"), (6 * HOUR, "6h"), (DAY, "24h")):
        so = win_stats(e.SRC.where(ok).to_numpy(), e.TR.to_numpy(), e.USD_SENT.to_numpy(), w)
        di = win_stats(e.DST.where(ok).to_numpy(), e.TR.to_numpy(), e.USD_RECV.to_numpy(), w)
        f[f"src_out_n_{tag}"], f[f"src_out_cv_{tag}"] = so[:, 0], so[:, 2]
        f[f"dst_in_n_{tag}"], f[f"dst_in_cv_{tag}"] = di[:, 0], di[:, 2]
        f[f"src_out_share_{tag}"] = e.USD_SENT / so[:, 1]
    codes = pd.Index(pd.unique(pd.concat([e.SRC, e.DST.dropna()])))
    g = e[ok & e.DST.notna().to_numpy()]
    # collector exit: does the receiver's next payment roughly equal everything it got in the 24h before?
    inflow_before_exit = window_sum(g.DST.to_numpy(), g.TR.to_numpy(), g.USD_RECV.to_numpy(),
                                    e.DST.to_numpy(), (L.NEXT_TS - e.TS.min()).to_numpy(dtype=float), DAY, codes)
    f["dst_exit_vs_inflow24"] = L.NEXT_USD.to_numpy() / np.where(inflow_before_exit > 0, inflow_before_exit, np.nan)
    inflow_src = window_sum(g.DST.to_numpy(), g.TR.to_numpy(), g.USD_RECV.to_numpy(),
                            e.SRC.to_numpy(), e.TR.to_numpy(dtype=float) - 1, DAY, codes)
    f["src_out_vs_inflow24"] = e.USD_SENT.to_numpy() / np.where(inflow_src > 0, inflow_src, np.nan)
    return f


def chain_features(e, L, lo=0.05, hi=1.05):
    """Forwarding hops leading into (back) and out of (fwd) each payment, within 48h, most value carried on."""
    pos = pd.Series(np.arange(len(e)), index=e.index)
    nxt, prv = pos.reindex(L.NEXT_H).to_numpy(), pos.reindex(L.PREV_H).to_numpy()
    ok_n = ((L.NEXT_TS - e.TS) <= LINK_WINDOW) & (L.NEXT_USD / e.USD_RECV).between(lo, hi)
    ok_p = ((e.TS - L.PREV_TS) <= LINK_WINDOW) & (e.USD_SENT / L.PREV_USD).between(lo, hi)
    ok_n, ok_p = ok_n.to_numpy() & ~np.isnan(nxt), ok_p.to_numpy() & ~np.isnan(prv)
    order = np.argsort(e.TS.to_numpy(), kind="stable")
    fwd, back = np.zeros(len(e)), np.zeros(len(e))
    for i in order[::-1]:
        if ok_n[i]:
            fwd[i] = 1 + fwd[int(nxt[i])]
    for i in order:
        if ok_p[i]:
            back[i] = 1 + back[int(prv[i])]
    return pd.DataFrame({"chain_fwd": fwd, "chain_back": back, "chain_len": fwd + back}, index=e.index)


def matched_links(e, hubs, window=7 * DAY, lo=0.8, hi=1.02):
    """Follow the money by value: the receiver's first later payment carrying 80-102% of what it received,
    up to 7 days later. Survives slow hops and mules who make ordinary payments in between."""
    ok = e.SUCCESS.eq(1) & e.USD_SENT.notna() & e.DST.notna() & (e.SRC != e.DST)
    inc = e[ok & ~e.DST.isin(hubs)].reset_index()[["TRANSACTION_HASH", "DST", "TS", "USD_RECV"]].rename(columns={"DST": "ACC"})
    out = e[ok].reset_index()[["TRANSACTION_HASH", "SRC", "TS", "USD_SENT"]].rename(
        columns={"SRC": "ACC", "TRANSACTION_HASH": "H2", "TS": "TS2"})
    m = inc.merge(out[out.ACC.isin(set(inc.ACC))], on="ACC")
    r = m.USD_SENT / m.USD_RECV
    m = m[(m.TS2 > m.TS) & (m.TS2 - m.TS <= window) & (r >= lo) & (r <= hi)]
    nxt = m.sort_values("TS2").drop_duplicates("TRANSACTION_HASH").set_index("TRANSACTION_HASH")
    prv = m.sort_values("TS").drop_duplicates("H2", keep="last").set_index("H2")
    M = pd.DataFrame(index=e.index)
    M["MNEXT_H"], M["MNEXT_TS"] = nxt.H2, nxt.TS2
    M["MPREV_H"], M["MPREV_TS"] = prv.TRANSACTION_HASH, prv.TS
    return M


def match_features(e, M):
    pos = pd.Series(np.arange(len(e)), index=e.index)
    nxt, prv = pos.reindex(M.MNEXT_H).to_numpy(), pos.reindex(M.MPREV_H).to_numpy()
    order = np.argsort(e.TS.to_numpy(), kind="stable")
    fwd, back = np.zeros(len(e)), np.zeros(len(e))
    for i in order[::-1]:
        if not np.isnan(nxt[i]):
            fwd[i] = 1 + fwd[int(nxt[i])]
    for i in order:
        if not np.isnan(prv[i]):
            back[i] = 1 + back[int(prv[i])]
    # value-preserving loop: follow matched forwards up to 8 hops; do we come back to the sender?
    src, dst = e.SRC.to_numpy(), e.DST.to_numpy()
    loop = np.zeros(len(e))
    for i in range(len(e)):
        j, k = nxt[i], 0
        while not np.isnan(j) and k < 8:
            j = int(j); k += 1
            if dst[j] == src[i]:
                loop[i] = k + 1; break
            j = nxt[j]
    f = pd.DataFrame(index=e.index)
    f["m_next_h"], f["m_prev_h"] = (M.MNEXT_TS - e.TS) / HOUR, (e.TS - M.MPREV_TS) / HOUR
    f["m_chain_fwd"], f["m_chain_back"], f["m_chain_len"] = fwd, back, fwd + back
    f["m_loop_len"] = np.where(loop > 0, loop, np.nan)
    # amount echo: how many other payments this week carry (almost) exactly this USD amount
    v = np.log(e.USD_SENT.where(e.USD_SENT > 1)).to_numpy()
    vs = np.sort(v[~np.isnan(v)])
    for tol, tag in ((0.002, "02"), (0.01, "1")):
        f[f"echo_{tag}"] = np.where(np.isnan(v), np.nan, np.searchsorted(vs, v + tol, "right") - np.searchsorted(vs, v - tol, "left") - 1)
    return f


def conv_features(e):
    """Currency behaviour: issuer swaps (USD from one issuer to USD from another), symbol changes, 3-leg
    paths, and each wallet's habits across all its payments."""
    f = pd.DataFrame(index=e.index)
    sym = lambda a: a.astype(str).str.split(":").str[-1]
    ss, rs, has = sym(e.ASSET_SENT), sym(e.ASSET_RECV), e.ASSET_SENT.notna()
    f["issuer_swap"] = ((ss == rs) & (e.ASSET_SENT != e.ASSET_RECV) & has).astype(int)
    f["symbol_change"] = ((ss != rs) & has).astype(int)
    f["three_legs"] = (e.N_LEGS >= 3).astype(int)
    for role, r in (("SRC", "s"), ("DST", "d")):
        g = pd.DataFrame({"a": e[role], "conv": e.IS_CONV, "swap": f.issuer_swap, "l3": f.three_legs}).groupby("a")
        f[f"{r}_conv_share"] = g.conv.transform("mean")
        f[f"{r}_swap_share"] = g.swap.transform("mean")
        f[f"{r}_3leg_share"] = g.l3.transform("mean")
    assets = pd.concat([pd.DataFrame({"a": e.SRC, "x": e.ASSET_SENT}), pd.DataFrame({"a": e.DST, "x": e.ASSET_RECV})]).dropna()
    na = assets.groupby("a").x.nunique()
    f["s_n_assets"], f["d_n_assets"] = e.SRC.map(na), e.DST.map(na)
    return f


def memo_keys(e):
    txt = e.MEMO_TEXT.fillna("").astype(str)
    prefix = txt.str.extract(r"^\s*([A-Za-z]+)", expand=False).str.upper()
    prefix = prefix.where(prefix.notna(), np.where(txt.eq(""), "<none>", "<other>"))
    pattern = txt.str.replace(r"[A-Za-z]+", "A", regex=True).str.replace(r"[0-9]+", "9", regex=True)
    return prefix, pattern.where(txt.ne(""), "<none>")


def memo_features(e):
    prefix, pattern = memo_keys(e)
    f = pd.DataFrame(index=e.index)
    f["memo_len"] = e.MEMO_TEXT.fillna("").astype(str).str.len()
    f["memo_prefix_freq"] = prefix.map(prefix.value_counts(normalize=True))
    f["memo_pattern_freq"] = pattern.map(pattern.value_counts(normalize=True))
    f["memo_pattern_senders"] = pattern.map(pd.DataFrame({"p": pattern, "s": e.SRC}).groupby("p").s.nunique())
    f["src_memo_share"] = e.HAS_MEMO.groupby(e.SRC).transform("mean")
    return f


def memo_target_encoding(e_tr, e_te, y, groups, k=20):
    """Laundering rate per memo prefix and pattern, learned from training labels. Out-of-fold on train
    (same folds as the model), full-train on test, smoothed towards the base rate."""
    (ptr, qtr), (pte, qte) = memo_keys(e_tr), memo_keys(e_te)
    prior = y.mean()

    def enc(fit, yf, apply):
        st = pd.DataFrame({"k": fit.to_numpy(), "y": yf.to_numpy()}).groupby("k").y.agg(["sum", "count"])
        return apply.map((st["sum"] + k * prior) / (st["count"] + k)).fillna(prior).to_numpy()
    out_tr, out_te = pd.DataFrame(index=e_tr.index), pd.DataFrame(index=e_te.index)
    for name, ktr, kte in (("memo_prefix_te", ptr, pte), ("memo_pattern_te", qtr, qte)):
        col = np.zeros(len(e_tr))
        for a, b in GroupKFold(n_splits=5).split(e_tr, y, groups):
            col[b] = enc(ktr.iloc[a], y.iloc[a], ktr.iloc[b])
        out_tr[name], out_te[name] = col, enc(ktr, y, kte)
    return out_tr, out_te


def probe_features(e):
    """Failed attempts: a sender that failed in the hour before, and failures followed by a success."""
    f = pd.DataFrame(index=e.index)
    failed = e.SUCCESS.eq(0).to_numpy()
    codes = pd.Index(pd.unique(e.SRC))
    fe, ok = e[failed], e[~failed]
    f["src_failed_1h_before"] = window_sum(fe.SRC.to_numpy(), fe.TR.to_numpy(), np.ones(len(fe)), e.SRC.to_numpy(),
                                           e.TR.to_numpy(dtype=float) - 1, HOUR, codes)
    after = window_sum(ok.SRC.to_numpy(), ok.TR.to_numpy(), np.ones(len(ok)), e.SRC.to_numpy(),
                       e.TR.to_numpy(dtype=float) + HOUR, HOUR, codes)
    f["failed_then_success_1h"] = np.where(failed, after, np.nan)
    return f


def subset_features(e, hubs, window=48 * HOUR, k=6):
    """Split-aware matching: do 1-3 of the sender's recent inflows add up to this payment, and do 1-3 of
    the receiver's next outflows add up to what it received? Catches peels, splits and merges."""
    from itertools import combinations
    combos = {n: [np.array(c) for r in range(1, min(3, n) + 1) for c in combinations(range(n), r)] for n in range(1, k + 1)}
    ok = e.SUCCESS.eq(1) & e.USD_SENT.notna()
    g = e[ok & e.DST.notna()]
    inflow = {a: (x.TS.to_numpy(), x.USD_RECV.to_numpy()) for a, x in g.sort_values("TS").groupby("DST", sort=False)}
    outflow = {a: (x.TS.to_numpy(), x.USD_SENT.to_numpy()) for a, x in e[ok].sort_values("TS").groupby("SRC", sort=False)}

    def best(ts, vs, lo, hi, target, last):
        i0, i1 = np.searchsorted(ts, lo, "left"), np.searchsorted(ts, hi, "right")
        sel = vs[max(i0, i1 - k):i1] if last else vs[i0:min(i1, i0 + k)]
        if len(sel) == 0 or not target > 0:
            return np.nan, np.nan
        errs = [(abs(sel[c].sum() - target) / target, len(c)) for c in combos[len(sel)]]
        return min(errs)
    n = len(e)
    in_err, in_k, out_err, out_k = (np.full(n, np.nan) for _ in range(4))
    for i, (sr, ds, t, vs, vr, good) in enumerate(zip(e.SRC, e.DST, e.TS, e.USD_SENT, e.USD_RECV, ok)):
        if not good:
            continue
        if sr in inflow and sr not in hubs:
            ts, v = inflow[sr]
            in_err[i], in_k[i] = best(ts, v, t - window, t - 1, vs, True)
        if ds in outflow and ds not in hubs:
            ts, v = outflow[ds]
            out_err[i], out_k[i] = best(ts, v, t + 1, t + window, vr, False)
    return pd.DataFrame({"in_subset_err": in_err, "in_subset_k": in_k, "out_subset_err": out_err, "out_subset_k": out_k}, index=e.index)


def dormancy_features(e):
    """Sleepers waking up: how long the sender was idle before this payment, and how idle the other
    wallets paying the same receiver within +-2h were."""
    act = pd.concat([pd.DataFrame({"ACC": e.SRC, "T": e.TS}), pd.DataFrame({"ACC": e.DST, "T": e.TS})]).dropna()
    act = act.drop_duplicates().sort_values("T")
    q = e.reset_index()[["TRANSACTION_HASH", "SRC", "TS"]].sort_values("TS")
    m = pd.merge_asof(q, act.rename(columns={"T": "PREV_T"}), left_on="TS", right_on="PREV_T", left_by="SRC", right_by="ACC",
                      direction="backward", allow_exact_matches=False).set_index("TRANSACTION_HASH")
    f = pd.DataFrame(index=e.index)
    f["src_idle_h"] = ((m.TS - m.PREV_T) / HOUR).reindex(e.index)
    li = np.log1p(f.src_idle_h.fillna(24 * 7)).to_numpy()
    w = win_stats(e.DST.to_numpy(), e.TR.to_numpy(), li, 2 * HOUR)
    f["dst_cohort_idle_2h"] = w[:, 1] / w[:, 0]
    f["dst_cohort_n_2h"] = w[:, 0]
    return f


def hubs_of(acct, q=0.995):
    return set(acct.index[acct.degree > max(acct.degree.quantile(q), 50)])


def core_edges(e, hubs):
    ok = e.SUCCESS.eq(1) & e.DST.notna() & e.USD_SENT.notna() & (e.SRC != e.DST) & ~e.SRC.isin(hubs) & ~e.DST.isin(hubs)
    return e[ok]


def cycle_features(e, hubs):
    """Shortest time-ordered path from the receiver back to the sender within 72h (non-hub wallets)."""
    c = core_edges(e, hubs).sort_values("TS")
    adj = {k: (g.TS.to_numpy(), g.DST.to_numpy(), g.USD_RECV.to_numpy()) for k, g in c.groupby("SRC", sort=False)}
    clen, cret = pd.Series(np.nan, index=e.index), pd.Series(np.nan, index=e.index)
    for h, s, d, t, v in zip(c.index, c.SRC, c.DST, c.TS, c.USD_SENT):
        if d not in adj or s not in adj:
            continue
        best, frontier, seen, found = {d: t}, [(d, t, 1)], 0, None
        while frontier and found is None and seen <= CYCLE_CAP:
            nf = []
            for u, tu, k in frontier:
                if u not in adj:
                    continue
                ts, ds, vs = adj[u]
                for j in range(np.searchsorted(ts, tu, "right"), np.searchsorted(ts, t + CYCLE_WINDOW, "right")):
                    seen += 1
                    if ds[j] == s:
                        found = (k + 1, vs[j]); break
                    if k + 1 < CYCLE_MAX_LEN and ts[j] < best.get(ds[j], np.inf):
                        best[ds[j]] = ts[j]; nf.append((ds[j], ts[j], k + 1))
                if found or seen > CYCLE_CAP:
                    break
            frontier = nf
        if found:
            clen[h], cret[h] = found[0], found[1] / v
    return pd.DataFrame({"cycle_len": clen, "cycle_return": cret, "in_cycle": clen.notna().astype(int)})


def sg_features(e, hubs):
    """Scatter-gather: for s->m then m->d within 48h, how many distinct middlemen m link s and d?"""
    c = core_edges(e, hubs)[["SRC", "DST", "TS", "USD_SENT"]].reset_index()
    two = c.merge(c, left_on="DST", right_on="SRC", suffixes=("", "_2"))
    two = two[(two.TS_2 > two.TS) & (two.TS_2 - two.TS <= LINK_WINDOW) & (two.SRC != two.DST_2)]
    f = pd.DataFrame(index=e.index)
    if len(two) == 0:
        return f.assign(sg_first=np.nan, sg_second=np.nan, sg_value_kept=np.nan)
    k = two.groupby(["SRC", "DST_2"]).agg(k=("DST", "nunique"), v_in=("USD_SENT", "sum"), v_out=("USD_SENT_2", "sum"))
    two = two.join(k, on=["SRC", "DST_2"])
    f["sg_first"] = two.groupby("TRANSACTION_HASH").k.max()
    f["sg_second"] = two.groupby("TRANSACTION_HASH_2").k.max()
    two["kept"] = two.v_out / two.v_in
    f["sg_value_kept"] = two[two.k > 1].groupby("TRANSACTION_HASH").kept.max()
    return f


def graph_features(e, acct, hubs):
    f = pd.DataFrame(index=e.index)
    ed = e.dropna(subset=["DST"])
    f["pair_n"] = ed.groupby(["SRC", "DST"]).TS.transform("size")
    rev = set(zip(ed.DST, ed.SRC))
    f["reciprocal"] = [int((d, s) in rev) for s, d in zip(e.SRC, e.DST)]
    f["src_is_hub"], f["dst_is_hub"] = e.SRC.isin(hubs).astype(int), e.DST.isin(hubs).astype(int)
    c = core_edges(e, hubs)
    G = nx.DiGraph(); G.add_edges_from(zip(c.SRC, c.DST))
    scc = pd.Series({n: i for i, comp in enumerate(nx.strongly_connected_components(G)) for n in comp}, dtype=float)
    wcc = pd.Series({n: i for i, comp in enumerate(nx.weakly_connected_components(G)) for n in comp}, dtype=float)
    scc_size, wcc_size = scc.map(scc.value_counts()), wcc.map(wcc.value_counts())
    f["same_scc"] = (e.SRC.map(scc) == e.DST.map(scc)).astype(int)
    f["scc_size"], f["wcc_size"] = e.SRC.map(scc_size), e.SRC.map(wcc_size)
    f["dst_wcc_size"] = e.DST.map(wcc_size)
    return f


def iforest_features(e, acct):
    cols = ["n_out", "n_in", "cp_out", "cp_in", "pass_through", "span_h", "log_usd_in", "log_usd_out", "out_cv", "conv_out"]
    X = acct[cols].replace([np.inf, -np.inf], np.nan).fillna(-1)
    score = pd.Series(-IsolationForest(n_estimators=200, random_state=0).fit(X).score_samples(X), index=acct.index)
    return pd.DataFrame({"s_iforest": e.SRC.map(score), "d_iforest": e.DST.map(score)}, index=e.index)


def gemini_block(e):
    """Features designed by Gemini (Argon 4) from our aggregate findings only; see src/gemini_feats.py."""
    from gemini_feats import gemini_features
    return gemini_features(e)


def relay_block(e):
    """Scripted-collector features (regular hold before payout), designed by Gemini; see src/gemini_relay.py."""
    from gemini_relay import relay_features
    return relay_features(e)


def burst_features(e, windows=(120, 300, 900, 2700)):
    """Rebuild bursts: link payments that touch the same wallet within w seconds (single linkage), then describe
    the group each payment lands in. Planted flows finish their whole shape in minutes; look-alikes take hours."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(e); rows = np.arange(n)
    ev = pd.DataFrame({"w": np.r_[e.SRC.astype(str).to_numpy(), e.DST.astype(str).to_numpy()],
                       "t": np.r_[e.TS.to_numpy(float), e.TS.to_numpy(float)], "r": np.r_[rows, rows]})
    ev = ev[ev.w != "nan"].sort_values(["w", "t"])
    same = ev.w.to_numpy()[1:] == ev.w.to_numpy()[:-1]
    gap = np.diff(ev.t.to_numpy())
    a, b = ev.r.to_numpy()[:-1], ev.r.to_numpy()[1:]
    f = pd.DataFrame(index=e.index)
    usd = e.USD_SENT.to_numpy(float)
    for w in windows:
        k = same & (gap <= w)
        g = coo_matrix((np.ones(k.sum()), (a[k], b[k])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        c = pd.DataFrame({"c": lab, "t": e.TS.to_numpy(float), "s": e.SRC.astype(str).to_numpy(),
                          "d": e.DST.astype(str).to_numpy(), "u": usd})
        grp = c.groupby("c")
        size = grp.c.transform("size").to_numpy()
        dur = (grp.t.transform("max") - grp.t.transform("min")).to_numpy()
        nw = grp.s.transform("nunique").to_numpy() + grp.d.transform("nunique").to_numpy()
        m = w // 60
        f[f"bu{m}_n"] = size
        f[f"bu{m}_dur"] = np.where(size > 1, dur, np.nan)
        f[f"bu{m}_rate"] = np.where(size > 1, size / (dur + 60), np.nan)
        f[f"bu{m}_wallets_per_tx"] = nw / size
        f[f"bu{m}_usd_cv"] = (grp.u.transform("std") / grp.u.transform("mean")).to_numpy()
        f[f"bu{m}_pos"] = (grp.t.rank(method="first").to_numpy() - 1) / np.maximum(size - 1, 1)
        if w in (300, 900) and os.environ.get("BURST_SHAPE", "1") == "1":
            c["cv"] = e.IS_CONV.to_numpy(float); c["a"] = e.ASSET_SENT.astype(str).to_numpy()
            g2 = c.groupby("c")
            f[f"bu{m}_conv"] = g2.cv.transform("mean").to_numpy()
            f[f"bu{m}_nsym"] = g2.a.transform("nunique").to_numpy()
            f[f"bu{m}_usd_med"] = np.log1p(g2.u.transform("median").to_numpy())
            f[f"bu{m}_nsrc"] = g2.s.transform("nunique").to_numpy()
            f[f"bu{m}_ndst"] = g2.d.transform("nunique").to_numpy()
            f[f"bu{m}_maxout"] = c.groupby(["c", "s"]).s.transform("size").groupby(c.c).transform("max").to_numpy()
            f[f"bu{m}_maxin"] = c.groupby(["c", "d"]).d.transform("size").groupby(c.c).transform("max").to_numpy()
            f[f"bu{m}_ret"] = c.groupby("c").apply(lambda g: len(set(g.s) & set(g.d)), include_groups=False).reindex(c.c).to_numpy()
        if w in (300, 900, 2700) and os.environ.get("BURST_KEPT", "0") == "1":
            uin = c.groupby(["c", "d"]).u.sum(); uout = c.groupby(["c", "s"]).u.sum()
            uin.index.names = uout.index.names = ["c", "w"]
            both = pd.concat([uin.rename("i"), uout.rename("o")], axis=1).dropna()
            mid = both.groupby(level="c").agg(i=("i", "sum"), o=("o", "sum"), k=("i", "size"))
            f[f"bu{m}_kept"] = c.c.map(mid.o / mid.i).to_numpy()
            f[f"bu{m}_mids"] = c.c.map(mid.k).fillna(0).to_numpy()
            dkept = (uout / uin.reindex(uout.index)).rename("dk")
            f[f"bu{m}_dst_kept"] = pd.MultiIndex.from_arrays([c.c, c.d]).map(dkept.to_dict().get).astype(float)
    return f


BLOCKS = ["tx", "account", "history", "timing", "chains", "match", "subset", "cycles", "sg", "graph", "iforest", "conv",
          "memo", "probe", "dormancy", "burst", "gemini", "relay", "repeat"]


def features(e, drop=()):
    acct, L = account_table(e), links(e)
    hubs = hubs_of(acct)
    M = matched_links(e, hubs)
    make = {"match": lambda: match_features(e, M), "conv": lambda: conv_features(e), "memo": lambda: memo_features(e),
            "probe": lambda: probe_features(e), "subset": lambda: subset_features(e, hubs),
            "dormancy": lambda: dormancy_features(e), "tx": lambda: tx_features(e), "account": lambda: account_features(e, acct),
            "history": lambda: history_features(e, acct, L), "timing": lambda: timing_features(e, L),
            "chains": lambda: chain_features(e, L), "cycles": lambda: cycle_features(e, hubs),
            "sg": lambda: sg_features(e, hubs), "graph": lambda: graph_features(e, acct, hubs),
            "iforest": lambda: iforest_features(e, acct), "burst": lambda: burst_features(e), "gemini": lambda: gemini_block(e), "relay": lambda: relay_block(e), "repeat": lambda: __import__("gemini_repeat").repeat_features(e)}
    parts = []
    for b in BLOCKS:
        if b not in drop:
            t = time.time(); parts.append(make[b]()); log(f"  {b:8s} {parts[-1].shape[1]:3d} cols  {time.time() - t:4.1f}s")
    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).astype(float)
    return X, {"L": L, "M": M, "hubs": hubs}


LOCAL_STACK = False  # (--local-stack) add +-24h neighbour scores
OPTIONAL = {"subset", "dormancy", "relay", "repeat"}  # (--add) built, but did not help on either practice set


def stack_features(e, L, M, score):
    """Neighbour suspicion from the previous round's scores, always leaving the payment's own score out."""
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
    out["stk_prev_in"] = p.reindex(L.PREV_H).to_numpy()
    out["stk_next_out"] = p.reindex(L.NEXT_H).to_numpy()
    out["stk_mprev"] = p.reindex(M.MPREV_H).to_numpy()
    out["stk_mnext"] = p.reindex(M.MNEXT_H).to_numpy()
    # time-local: the same wallet's other payments within +-24h only (warm mules live normal lives too)
    if LOCAL_STACK:
        lng = pd.concat([pd.DataFrame({"ACC": e.SRC, "TR": e.TR, "p": p, "role": 0}),
                         pd.DataFrame({"ACC": e.DST, "TR": e.TR, "p": p, "role": 1})])
        w = win_stats(lng.ACC.to_numpy(), lng.TR.to_numpy(), lng.p.to_numpy(), DAY)
        loc = (w[:, 1] - lng.p.to_numpy()) / np.where(w[:, 0] > 1, w[:, 0] - 1, np.nan)
        out["stk_src_local24"], out["stk_dst_local24"] = loc[:len(e)], loc[len(e):]
    pair = pd.DataFrame({"SRC": e.SRC, "DST": e.DST, "p": p})
    g = pair.groupby(["SRC", "DST"]).p
    out["stk_pair_mean"] = (g.transform("sum") - p) / (g.transform("size") - 1).replace(0, np.nan)
    return out


# ---------- 3. model ----------
PARAMS = dict(objective="binary", metric="None", learning_rate=0.02, num_leaves=15, min_child_samples=60,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=8)
PARAMS.update(json.loads(os.environ.get("LGB_PARAMS", "{}")))  # quick experiments without editing the file


def label_scenarios(d):
    """Organiser scenario ids (train labels only), or None. Used ONLY to build CV folds, never as a feature."""
    lab = read_any(d, "label")
    col = next((c for c in lab.columns if "SCENARIO" in c.upper()), None)
    return None if col is None else lab.set_index("TRANSACTION_HASH")[col]


def label_families(d):
    """Organiser scenario family per train tx (incl. innocent look-alikes), used only as a helper TARGET."""
    lab = read_any(d, "label")
    col = next((c for c in lab.columns if c.upper() == "FAMILY"), None)
    return None if col is None else lab.set_index("TRANSACTION_HASH")[col].fillna("none").astype(str)


def family_aux(F, Ft, fam, groups, seed, rounds=150):
    """Predict which scenario family each tx belongs to (out-of-fold on train, full fit for test). The main model
    then knows the shape context, since the suspicious-vs-look-alike rules flip by family."""
    codes, names = pd.factorize(fam)
    p = dict(objective="multiclass", num_class=len(names), learning_rate=0.08, num_leaves=15, min_child_samples=40,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=seed)
    oof = np.zeros((len(F), len(names)))
    for tr, va in GroupKFold(n_splits=5).split(F, codes, groups):
        m = lgb.train(p, lgb.Dataset(F.iloc[tr], codes[tr]), rounds)
        oof[va] = m.predict(F.iloc[va])
    pt = lgb.train(p, lgb.Dataset(F, codes), rounds).predict(Ft)
    cols = [f"fam_p_{n}" for n in names]
    return pd.DataFrame(oof, F.index, cols), pd.DataFrame(pt, Ft.index, cols)


def scenario_groups(e, y, hubs, scen=None):
    """CV groups: each connected laundering case (hubs excluded) is one group; normal payments group by sender.
    If the labels carry organiser scenario ids, those define the groups instead (both suspicious and look-alike)."""
    if scen is not None:
        sc = scen.reindex(e.index)
        key = pd.Series(np.where(sc.notna(), "S" + sc.astype(str), "N" + e.SRC.astype(str)), index=e.index)
        pos_sizes = key[y.eq(1)].value_counts()
        log(f"CV groups (scenario ids): {key.nunique()} ({len(pos_sizes)} laundering), largest {pos_sizes.max()}")
        return pd.factorize(key)[0]
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    pos = e[y.eq(1).to_numpy()]
    for s, d in zip(pos.SRC, pos.DST):
        if s not in hubs and d not in hubs and isinstance(d, str):
            parent[find(s)] = find(d)
        elif s not in hubs:
            find(s)
    key = pd.Series(np.where(y.eq(1), ["P" + find(s if s not in hubs else d) for s, d in zip(e.SRC, e.DST)],
                             "N" + e.SRC.astype(str)), index=e.index)
    # Cases that share mules chain into one giant group, which leaves a fold with most of the positives.
    # Split any group bigger than a fortieth of the data by day, so no single fold is starved.
    size = key.map(key.value_counts())
    day = ((e.TS - e.TS.min()) // DAY).astype(str)
    key = key.where(size <= len(e) / 40, key + "|" + day)
    pos_sizes = key[y.eq(1)].value_counts()
    log(f"CV groups: {key.nunique()} ({len(pos_sizes)} laundering), largest laundering group {pos_sizes.max()} of {int(y.sum())}")
    return pd.factorize(key)[0]


def ap_eval(pred, data):
    return "ap", aps(data.get_label(), pred), True


def cv_oof(X, y, groups, seed, rounds=3000, folds=5):
    oof, best = np.zeros(len(y)), []
    for tr, va in GroupKFold(n_splits=folds).split(X, y, groups):
        stop = os.environ.get("STOP_METRIC", "logloss")  # "logloss": smoother early stopping than AP on ~250 positives
        prm = {**PARAMS, "seed": seed, **({"metric": "binary_logloss"} if stop == "logloss" else {})}
        m = lgb.train(prm, lgb.Dataset(X.iloc[tr], y.iloc[tr]), rounds,
                      valid_sets=[lgb.Dataset(X.iloc[va], y.iloc[va])], feval=None if stop == "logloss" else ap_eval,
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        best.append(m.best_iteration)
    return oof, int(np.mean(best) * 1.1) + 1


def fit_predict(X, y, Xt, seed, rounds, n_seeds):
    preds = []
    for s in range(seed, seed + n_seeds):
        m = lgb.train({**PARAMS, "seed": s}, lgb.Dataset(X, y), max(rounds, 50))
        preds.append(m.predict(Xt))
    return np.mean(preds, axis=0), m


def xgb_oof_pred(X, y, Xt, groups, seed):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="aucpr", eta=0.05, max_depth=6, subsample=0.8,
             colsample_bytree=0.7, min_child_weight=3, nthread=8, seed=seed)
    oof, best = np.zeros(len(y)), []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        dtr, dva = xgb.DMatrix(X.iloc[tr], y.iloc[tr]), xgb.DMatrix(X.iloc[va], y.iloc[va])
        m = xgb.train(p, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=100, verbose_eval=False)
        oof[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1)); best.append(m.best_iteration)
    m = xgb.train(p, xgb.DMatrix(X, y), int(np.mean(best) * 1.1) + 1)
    return oof, m.predict(xgb.DMatrix(Xt))


def rank01(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="submission.csv"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stack", type=int, default=2, help="stacking rounds (0 = none)")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--xgb", action="store_true"); ap.add_argument("--drop", default="")
    ap.add_argument("--stack-dropout", type=float, default=0.0, help="share of stacking values hidden while training")
    ap.add_argument("--vtags", action="store_true", help="split busy receivers per destination tag")
    ap.add_argument("--local-stack", action="store_true", help="add +-24h neighbour scores")
    ap.add_argument("--add", default="", help="optional blocks to switch on: subset,dormancy")
    ap.add_argument("--test-labels", help="mock data only: score the output"); ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.stack, a.seeds = min(a.stack, 1), 1
    drop = set(filter(None, a.drop.split(",")))
    global VIRTUAL_TAGS, LOCAL_STACK
    VIRTUAL_TAGS, LOCAL_STACK = a.vtags, a.local_stack
    drop |= OPTIONAL - set(filter(None, a.add.split(",")))
    e_tr, e_te = build_edges(*load_split(a.train)), build_edges(*load_split(a.test))
    y = load_labels(a.train).reindex(e_tr.index).fillna(0).astype(int)
    log(f"train {len(e_tr)} tx ({y.mean():.2%} suspicious, {y.sum()}), test {len(e_te)} tx")
    log("features train:"); F, ctx = features(e_tr, drop)
    log("features test:"); Ft, ctx_t = features(e_te, drop)
    Ft = Ft.reindex(columns=F.columns)
    groups = scenario_groups(e_tr, y, ctx["hubs"], label_scenarios(a.train))
    if "memo" not in drop and e_tr.HAS_MEMO.any():
        te_tr, te_te = memo_target_encoding(e_tr, e_te, y, groups)
        F, Ft = pd.concat([F, te_tr], axis=1), pd.concat([Ft, te_te], axis=1)
    if os.environ.get("FAMILY_AUX", "1") == "1":
        fam = label_families(a.train)
        if fam is not None:
            fa, fb = family_aux(F, Ft, fam.reindex(e_tr.index), groups, a.seed)
            F, Ft = pd.concat([F, fa], axis=1), pd.concat([Ft, fb], axis=1)
            log(f"family helper: {fa.shape[1]} out-of-fold family probabilities added")
    X, Xt, history = F, Ft, []
    for r in range(a.stack + 1):
        if r:
            S = stack_features(e_tr, ctx["L"], ctx["M"], oof)
            if a.stack_dropout:  # hide some neighbour scores so the model keeps using its own evidence
                S = S.mask(np.random.default_rng(a.seed + r).random(S.shape) < a.stack_dropout)
            X = pd.concat([F, S], axis=1)
            Xt = pd.concat([Ft, stack_features(e_te, ctx_t["L"], ctx_t["M"], pred)], axis=1)
        oof, rounds = cv_oof(X, y, groups, a.seed)
        last = r == a.stack
        pred, model = fit_predict(X, y, Xt, a.seed, rounds, a.seeds if last else 1)
        history.append(round(100 * aps(y, oof), 2))
        log(f"round {r}: CV AP {history[-1]:.2f}  AUC {roc_auc_score(y, oof):.4f}  trees {rounds}  ({X.shape[1]} features)")
    final, final_oof = pred, oof
    if a.xgb:
        xo, xp = xgb_oof_pred(X, y, Xt, groups, a.seed)
        log(f"xgboost: CV AP {100 * aps(y, xo):.2f}")
        best_w = max((0, 0.25, 0.5), key=lambda w: aps(y, (1 - w) * rank01(oof) + w * rank01(xo)))
        if best_w:
            final = (1 - best_w) * rank01(pred) + best_w * rank01(xp) + 1e-6 * pred
            final_oof = (1 - best_w) * rank01(oof) + best_w * rank01(xo)
        log(f"blend weight for xgboost: {best_w}  ->  CV AP {100 * aps(y, final_oof):.2f}")
    imp = pd.Series(model.feature_importance("gain"), model.feature_name()).sort_values(ascending=False)
    log("top features: " + ", ".join(imp.index[:15]))
    sub = pd.DataFrame({"TRANSACTION_HASH": e_te.index, "score": np.clip(final, 0, 1)})
    assert sub.TRANSACTION_HASH.is_unique and len(sub) == len(e_te) and sub.score.notna().all()
    sub.to_csv(a.out, index=False)
    cv_ap = 100 * aps(y, final_oof)
    msg = f"wrote {a.out}: {len(sub)} rows, {sub.score.nunique()} distinct scores, CV AP {cv_ap:.2f}"
    if a.test_labels:
        yt = upper(pd.read_csv(a.test_labels)).set_index("TRANSACTION_HASH")["LABEL"].reindex(sub.TRANSACTION_HASH)
        msg += f"  | MOCK TEST AP {100 * aps(yt, sub.score):.2f}"
    log(msg)
    os.makedirs("runs", exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open("runs/log.csv", "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')},{commit},{a.seed},{X.shape[1]},{'/'.join(map(str, history))},{cv_ap:.2f},"
                 f"{'+'.join(sorted(drop)) or '-'},{a.out}\n")
    imp.to_csv(f"runs/importance_{os.path.basename(a.out)}.csv")
    if a.dump:  # for error analysis: features, label and out-of-fold score of every training tx
        X.assign(y=y.values, oof=final_oof, SRC=e_tr.SRC, DST=e_tr.DST, TS=e_tr.TS).to_parquet("runs/oof.parquet")


if __name__ == "__main__":
    main()
