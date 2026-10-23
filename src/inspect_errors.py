"""Error analysis after `pipeline.py --dump`: which laundering did we miss, which innocent payments did we flag?

    python src/inspect_errors.py                 # 8 worst misses and 8 worst false alarms, with their flows
    python src/inspect_errors.py --hash ABC123   # one transaction and the money around it

For each case it prints the payment, the payment that funded the sender, and the receiver's next payment,
with the label and out-of-fold score of each, so you can see what the model could not.
"""
import argparse
import pandas as pd

KEY = ["log_usd_sent", "src_since_funded_h", "dst_next_out_h", "src_fwd_ratio", "dst_fwd_ratio", "m_chain_len",
       "chain_len", "in_cycle", "sg_first", "s_n_tx", "d_n_tx", "pair_n", "s_pass_through", "d_pass_through"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8); ap.add_argument("--hash")
a = ap.parse_args()
d = pd.read_parquet("runs/oof.parquet")
d["rank"] = d.oof.rank(ascending=False).astype(int)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)


def flow(h):
    r = d.loc[h]
    near = d[(d.DST == r.SRC) | (d.SRC == r.DST) | (d.SRC == r.SRC) | (d.DST == r.DST)]
    near = near[(near.TS - r.TS).abs() <= 3 * 86400].sort_values("TS")
    t = near.assign(hours=((near.TS - r.TS) / 3600).round(1), usd=near.log_usd_sent.pipe(lambda s: (s.rpow(2.718281828) - 1).round(2)),
                    src=near.SRC.str[:8], dst=near.DST.str[:8], me=(near.index == h).astype(int))
    print(t[["me", "hours", "src", "dst", "usd", "y", "oof", "rank"]].head(25).to_string())


if a.hash:
    m = d[d.index.str.startswith(a.hash.upper())]
    print(m[KEY + ["y", "oof", "rank"]].T.to_string()); flow(m.index[0])
else:
    n_pos = int(d.y.sum())
    print(f"{n_pos} laundering transactions; a perfect model ranks them all in the top {n_pos}.\n")
    misses = d[d.y == 1].sort_values("oof").head(a.n)
    alarms = d[d.y == 0].sort_values("oof", ascending=False).head(a.n)
    for title, rows in (("MISSED LAUNDERING (lowest-scored positives)", misses), ("FALSE ALARMS (highest-scored innocents)", alarms)):
        print("=" * 100, f"\n{title}\n", rows[[c for c in KEY if c in rows] + ["oof", "rank"]].round(3).to_string(), "\n")
    for h in list(misses.index[:2]) + list(alarms.index[:2]):
        print("-" * 100, f"\nflow around {h[:12]} (y={int(d.loc[h].y)}, rank {d.loc[h]['rank']})"); flow(h)
