"""Fake challenge data in the AWS/SonarX XRPL schema, for testing the pipeline before the real data exists.

Plants the training-deck shapes (fan-in, fan-out, scatter-gather, cycle, layered stack) among normal
activity (peer payments, merchants, exchange hubs, payroll). Writes <out>/{train,test}/ plus
test labels in <out>/test_labels_hidden.csv so local scoring works end to end.

    python src/mock_data.py --out mock
"""
import argparse, json, os, string
import numpy as np, pandas as pd

RIPPLE_EPOCH = 946684800
XRP_TOKEN = "0x1111222211112222111122221111222211112222"
ALPHA = "rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz"


def addr(rng):
    return "r" + "".join(rng.choice(list(ALPHA), 33))


def txhash(rng):
    return "".join(rng.choice(list("0123456789ABCDEF"), 64))


class Book:
    def __init__(self, rng, t0):
        self.rng, self.t0 = rng, t0
        self.tx, self.tr, self.lab = [], [], []
        self.issuer = {s: addr(rng) for s in ["USD", "EUR", "GBP"]}
        self.lps = [addr(rng) for _ in range(6)]  # liquidity providers / market makers

    def pay(self, src, dst, xrp_value, t, label=0, convert=False, tag=None):
        rng, h = self.rng, txhash(self.rng)
        dt = pd.Timestamp(self.t0 + t, unit="s", tz="UTC")
        fee = int(rng.integers(10, 16))
        extra = {"Destination": dst}
        if tag is not None:
            extra["DestinationTag"] = int(tag)
        if convert:  # pay XRP, receiver gets a token, 2 legs through an LP
            sym = rng.choice(["USD", "EUR", "GBP"])
            rate = {"USD": 0.55, "EUR": 0.5, "GBP": 0.43}[sym]
            tok = round(xrp_value * rate * rng.uniform(0.995, 1.0), 6)
            lp = rng.choice(self.lps)
            legs = [(src, lp, XRP_TOKEN, "XRP", xrp_value), (lp, dst, self.issuer[sym], sym, tok)]
            extra["Amount"] = {"currency": sym, "issuer": self.issuer[sym], "value": str(tok)}
            extra["SendMax"] = str(int(xrp_value * 1e6 * 1.01))
            delivered = json.dumps({"currency": sym, "issuer": self.issuer[sym], "value": str(tok)})
        else:
            legs = [(src, dst, XRP_TOKEN, "XRP", xrp_value)]
            extra["Amount"] = str(int(xrp_value * 1e6))
            delivered = json.dumps(str(int(xrp_value * 1e6)))
        self.tx.append(dict(TRANSACTION_HASH=h, LEDGER_INDEX=90_000_000 + t // 4, DATETIME=dt,
                            TIMESTAMP=self.t0 + t - RIPPLE_EPOCH, ACCOUNT=src, TRANSACTION_TYPE="Payment",
                            SOURCE_FEE=fee, FEE=fee / 1e6, TRANSACTION_RESULT="tesSUCCESS", SUCCESS=True,
                            DELIVERED_AMOUNT=delivered, EXTRA_TRANSACTION_FIELDS=json.dumps(extra), MEMOS=None))
        for i, (a, b, tk, sym, v) in enumerate(legs):
            self.tr.append(dict(TRANSACTION_HASH=h, TRANSFER_INDEX=i if len(legs) > 1 else -999, DATETIME=dt,
                                FROM_ADDRESS=a, TO_ADDRESS=b, TOKEN_ADDRESS=tk, SYMBOL=sym,
                                SOURCE_VALUE=v * 1e6 if sym == "XRP" else v, VALUE=v,
                                TRANSFER_TYPE="direct_payment" if len(legs) == 1 else "path_payment"))
        self.lab.append((h, label))


def build(seed, n_users=3000, n_normal=38000, n_schemes=90):
    rng = np.random.default_rng(seed)
    week = 7 * 86400
    b = Book(rng, 1_792_000_000 + seed * week)
    users = [addr(rng) for _ in range(n_users)]
    exchanges = [addr(rng) for _ in range(5)]
    merchants = [addr(rng) for _ in range(40)]
    employers = [addr(rng) for _ in range(8)]
    amt = lambda: float(np.round(np.exp(rng.normal(3.5, 1.4)), 6))
    for _ in range(n_normal):
        t, r = int(rng.integers(0, week)), rng.random()
        u, v = rng.choice(users), rng.choice(users)
        if r < 0.35: b.pay(u, v, amt(), t, convert=rng.random() < 0.1)
        elif r < 0.55: b.pay(u, rng.choice(merchants), amt(), t)
        elif r < 0.75: b.pay(u, rng.choice(exchanges), amt(), t, tag=rng.integers(1e5, 1e7))
        elif r < 0.92: b.pay(rng.choice(exchanges), u, amt(), t)
        else: b.pay(rng.choice(employers), u, round(rng.uniform(800, 1200), 2), t)
    for _ in range(n_schemes):
        kind, t = rng.choice(["fan_in", "fan_out", "scatter_gather", "cycle", "stack"]), int(rng.integers(0, week - 86400))
        total = float(rng.uniform(5_000, 60_000))
        fresh = lambda k: [addr(rng) for _ in range(k)]
        if kind == "fan_in":
            mules, col = fresh(int(rng.integers(6, 15))), addr(rng)
            for m in mules:
                b.pay(m, col, total / len(mules) * rng.uniform(0.9, 1.1), t + int(rng.integers(0, 7200)), 1)
            b.pay(col, rng.choice(exchanges), total * 0.98, t + 9000, 1, tag=rng.integers(1e5, 1e7))
        elif kind == "fan_out":
            src = addr(rng)
            b.pay(rng.choice(users), src, total, t, 1)
            for d in fresh(int(rng.integers(6, 15))):
                b.pay(src, d, total / 10 * rng.uniform(0.97, 1.0), t + int(rng.integers(300, 5400)), 1)
        elif kind == "scatter_gather":
            src, dst, mids = addr(rng), addr(rng), fresh(int(rng.integers(3, 7)))
            for m in mids:
                dt = int(rng.integers(600, 3600))
                b.pay(src, m, total / len(mids), t + dt, 1, convert=rng.random() < 0.3)
                b.pay(m, dst, total / len(mids) * 0.999, t + dt + int(rng.integers(600, 4000)), 1)
        elif kind == "cycle":
            ring = fresh(int(rng.integers(3, 7)))
            for i in range(len(ring)):
                b.pay(ring[i], ring[(i + 1) % len(ring)], total * (0.999 ** i), t + i * int(rng.integers(300, 4000)), 1,
                      convert=rng.random() < 0.25)
        else:  # layered stack: 2-4 rows of fresh middlemen, slow ("hard means history")
            layer = fresh(int(rng.integers(2, 4)))
            for i, a in enumerate(layer):
                b.pay(rng.choice(users), a, total / len(layer), t + i * 600, 1)
            tt = t
            for _ in range(int(rng.integers(2, 5))):
                nxt = fresh(len(layer))
                tt += int(rng.integers(3600, 6 * 3600))
                for a, c in zip(layer, nxt):
                    b.pay(a, c, total / len(layer) * 0.995, tt + int(rng.integers(0, 1800)), 1)
                layer = nxt
    tx, tr = pd.DataFrame(b.tx), pd.DataFrame(b.tr)
    lab = pd.DataFrame(b.lab, columns=["TRANSACTION_HASH", "LABEL"])
    blocks = tx[["LEDGER_INDEX", "DATETIME"]].drop_duplicates("LEDGER_INDEX").rename(columns={"DATETIME": "CLOSE_TIME"})
    hours = pd.date_range(tx.DATETIME.min().floor("h"), tx.DATETIME.max().ceil("h"), freq="h")
    rates = pd.concat([pd.DataFrame({"HOUR": hours, "SYMBOL": s, "TOKEN_ADDRESS": ta, "USD_PRICE": p * rng.uniform(0.97, 1.03, len(hours))})
                       for s, ta, p in [("XRP", XRP_TOKEN, 0.55), ("USD", b.issuer["USD"], 1.0),
                                        ("EUR", b.issuer["EUR"], 1.1), ("GBP", b.issuer["GBP"], 1.27)]])
    return tx, tr, blocks, rates, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mock")
    a = ap.parse_args()
    for split, seed in [("train", 1), ("test", 2)]:
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        tx, tr, blocks, rates, lab = build(seed)
        tx.to_parquet(f"{d}/transactions.parquet"); tr.to_parquet(f"{d}/transfers.parquet")
        blocks.to_parquet(f"{d}/blocks.parquet"); rates.to_csv(f"{d}/market_rates.csv", index=False)
        lab.to_csv(f"{d}/labels.csv" if split == "train" else f"{a.out}/test_labels_hidden.csv", index=False)
        print(split, len(tx), "tx", len(tr), "transfers", f"{lab.LABEL.mean():.2%} suspicious")


if __name__ == "__main__":
    main()
