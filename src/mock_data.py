"""Realistic fake challenge data in the AWS/SonarX XRPL schema, for testing before the real data exists.

Normal life includes a legitimate twin for every laundering shape: payroll (fan-out), merchant sweeps
(fan-in then one exit), payment processors (gather-scatter), forwarding services (fast pass-through),
market-maker loops (cycles with conversions), company treasury moves (scatter-gather) and dust bots.
Laundering mixes fresh accounts with recruited mules that have normal histories. Difficulty follows the
deck (64% easy, 18% medium, 18% hard), and hard cases are slow and span days. The test week adds shapes
that never appear in training (irregular networks, peel chains) and runs 30% slower, as the README warns.

    python src/mock_data.py --out mock
"""
import argparse, json, os
import numpy as np, pandas as pd

RIPPLE_EPOCH = 946684800
XRP_TOKEN = "0x1111222211112222111122221111222211112222"
ALPHA = list("rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz")
PRICE = {"XRP": 0.55, "USD": 1.0, "EUR": 1.1, "GBP": 1.27}
HOUR, DAY = 3600, 86400
WEEK = 7 * DAY
TRAIN_SHAPES = ["fan_in", "fan_out", "scatter_gather", "gather_scatter", "cycle", "stack", "bipartite"]
TEST_SHAPES = TRAIN_SHAPES + ["irregular", "peel_chain"]


class World:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.t0 = 1_792_000_000 + seed * WEEK
        self.tx, self.tr, self.lab = [], [], []
        self.issuer = {a: self.addr() for a in ("USD", "EUR", "GBP")}
        self.lps = [self.addr() for _ in range(8)]
        self.inflow = {}  # account -> list of (t, usd), used for merchant sweeps
        self.meta, self.case = {}, ("", "")  # laundering tx -> (shape, difficulty), for per-case scoring

    def addr(self):
        return "r" + "".join(self.rng.choice(ALPHA, 33))

    def when(self):
        """A daytime-weighted moment in the week."""
        w = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 6, 5, 4, 3, 2], float)
        return int(self.rng.integers(0, 7) * DAY + self.rng.choice(24, p=w / w.sum()) * HOUR + self.rng.integers(0, HOUR))

    def pay(self, src, dst, usd, t, label=0, asset="XRP", out=None, tag=None, fail_p=0.01):
        rng, t, out = self.rng, int(t), out or asset
        if not 0 <= t < WEEK or usd <= 0:
            return None
        h = rng.bytes(32).hex().upper()
        dt = pd.Timestamp(self.t0 + t, unit="s", tz="UTC")
        fee = int(rng.integers(10, 16))
        failed = label == 0 and rng.random() < fail_p
        if out == asset:
            assets = [asset]
        elif rng.random() < 0.6:
            assets = [asset, out]
        else:
            assets = [asset, rng.choice([a for a in PRICE if a not in (asset, out)]), out]
        hops = [src, *rng.choice(self.lps, len(assets) - 1, replace=False), dst]
        extra, value, legs = {"Destination": dst}, usd, []
        for i, a in enumerate(assets):
            v = round(value / PRICE[a], 6)
            legs.append((hops[i], hops[i + 1], a, v))
            value *= rng.uniform(0.996, 1.0) if len(assets) > 1 else 1.0
        if tag is not None:
            extra["DestinationTag"] = int(tag)
        last = legs[-1]
        amount = str(int(last[3] * 1e6)) if last[2] == "XRP" else {"currency": last[2], "issuer": self.issuer[last[2]], "value": str(last[3])}
        extra["Amount"] = amount
        if len(assets) > 1:
            first = legs[0]
            extra["SendMax"] = str(int(first[3] * 1.01e6)) if first[2] == "XRP" else {"currency": first[2], "issuer": self.issuer[first[2]], "value": str(round(first[3] * 1.01, 6))}
        self.tx.append(dict(TRANSACTION_HASH=h, LEDGER_INDEX=90_000_000 + t // 4, DATETIME=dt, TIMESTAMP=self.t0 + t - RIPPLE_EPOCH,
                            ACCOUNT=src, TRANSACTION_TYPE="Payment", SOURCE_FEE=fee, FEE=fee / 1e6,
                            TRANSACTION_RESULT="tecPATH_DRY" if failed else "tesSUCCESS", SUCCESS=not failed,
                            DELIVERED_AMOUNT="unavailable" if failed else json.dumps(amount), EXTRA_TRANSACTION_FIELDS=json.dumps(extra),
                            MEMOS=None))
        if not failed:
            for i, (a, b, sym, v) in enumerate(legs):
                self.tr.append(dict(TRANSACTION_HASH=h, TRANSFER_INDEX=i if len(legs) > 1 else -999, DATETIME=dt, FROM_ADDRESS=a, TO_ADDRESS=b,
                                    TOKEN_ADDRESS=XRP_TOKEN if sym == "XRP" else self.issuer[sym], SYMBOL=sym,
                                    SOURCE_VALUE=v * 1e6 if sym == "XRP" else v, VALUE=v,
                                    TRANSFER_TYPE="direct_payment" if len(legs) == 1 else "path_payment"))
            self.inflow.setdefault(dst, []).append((t, usd))
        self.lab.append((h, label))
        return h


def normal_life(w, n_users=4500):
    rng = w.rng
    U = [w.addr() for _ in range(n_users)]
    ex = [w.addr() for _ in range(6)]
    merch = [w.addr() for _ in range(60)]
    home, tags = rng.integers(0, len(ex), n_users), rng.integers(10**5, 10**7, n_users)
    scale = np.exp(rng.normal(5.0, 1.4, n_users))
    rate = np.exp(rng.normal(1.7, 0.7, n_users))
    friends = [rng.choice(n_users, rng.integers(3, 12)) for _ in range(n_users)]
    for u in range(n_users):  # everyday payments
        for _ in range(rng.poisson(rate[u])):
            t, r, amt = w.when(), rng.random(), scale[u] * np.exp(rng.normal(0, 0.6))
            if r < 0.38:
                w.pay(U[u], U[rng.choice(friends[u])], amt, t, asset=rng.choice(["XRP", "XRP", "USD", "EUR"]))
            elif r < 0.60:
                w.pay(U[u], rng.choice(merch), amt, t)
            elif r < 0.77:
                w.pay(U[u], ex[home[u]], amt * 2, t, tag=tags[u])
            elif r < 0.95:
                w.pay(ex[home[u]], U[u], amt * 2, t)
            else:
                w.pay(U[u], U[u], amt, t, asset="XRP", out=rng.choice(["USD", "EUR", "GBP"]))
    for _ in range(900):  # quick forwarders: a withdrawal, then most of it straight on (pass-through twin)
        u, t, a = rng.integers(n_users), w.when(), scale[rng.integers(n_users)] * 3
        w.pay(ex[home[u]], U[u], a, t)
        nxt = U[rng.choice(friends[u])] if rng.random() < 0.6 else rng.choice(merch)
        w.pay(U[u], nxt, a * rng.uniform(0.85, 1.0), t + rng.integers(60, 3 * HOUR))
    for _ in range(400):  # bill splitting: one person pays several friends similar amounts (fan-out twin)
        u, t, a = rng.integers(n_users), w.when(), scale[rng.integers(n_users)]
        for f in rng.choice(friends[u], min(len(friends[u]), rng.integers(3, 9)), replace=False):
            w.pay(U[u], U[f], a * rng.uniform(0.95, 1.05), t + rng.integers(0, HOUR))
    for _ in range(400):  # group collections: friends chip in, the organiser pays the total on (fan-in twin)
        u, t, a = rng.integers(n_users), w.when(), scale[rng.integers(n_users)]
        payers = rng.choice(friends[u], min(len(friends[u]), rng.integers(4, 12)), replace=False)
        for f in payers:
            w.pay(U[f], U[u], a * rng.uniform(0.95, 1.05), t + rng.integers(0, 6 * HOUR))
        w.pay(U[u], rng.choice(merch), a * len(payers), t + rng.integers(6 * HOUR, DAY))
    for _ in range(250):  # friends' money going round: A pays B, B pays C, C pays A back (cycle twin)
        u, t, a = rng.integers(n_users), w.when(), scale[rng.integers(n_users)]
        ring = [u, *rng.choice(friends[u], min(len(friends[u]), rng.integers(2, 4)), replace=False)]
        for i in range(len(ring)):
            t += rng.integers(HOUR, 2 * DAY)
            w.pay(U[ring[i]], U[ring[(i + 1) % len(ring)]], a * rng.uniform(0.8, 1.2), t)
    for _ in range(15):  # payroll: an employer funded just before paying many staff similar amounts
        boss, staff = w.addr(), rng.choice(n_users, rng.integers(10, 40), replace=False)
        payday, salary = rng.integers(0, 7) * DAY + 9 * HOUR, rng.uniform(600, 3000)
        w.pay(rng.choice(ex), boss, salary * len(staff) * 1.02, payday - rng.integers(1, 6) * HOUR)
        for s in staff:
            w.pay(boss, U[s], salary * rng.uniform(0.95, 1.05), payday + rng.integers(0, 1800))
    for m in merch:  # merchant sweeps: collect from many customers, then one exit (a fan-in twin)
        for sweep in sorted(rng.choice(np.arange(1, 7) * DAY + 20 * HOUR, 2, replace=False)):
            got = sum(u for t, u in w.inflow.get(m, []) if sweep - 3.5 * DAY < t <= sweep)
            w.pay(m, rng.choice(ex), got * 0.97, sweep, tag=rng.integers(10**5, 10**7))
    procs = [w.addr() for _ in range(3)]
    for _ in range(60):  # payment processor batches (a gather-scatter twin)
        p, t = rng.choice(procs), w.when()
        payers = rng.choice(n_users, rng.integers(20, 50), replace=False)
        amts = [scale[i] * np.exp(rng.normal(0, 0.4)) for i in payers]
        for i, a in zip(payers, amts):
            w.pay(U[i], p, a, t + rng.integers(0, HOUR))
        sellers = rng.choice(merch, rng.integers(5, 10), replace=False)
        for s, part in zip(sellers, rng.dirichlet(np.ones(len(sellers))) * sum(amts) * 0.98):
            w.pay(p, s, part, t + HOUR + rng.integers(0, 1800))
    fwd = [w.addr() for _ in range(5)]
    for _ in range(300):  # forwarding services: in and straight out (a pass-through twin)
        f, t, a = rng.choice(fwd), w.when(), np.exp(rng.normal(5, 1))
        w.pay(U[rng.integers(n_users)], f, a, t)
        w.pay(f, U[rng.integers(n_users)], a * 0.9999, t + rng.integers(5, 60))
    mm = [w.addr() for _ in range(4)]
    for _ in range(120):  # market makers looping value through conversions (a cycle twin)
        t, a, ring = w.when(), rng.uniform(2000, 30000), rng.permutation(mm)[:3]
        assets = ["XRP", "USD", "EUR", "XRP"]
        for i in range(3):
            w.pay(ring[i], ring[(i + 1) % 3], a * 0.999 ** i, t + i * rng.integers(20, 300), asset=assets[i], out=assets[i + 1])
    for _ in range(5):  # company treasury: split to sub-accounts, then gather into operations (a scatter-gather twin)
        main, a, b, ops = (w.addr() for _ in range(4))
        for day in range(7):
            t, amt = day * DAY + 10 * HOUR, rng.uniform(5000, 50000)
            w.pay(rng.choice(ex), main, amt, t - HOUR)
            w.pay(main, a, amt * 0.5, t); w.pay(main, b, amt * 0.5, t + 60)
            w.pay(a, ops, amt * 0.5, t + 3 * HOUR); w.pay(b, ops, amt * 0.5, t + 3 * HOUR + 120)
    for _ in range(3):  # dust bots: tiny payments round a ring of fresh accounts
        ring, t = [w.addr() for _ in range(10)], w.when()
        for i in range(40):
            w.pay(ring[i % 10], ring[(i + 1) % 10], 0.00001 * PRICE["XRP"], t + i * 30)
    return U, ex


def launder(w, U, ex, n, shapes, slow=1.0):
    rng = w.rng
    for _ in range(n):
        diff = rng.choice(["easy", "medium", "hard"], p=[0.64, 0.18, 0.18])
        dmin, dmax, p_fresh, noise, p_conv = {"easy": (60, 1800, 0.6, 0.1, 0.05), "medium": (1800, 6 * HOUR, 0.35, 0.15, 0.2),
                                              "hard": (6 * HOUR, 40 * HOUR, 0.15, 0.3, 0.4)}[diff]
        dmin, dmax = dmin * slow, dmax * slow
        gap = lambda: rng.uniform(dmin, dmax)
        new = lambda: w.addr() if rng.random() < p_fresh else U[rng.integers(len(U))]
        many = lambda lo, hi: [new() for _ in range(rng.integers(lo, hi))]
        parts = lambda total, k: total * (p := np.exp(rng.normal(0, noise, k))) / p.sum()

        def L(a, b, usd, t, tag=None):
            asset = rng.choice(["XRP", "XRP", "USD", "EUR"])
            out = rng.choice([x for x in PRICE if x != asset]) if rng.random() < p_conv else asset
            h = w.pay(a, b, usd, t, label=1, asset=asset, out=out, tag=tag)
            if h:
                w.meta[h] = (shape, diff)

        def place(acct, usd, t):  # dirty money enters: an exchange withdrawal or a dirty source
            L(rng.choice(ex) if rng.random() < 0.5 else w.addr(), acct, usd, t)

        def exit_(acct, usd, t):  # integration: deposit at an exchange, or one more hop
            if rng.random() < 0.6:
                L(acct, rng.choice(ex), usd, t, tag=rng.integers(10**5, 10**7))
            else:
                L(acct, new(), usd, t)

        shape, total = str(rng.choice(shapes)), float(np.exp(rng.normal(8.2, 1.0)))
        t = rng.uniform(0, max(HOUR, WEEK - 8 * dmax))
        if shape == "fan_in":
            mules, col = many(6, 15), new()
            tt = t
            for m, p in zip(mules, parts(total, len(mules))):
                place(m, p, t + rng.uniform(0, dmax)); tt = max(tt, t + dmax + gap())
                L(m, col, p * 0.999, tt)
            exit_(col, total * 0.995, tt + gap())
        elif shape == "fan_out":
            src = new(); place(src, total, t)
            for d, p in zip(many(6, 15), parts(total * 0.998, rng.integers(6, 15))):
                t1 = t + gap(); L(src, d, p, t1)
                if rng.random() < 0.5:
                    exit_(d, p * 0.998, t1 + gap())
        elif shape == "scatter_gather":
            src, dst, mids = new(), new(), many(3, 7)
            place(src, total, t); tt = t
            for m, p in zip(mids, parts(total, len(mids))):
                t1 = t + gap(); L(src, m, p, t1)
                t2 = t1 + gap(); L(m, dst, p * 0.999, t2); tt = max(tt, t2)
            exit_(dst, total * 0.995, tt + gap())
        elif shape == "gather_scatter":
            hub, tt = new(), t
            for s, p in zip(many(4, 10), parts(total, rng.integers(4, 10))):
                place(s, p, t); t1 = t + gap(); L(s, hub, p, t1); tt = max(tt, t1)
            for d, p in zip(many(4, 10), parts(total * 0.995, rng.integers(4, 10))):
                t2 = tt + gap(); L(hub, d, p, t2)
                if rng.random() < 0.4:
                    exit_(d, p, t2 + gap())
        elif shape == "cycle":
            ring = many(3, 7); place(ring[0], total, t); v = total
            for i in range(len(ring)):
                t += gap(); L(ring[i], ring[(i + 1) % len(ring)], v, t); v *= 0.998
            exit_(ring[0], v, t + gap())
        elif shape == "stack":
            layer = many(2, 4)
            for a, p in zip(layer, parts(total, len(layer))):
                place(a, p, t)
            vals = dict(zip(layer, parts(total, len(layer))))
            for _ in range(rng.integers(2, 5)):
                t += gap(); nxt = many(2, 4); nv = {b: 0.0 for b in nxt}
                for a in layer:
                    targets = rng.choice(nxt, rng.integers(1, min(2, len(nxt)) + 1), replace=False)
                    for b, p in zip(targets, parts(vals[a] * 0.998, len(targets))):
                        L(a, b, p, t + rng.uniform(0, dmax)); nv[b] += p
                layer, vals = [b for b in nxt if nv[b] > 0], {b: v for b, v in nv.items() if v > 0}
            t += dmax
            for a in layer:
                exit_(a, vals[a], t + gap())
        elif shape == "bipartite":
            A, B = many(3, 6), many(3, 6)
            for a, p in zip(A, parts(total, len(A))):
                place(a, p, t)
                for b, q in zip(B, parts(p * 0.998, len(B))):
                    L(a, b, q, t + gap())
            for b in B:
                exit_(b, total / len(B) * 0.99, t + 2 * dmax + gap())
        elif shape == "irregular":  # test only: splits, merges and one return, no template
            nodes = many(6, 10); place(nodes[0], total, t); held = {nodes[0]: total}
            for i in range(1, len(nodes)):
                srcs = rng.choice(nodes[:i], min(i, rng.integers(1, 3)), replace=False)
                for s in srcs:
                    if held.get(s, 0) > 1:
                        p = held[s] * rng.uniform(0.3, 0.7); held[s] -= p
                        t += gap() / 2; L(s, nodes[i], p, t); held[nodes[i]] = held.get(nodes[i], 0) + p
            back = held[nodes[-1]] * 0.5
            L(nodes[-1], nodes[1], back, t + gap()); exit_(nodes[-1], held[nodes[-1]] - back, t + 2 * gap())
        else:  # peel_chain, test only: each hop forwards most and peels a slice to a cash-out
            a, v = new(), total; place(a, v, t)
            for _ in range(rng.integers(6, 12)):
                t += gap(); peel = v * rng.uniform(0.05, 0.15); b = new()
                L(a, b, v - peel, t); exit_(a, peel, t + 60); a, v = b, v - peel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mock")
    ap.add_argument("--schemes", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0, help="world seed offset")
    a = ap.parse_args()
    for split, seed, shapes, slow in [("train", 1 + 2 * a.seed, TRAIN_SHAPES, 1.0), ("test", 2 + 2 * a.seed, TEST_SHAPES, 1.3)]:
        w = World(seed)
        U, ex = normal_life(w)
        launder(w, U, ex, a.schemes, shapes, slow)
        tx = pd.DataFrame(w.tx).sort_values("DATETIME")
        tr, lab = pd.DataFrame(w.tr), pd.DataFrame(w.lab, columns=["TRANSACTION_HASH", "LABEL"])
        hours = pd.date_range(tx.DATETIME.min().floor("h"), tx.DATETIME.max().ceil("h"), freq="h")
        rates = pd.concat([pd.DataFrame({"HOUR": hours, "SYMBOL": s, "TOKEN_ADDRESS": XRP_TOKEN if s == "XRP" else w.issuer[s],
                                         "USD_PRICE": p * np.exp(np.cumsum(w.rng.normal(0, 0.003, len(hours))))}) for s, p in PRICE.items()])
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        tx.to_parquet(f"{d}/transactions.parquet"); tr.to_parquet(f"{d}/transfers.parquet")
        tx[["LEDGER_INDEX", "DATETIME"]].drop_duplicates("LEDGER_INDEX").rename(columns={"DATETIME": "CLOSE_TIME"}).to_parquet(f"{d}/blocks.parquet")
        rates.to_csv(f"{d}/market_rates.csv", index=False)
        meta = pd.DataFrame([(h, s, df) for h, (s, df) in w.meta.items()], columns=["TRANSACTION_HASH", "SHAPE", "DIFFICULTY"])
        lab.to_csv(f"{d}/labels.csv", index=False) if split == "train" else lab.merge(meta, how="left").to_csv(f"{a.out}/test_labels_hidden.csv", index=False)
        print(f"{split}: {len(tx)} tx, {len(tr)} transfers, {lab.LABEL.mean():.2%} suspicious")


if __name__ == "__main__":
    main()
