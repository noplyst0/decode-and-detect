"""Independent practice-data generator written by Google Gemini (Argon 4, High thinking) on 2026-10-24 from a
brief describing the challenge only (it never saw our code). One indentation slip fixed by hand. Use it to test
that improvements generalise beyond our own simulator (src/mock_data.py):

    python src/mock_gemini.py --out mockG --seed 7
"""
"""
Adversarial Synthetic XRPL AML Dataset Generator
For "Decode & Detect: The XRPL AI Hack"
Usage: python gen.py --out DIR --seed N
"""

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

XRP_TOKEN_ADDRESS = "0x1111222211112222111122221111222211112222"
XRPL_ALPHABET = "rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz"

TRAIN_TARGET_TX = 40131
TEST_TARGET_TX = 40101

# Exact 64% / 18% / 18% difficulty breakdowns for suspicious transactions
TRAIN_SUSP_COUNTS = {"easy": 768, "medium": 216, "hard": 216}  # Total = 1,200 (2.99%)
TEST_SUSP_COUNTS = {"easy": 832, "medium": 234, "hard": 234}   # Total = 1,300 (3.24%)

TRAIN_SHAPES = [
    "fan_in",
    "fan_out",
    "scatter_gather",
    "gather_scatter",
    "cycle",
    "layered_stack",
    "bipartite",
    "irregular_network",
]

TEST_SHAPES = TRAIN_SHAPES + [
    "peel_chain",
    "synchronized_burst_bridge",
]


def make_xrpl_address(rng: np.random.Generator, prefix_hint: str = "") -> str:
    """Generate a realistic classic XRPL address starting with 'r'."""
    length = int(rng.integers(25, 34))
    clean_hint = "".join(c for c in prefix_hint if c in XRPL_ALPHABET)[:4]
    rem = length - 1 - len(clean_hint)
    chars = [XRPL_ALPHABET[i] for i in rng.integers(0, len(XRPL_ALPHABET), size=rem)]
    return "r" + clean_hint + "".join(chars)


def make_tx_hash(split_name: str, idx: int, seed: int) -> str:
    """Deterministic 64-char uppercase hex transaction hash."""
    payload = f"{split_name}:{seed}:{idx}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


class Ecosystem:
    """Shared XRPL address and asset universe across train and test weeks."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # Token issuers (notice TWO distinct USD issuers to test issuer+currency handling)
        self.issuer_usd_a = make_xrpl_address(rng, "USDa")
        self.issuer_usd_b = make_xrpl_address(rng, "USDb")
        self.issuer_eur = make_xrpl_address(rng, "EURg")
        self.issuer_gbp = make_xrpl_address(rng, "GBPg")

        self.assets = [
            {"id": "XRP", "symbol": "XRP", "token_address": XRP_TOKEN_ADDRESS},
            {"id": "USD_A", "symbol": "USD", "token_address": self.issuer_usd_a},
            {"id": "USD_B", "symbol": "USD", "token_address": self.issuer_usd_b},
            {"id": "EUR", "symbol": "EUR", "token_address": self.issuer_eur},
            {"id": "GBP", "symbol": "GBP", "token_address": self.issuer_gbp},
        ]
        self.asset_map = {a["id"]: a for a in self.assets}

        # Major actors
        self.exchanges = [make_xrpl_address(rng, "Exch") for _ in range(6)]
        self.exchange_hot = [make_xrpl_address(rng, "ExHt") for _ in range(6)]
        self.lps = [make_xrpl_address(rng, "LPmm") for _ in range(10)]
        self.merchants = [make_xrpl_address(rng, "Mrch") for _ in range(45)]
        self.employers = [make_xrpl_address(rng, "Corp") for _ in range(18)]
        self.processors = [make_xrpl_address(rng, "Proc") for _ in range(12)]
        self.treasuries = [make_xrpl_address(rng, "Trsy") for _ in range(15)]
        self.arb_bots = [make_xrpl_address(rng, "ArbB") for _ in range(20)]
        self.spam_bots = [make_xrpl_address(rng, "Spam") for _ in range(8)]

        # Retail / peer population
        self.peers = [make_xrpl_address(rng, "User") for _ in range(1800)]
        # Warm mules: subset of peers that participate in normal life AND medium/hard AML
        self.warm_mules_train = list(rng.choice(self.peers, size=220, replace=False))
        self.warm_mules_test = list(rng.choice(self.peers, size=240, replace=False))
        # Dedicated fresh mules per split
        self.fresh_mules_train = [make_xrpl_address(rng, "Mule") for _ in range(450)]
        self.fresh_mules_test = [make_xrpl_address(rng, "Mule") for _ in range(500)]

        # Assign persistent exchange destination tags to peers/mules
        self.user_tags = {}
        for u in self.peers + self.fresh_mules_train + self.fresh_mules_test:
            self.user_tags[u] = int(rng.integers(100000, 99999999))


def generate_market_rates(
    start_dt: datetime, hours: int, eco: Ecosystem, rng: np.random.Generator, is_test: bool
) -> tuple[pd.DataFrame, dict]:
    """
    Generate hourly USD_PRICE for each asset with realistic drift.
    Test week features a +22% XRP price regime shift and higher volatility, plus a transient
    USD_B minor depeg/basis widening to punish static unnormalized amount thresholds.
    """
    rows = []
    rate_lookup = {}  # (hour_idx, asset_id) -> usd_price

    base_xrp = 0.68 if is_test else 0.54
    xrp_drift = 0.0008 if is_test else 0.0001
    xrp_vol = 0.014 if is_test else 0.007

    base_eur = 1.09 if is_test else 1.08
    base_gbp = 1.29 if is_test else 1.27

    xrp_p = base_xrp
    eur_p = base_eur
    gbp_p = base_gbp
    usd_a_p = 1.0000
    usd_b_p = 0.9995

    for h in range(hours):
        hr_dt = start_dt + timedelta(hours=h)
        hr_str = hr_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        xrp_p = max(0.25, xrp_p * math.exp(xrp_drift + rng.normal(0, xrp_vol)))
        eur_p = max(0.90, eur_p * math.exp(rng.normal(0, 0.0025)))
        gbp_p = max(1.05, gbp_p * math.exp(rng.normal(0, 0.0028)))
        usd_a_p = float(np.clip(1.0 + rng.normal(0, 0.0004), 0.998, 1.002))

        # In test week around hours 60..110, USD_B experiences a 1.5% discount basis
        if is_test and 60 <= h <= 110:
            usd_b_p = float(np.clip(0.985 + rng.normal(0, 0.0015), 0.980, 0.992))
        else:
            usd_b_p = float(np.clip(1.0 + rng.normal(0, 0.0006), 0.997, 1.003))

        prices = {
            "XRP": round(xrp_p, 6),
            "USD_A": round(usd_a_p, 6),
            "USD_B": round(usd_b_p, 6),
            "EUR": round(eur_p, 6),
            "GBP": round(gbp_p, 6),
        }

        for asset in eco.assets:
            aid = asset["id"]
            p = prices[aid]
            rate_lookup[(h, aid)] = p
            rows.append(
                {
                    "HOUR": hr_str,
                    "SYMBOL": asset["symbol"],
                    "TOKEN_ADDRESS": asset["token_address"],
                    "USD_PRICE": p,
                }
            )

    return pd.DataFrame(rows), rate_lookup


def format_xrpl_amount(asset: dict, human_val: float) -> str | dict:
    """Format an amount as XRPL JSON representation (drops string for XRP, dict for token)."""
    if asset["symbol"] == "XRP":
        drops = max(1, int(round(human_val * 1_000_000)))
        return str(drops)
    else:
        return {
            "currency": asset["symbol"],
            "issuer": asset["token_address"],
            "value": f"{human_val:.4f}".rstrip("0").rstrip(".") if human_val >= 0.01 else f"{human_val:.6f}",
        }


class LedgerBuilder:
    """Accumulates transactions and transfer legs for a 7-day split."""

    def __init__(
        self,
        split_name: str,
        start_dt: datetime,
        eco: Ecosystem,
        rate_lookup: dict,
        rng: np.random.Generator,
        seed: int,
    ):
        self.split_name = split_name
        self.start_dt = start_dt
        self.eco = eco
        self.rate_lookup = rate_lookup
        self.rng = rng
        self.seed = seed
        self.raw_events = []  # List of dicts before sorting and hash assignment

        # Generate 7 days of blocks (1 ledger every 4 seconds = 151,200 ledgers)
        self.base_ledger_index = 84_000_000 if split_name == "train" else 84_200_000
        self.num_ledgers = (7 * 86400) // 4

    def usd_to_units(self, usd_amount: float, asset_id: str, dt: datetime) -> float:
        h = int(np.clip((dt - self.start_dt).total_seconds() // 3600, 0, 167))
        price = self.rate_lookup[(h, asset_id)]
        return max(0.000001, usd_amount / price)

    def add_payment(
        self,
        dt: datetime,
        sender: str,
        receiver: str,
        usd_amount: float,
        src_asset_id: str = "XRP",
        dst_asset_id: str | None = None,
        n_legs: int = 1,
        success: bool = True,
        tx_result: str = "tesSUCCESS",
        dest_tag: int | None = None,
        memo: str | None = None,
        label: int = 0,
        shape: str = "",
        difficulty: str = "",
        partial_slippage: float = 1.0,
    ):
        """
        Record a single XRPL Payment transaction (1-leg direct, or 2/3-leg cross-currency via LPs).
        """
        if dst_asset_id is None:
            dst_asset_id = src_asset_id

        # Clamp timestamp strictly inside the 7-day window
        min_ts = self.start_dt.timestamp()
        max_ts = (self.start_dt + timedelta(days=7) - timedelta(seconds=4)).timestamp()
        clamped_ts = float(np.clip(dt.timestamp(), min_ts, max_ts))
        # Snap to 4-second ledger close boundary
        offset_sec = int(round(clamped_ts - min_ts))
        ledger_offset = min(max(0, offset_sec // 4), self.num_ledgers - 1)
        close_dt = self.start_dt + timedelta(seconds=ledger_offset * 4)
        ledger_idx = self.base_ledger_index + ledger_offset

        src_asset = self.eco.asset_map[src_asset_id]
        dst_asset = self.eco.asset_map[dst_asset_id]

        src_units = self.usd_to_units(usd_amount, src_asset_id, close_dt)
        delivered_usd = usd_amount * partial_slippage
        dst_units = self.usd_to_units(delivered_usd, dst_asset_id, close_dt)

        nominal_dst_units = self.usd_to_units(usd_amount, dst_asset_id, close_dt)

        source_fee = int(self.rng.integers(10, 16))  # 10-15 drops burned on every tx
        fee_xrp = source_fee / 1_000_000.0

        # Build EXTRA_TRANSACTION_FIELDS
        amt_obj = format_xrpl_amount(dst_asset, nominal_dst_units)
        extra = {
            "Destination": receiver,
            "Amount": amt_obj,
        }

        is_cross = (src_asset_id != dst_asset_id) or (n_legs > 1)
        if is_cross or partial_slippage < 0.999:
            # SendMax is ceiling in source currency (>= first leg paid)
            ceiling_mult = float(self.rng.uniform(1.01, 1.06))
            extra["SendMax"] = format_xrpl_amount(src_asset, src_units * ceiling_mult)

        if dest_tag is not None:
            extra["DestinationTag"] = int(dest_tag)

        if not success:
            delivered_str = "unavailable"
            legs = []
        else:
            deliv_obj = format_xrpl_amount(dst_asset, dst_units)
            delivered_str = deliv_obj if isinstance(deliv_obj, str) else json.dumps(deliv_obj, separators=(",", ":"))

            legs = []
            if not is_cross and n_legs == 1:
                legs.append(
                    self._make_leg(
                        -999,
                        close_dt,
                        sender,
                        receiver,
                        dst_asset,
                        dst_units,
                    )
                )
            elif n_legs == 2 or (is_cross and n_legs < 3):
                lp1 = str(self.rng.choice(self.eco.lps))
                legs.append(self._make_leg(0, close_dt, sender, lp1, src_asset, src_units))
                legs.append(self._make_leg(1, close_dt, lp1, receiver, dst_asset, dst_units))
            else:
                # 3-leg path payment: Sender -> LP1 (src_asset) -> LP2 (bridge asset) -> Receiver (dst_asset)
                lp1, lp2 = self.rng.choice(self.eco.lps, size=2, replace=False)
                bridge_id = "XRP" if (src_asset_id != "XRP" and dst_asset_id != "XRP") else "USD_A"
                if bridge_id in (src_asset_id, dst_asset_id):
                    bridge_id = "EUR"
                bridge_asset = self.eco.asset_map[bridge_id]
                bridge_units = self.usd_to_units(usd_amount * 0.998, bridge_id, close_dt)
                legs.append(self._make_leg(0, close_dt, sender, str(lp1), src_asset, src_units))
                legs.append(self._make_leg(1, close_dt, str(lp1), str(lp2), bridge_asset, bridge_units))
                legs.append(self._make_leg(2, close_dt, str(lp2), receiver, dst_asset, dst_units))

        self.raw_events.append(
            {
                "LEDGER_INDEX": ledger_idx,
                "DATETIME": close_dt,
                "ACCOUNT": sender,
                "TRANSACTION_TYPE": "Payment",
                "SOURCE_FEE": source_fee,
                "FEE": fee_xrp,
                "TRANSACTION_RESULT": tx_result,
                "SUCCESS": bool(success),
                "DELIVERED_AMOUNT": delivered_str,
                "EXTRA_TRANSACTION_FIELDS": json.dumps(extra, separators=(",", ":")),
                "MEMOS": memo,
                "LABEL": int(label),
                "SHAPE": shape if label == 1 else "",
                "DIFFICULTY": difficulty if label == 1 else "",
                "legs": legs,
            }
        )

    @staticmethod
    def _make_leg(
        transfer_idx: int,
        dt: datetime,
        frm: str,
        to: str,
        asset: dict,
        human_val: float,
    ) -> dict:
        if asset["symbol"] == "XRP":
            drops = float(max(1, int(round(human_val * 1_000_000))))
            val = drops / 1_000_000.0
            ttype = "native"
        else:
            val = round(float(human_val), 6)
            drops = val
            ttype = "issued_currency"
        return {
            "TRANSFER_INDEX": transfer_idx,
            "DATETIME": dt,
            "FROM_ADDRESS": frm,
            "TO_ADDRESS": to,
            "TOKEN_ADDRESS": asset["token_address"],
            "SYMBOL": asset["symbol"],
            "SOURCE_VALUE": drops,
            "VALUE": val,
            "TRANSFER_TYPE": ttype,
        }


def emit_pattern_instance(
    builder: LedgerBuilder,
    shape: str,
    label: int,
    difficulty: str,
    is_test: bool,
    max_tx: int,
) -> int:
    """
    Generate either a suspicious laundering scenario (label=1) or its legitimate twin (label=0)
    for the requested topological shape. Returns exact number of transactions emitted (<= max_tx).
    """
    rng = builder.rng
    eco = builder.eco
    start_dt = builder.start_dt
    before_count = len(builder.raw_events)

    # Pick pool of participants based on label & difficulty
    if label == 1:
        if difficulty == "easy":
            pool = eco.fresh_mules_test if is_test else eco.fresh_mules_train
            span_seconds = float(rng.uniform(300, 5400))  # 5m to 90m
            cross_prob = 0.12
            three_leg_prob = 0.05
            base_usd = float(rng.lognormal(mean=6.1, sigma=0.55))  # ~$450
            peel_retention = float(rng.uniform(0.005, 0.02))
            forged_memo_prob = 0.05
        elif difficulty == "medium":
            warm_pool = eco.warm_mules_test if is_test else eco.warm_mules_train
            fresh_pool = eco.fresh_mules_test if is_test else eco.fresh_mules_train
            pool = warm_pool + fresh_pool[:80]
            span_seconds = float(rng.uniform(7200, 72000))  # 2h to 20h
            cross_prob = 0.48
            three_leg_prob = 0.30
            base_usd = float(rng.lognormal(mean=5.8, sigma=0.75))  # overlaps normal commerce
            peel_retention = float(rng.uniform(0.02, 0.06))
            forged_memo_prob = 0.25
        else:  # hard
            pool = eco.warm_mules_test if is_test else eco.warm_mules_train
            span_seconds = float(rng.uniform(86400 * 1.5, 86400 * 5.2))  # 1.5 to 5.2 days
            cross_prob = 0.72
            three_leg_prob = 0.55
            # Quantile-matched to normal peer/shopping amounts
            base_usd = float(rng.lognormal(mean=5.2, sigma=0.85))
            peel_retention = float(rng.uniform(0.03, 0.08))
            forged_memo_prob = 0.45
    else:
        # Legitimate twin! Uses real business hubs, employers, merchants, treasuries, peers
        pool = eco.peers
        span_seconds = float(rng.uniform(600, 86400 * 2.5))
        cross_prob = 0.32
        three_leg_prob = 0.18
        base_usd = float(rng.lognormal(mean=5.5, sigma=0.85))
        peel_retention = float(rng.uniform(0.0, 0.03))
        forged_memo_prob = 0.0

    max_start_offset = max(3600.0, 7 * 86400 - span_seconds - 1800.0)
    t0 = start_dt + timedelta(seconds=float(rng.uniform(600.0, max_start_offset)))

    asset_ids = ["XRP", "USD_A", "USD_B", "EUR", "GBP"]
    asset_probs = [0.50, 0.22, 0.14, 0.08, 0.06] if not is_test else [0.42, 0.22, 0.20, 0.09, 0.07]

    def pick_leg_spec():
        src_a = str(rng.choice(asset_ids, p=asset_probs))
        if rng.random() < cross_prob:
            dst_a = str(rng.choice([a for a in asset_ids if a != src_a]))
            n_l = 3 if rng.random() < three_leg_prob else 2
        else:
            dst_a = src_a
            n_l = 1
        return src_a, dst_a, n_l

    def pick_memo(legit_prefix: str = "REF"):
        if label == 1:
            if rng.random() < forged_memo_prob:
                return f"INV-2026-{int(rng.integers(1000, 9999))}"
            return None
        else:
            if rng.random() < 0.55:
                return f"{legit_prefix}-{int(rng.integers(10000, 99999))}"
            return None

    def step_emit(dt, s, r, usd, dest_tag=None, memo=None, force_cross=False):
        if len(builder.raw_events) - before_count >= max_tx:
            return
        src_a, dst_a, n_l = pick_leg_spec()
        if force_cross and src_a == dst_a:
            dst_a = "USD_B" if src_a != "USD_B" else "USD_A"
            n_l = 3
        slip = float(rng.uniform(0.96, 0.998)) if n_l > 1 else 1.0
        builder.add_payment(
            dt=dt,
            sender=s,
            receiver=r,
            usd_amount=max(1.5, usd),
            src_asset_id=src_a,
            dst_asset_id=dst_a,
            n_legs=n_l,
            success=True,
            tx_result="tesSUCCESS",
            dest_tag=dest_tag,
            memo=memo,
            label=label,
            shape=shape,
            difficulty=difficulty,
            partial_slippage=slip,
        )

    # Occasionally prepend a failed probe transaction (tecPATH_DRY) inside the scenario
    if max_tx >= 4 and rng.random() < (0.18 if label == 1 else 0.08):
        probe_w = str(rng.choice(pool))
        probe_dst = str(rng.choice(pool))
        builder.add_payment(
            dt=t0 - timedelta(seconds=float(rng.uniform(12, 90))),
            sender=probe_w,
            receiver=probe_dst,
            usd_amount=base_usd * 2.5,
            src_asset_id="USD_A",
            dst_asset_id="EUR",
            n_legs=2,
            success=False,
            tx_result="tecPATH_DRY",
            dest_tag=None,
            memo=None,
            label=label,
            shape=shape,
            difficulty=difficulty,
        )

    rem_budget = max_tx - (len(builder.raw_events) - before_count)
    if rem_budget <= 0:
        return len(builder.raw_events) - before_count

    if shape == "fan_in":
        k = min(rem_budget, int(rng.integers(5, 14)))
        hub = str(rng.choice(pool if label == 1 else eco.merchants))
        spokes = rng.choice(pool, size=k, replace=False)
        exch = str(rng.choice(eco.exchanges))
        has_offramp = (k >= 6 and label == 1)
        n_in = k - 1 if has_offramp else k
        total_in = 0.0
        for i in range(n_in):
            dt = t0 + timedelta(seconds=float(rng.uniform(0, span_seconds * 0.8)))
            amt = base_usd * float(rng.uniform(0.6, 1.5))
            total_in += amt
            step_emit(dt, str(spokes[i]), hub, amt, memo=pick_memo("ORDER"))
        if has_offramp:
            dt_out = t0 + timedelta(seconds=span_seconds * 0.95)
            step_emit(
                dt_out,
                hub,
                exch,
                total_in * (1.0 - peel_retention),
                dest_tag=eco.user_tags.get(hub, 555001),
                memo=pick_memo("SETTLE"),
            )

    elif shape == "fan_out":
        k = min(rem_budget, int(rng.integers(5, 14)))
        hub = str(rng.choice(pool if label == 1 else eco.employers))
        spokes = rng.choice(pool, size=k, replace=False)
        exch = str(rng.choice(eco.exchanges))
        has_placement = (k >= 6 and label == 1)
        n_out = k - 1 if has_placement else k
        total_out = base_usd * n_out
        if has_placement:
            step_emit(
                t0,
                exch,
                hub,
                total_out * 1.03,
                memo=pick_memo("WD"),
            )
        weights = rng.dirichlet(np.ones(n_out) * (8.0 if difficulty == "easy" else 2.0))
        for i in range(n_out):
            dt = t0 + timedelta(seconds=float(rng.uniform(span_seconds * 0.1, span_seconds)))
            step_emit(
                dt,
                hub,
                str(spokes[i]),
                total_out * float(weights[i]) * (1.0 - peel_retention),
                memo=pick_memo("PAYROLL"),
            )

    elif shape == "scatter_gather":
        m = min(max(2, rem_budget // 2), int(rng.integers(3, 8)))
        src = str(rng.choice(pool if label == 1 else eco.treasuries))
        dst = str(rng.choice(pool if label == 1 else eco.treasuries))
        if dst == src:
            dst = str(rng.choice(eco.exchanges))
        mids = rng.choice(pool, size=m, replace=False)
        weights = rng.dirichlet(np.ones(m) * (6.0 if difficulty == "easy" else 1.8))
        total_val = base_usd * m
        for i in range(m):
            dt1 = t0 + timedelta(seconds=float(rng.uniform(0, span_seconds * 0.45)))
            leg_val = total_val * float(weights[i])
            step_emit(dt1, src, str(mids[i]), leg_val, memo=pick_memo("ALLOC"))
        for i in range(m):
            dt2 = t0 + timedelta(seconds=float(rng.uniform(span_seconds * 0.5, span_seconds)))
            leg_val = total_val * float(weights[i]) * (1.0 - peel_retention)
            tag = eco.user_tags.get(dst, 771122) if dst in eco.exchanges else None
            step_emit(dt2, str(mids[i]), dst, leg_val, dest_tag=tag, memo=pick_memo("RECON"))

    elif shape == "gather_scatter":
        m = min(max(2, rem_budget // 2), int(rng.integers(3, 7)))
        hub = str(rng.choice(pool if label == 1 else eco.processors))
        senders = rng.choice(pool, size=m, replace=False)
        receivers = rng.choice(pool, size=m, replace=False)
        total_in = 0.0
        for i in range(m):
            dt1 = t0 + timedelta(seconds=float(rng.uniform(0, span_seconds * 0.45)))
            amt = base_usd * float(rng.uniform(0.7, 1.4))
            total_in += amt
            step_emit(dt1, str(senders[i]), hub, amt, memo=pick_memo("BATCH-IN"))
        weights = rng.dirichlet(np.ones(m) * 2.5)
        for i in range(m):
            dt2 = t0 + timedelta(seconds=float(rng.uniform(span_seconds * 0.52, span_seconds)))
            amt_out = total_in * float(weights[i]) * (1.0 - peel_retention)
            step_emit(dt2, hub, str(receivers[i]), amt_out, memo=pick_memo("DISB"))

    elif shape == "cycle":
        k = min(rem_budget, int(rng.integers(3, 8)))
        ring = list(rng.choice(pool if label == 1 else eco.arb_bots, size=k, replace=False))
        cur_val = base_usd * 2.2
        for i in range(k):
            s = str(ring[i])
            r = str(ring[(i + 1) % k])
            dt = t0 + timedelta(seconds=(span_seconds * (i + 1) / k))
            cur_val *= 1.0 - peel_retention * 0.5
            step_emit(dt, s, r, cur_val, memo=pick_memo("ARB"))

    elif shape == "layered_stack":
        # 3 layers of nodes with directed edges between consecutive layers
        n_layers = 3
        width = min(3, max(2, int(math.sqrt(max(4, rem_budget // 2)))))
        layers = [list(rng.choice(pool, size=width, replace=False)) for _ in range(n_layers)]
        for l_idx in range(n_layers - 1):
            for u in layers[l_idx]:
                targets = rng.choice(layers[l_idx + 1], size=min(2, width), replace=False)
                for v in targets:
                    dt = t0 + timedelta(
                        seconds=float(
                            rng.uniform(
                                span_seconds * (l_idx / (n_layers - 1)) * 0.8,
                                span_seconds * ((l_idx + 1) / (n_layers - 1)),
                            )
                        )
                    )
                    amt = base_usd * float(rng.uniform(0.65, 1.35)) * ((1.0 - peel_retention) ** l_idx)
                    step_emit(dt, str(u), str(v), amt, memo=pick_memo("TRSY"))

    elif shape == "bipartite":
        left_n = min(4, max(2, int(rng.integers(2, 5))))
        right_n = min(4, max(2, int(rng.integers(2, 5))))
        left_nodes = list(rng.choice(pool, size=left_n, replace=False))
        right_nodes = list(
            rng.choice(pool if label == 1 else eco.merchants, size=right_n, replace=False)
        )
        for u in left_nodes:
            for v in right_nodes:
                if rng.random() < 0.75:
                    dt = t0 + timedelta(seconds=float(rng.uniform(0, span_seconds)))
                    amt = base_usd * float(rng.uniform(0.5, 1.6))
                    step_emit(dt, str(u), str(v), amt, memo=pick_memo("B2B"))

    elif shape == "irregular_network":
        k = min(rem_budget, int(rng.integers(6, 13)))
        sub = list(rng.choice(pool, size=min(len(pool), max(5, k // 2 + 2)), replace=False))
        cur_dt = t0
        for i in range(k):
            u, v = rng.choice(sub, size=2, replace=False)
            cur_dt += timedelta(seconds=float(rng.uniform(30, max(60.0, span_seconds / k))))
            amt = base_usd * float(rng.uniform(0.4, 1.9))
            step_emit(cur_dt, str(u), str(v), amt, memo=pick_memo("SPLIT"))

    elif shape == "peel_chain":
        # Test-only novel topology: sequential chain where each hop peels a cross-currency
        # slice to an exchange DestinationTag and passes the remainder to the next hop.
        hops = min(max(2, rem_budget // 2), int(rng.integers(4, 8)))
        chain = list(rng.choice(pool, size=hops + 1, replace=False))
        running_usd = base_usd * hops * 1.4
        cur_dt = t0
        for i in range(hops):
            u = str(chain[i])
            v = str(chain[i + 1])
            exch = str(rng.choice(eco.exchanges))
            peel_frac = float(rng.uniform(0.12, 0.24))
            peel_usd = running_usd * peel_frac
            forward_usd = running_usd * (1.0 - peel_frac - peel_retention)
            cur_dt += timedelta(seconds=float(rng.uniform(120, max(240.0, span_seconds / hops))))
            # Peel off-ramp leg (forced cross-currency conversion to exchange tag)
            step_emit(
                cur_dt,
                u,
                exch,
                peel_usd,
                dest_tag=eco.user_tags.get(u, 880000 + i),
                memo=pick_memo("PEEL"),
                force_cross=True,
            )
            # Main chain continuation leg
            step_emit(
                cur_dt + timedelta(seconds=float(rng.uniform(15, 180))),
                u,
                v,
                forward_usd,
                memo=pick_memo("FWD"),
            )
            running_usd = forward_usd

    elif shape == "synchronized_burst_bridge":
        # Test-only novel topology: dormant sleeper wallets wake within a tight 90-second window,
        # bridge USD_A -> XRP -> USD_B via 3-leg swaps to collector nodes, then sweep to a single tag.
        m = min(max(2, rem_budget // 2), int(rng.integers(4, 8)))
        sleepers = list(rng.choice(pool, size=m, replace=False))
        collector = str(rng.choice(pool))
        exch = str(rng.choice(eco.exchanges))
        shared_tag = int(rng.integers(4000000, 4999999))
        total_bridged = 0.0
        for i in range(m):
            burst_dt = t0 + timedelta(seconds=float(rng.uniform(0, 90)))
            amt = base_usd * float(rng.uniform(0.85, 1.25))
            total_bridged += amt
            step_emit(
                burst_dt,
                str(sleepers[i]),
                collector,
                amt,
                memo=pick_memo("SYNC"),
                force_cross=True,
            )
        # Final consolidated integration deposit to exchange
        step_emit(
            t0 + timedelta(seconds=float(rng.uniform(300, max(600.0, span_seconds * 0.5)))),
            collector,
            exch,
            total_bridged * (1.0 - peel_retention),
            dest_tag=shared_tag,
            memo=pick_memo("INTG"),
            force_cross=True,
        )

    return len(builder.raw_events) - before_count


def populate_split(
    split_name: str,
    target_total_tx: int,
    susp_targets: dict,
    shapes_list: list[str],
    start_dt: datetime,
    eco: Ecosystem,
    rate_lookup: dict,
    rng: np.random.Generator,
    seed: int,
    is_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all transactions, transfers, and blocks for a single 7-day split."""
    builder = LedgerBuilder(split_name, start_dt, eco, rate_lookup, rng, seed)

    # 1. Generate exact suspicious transactions per difficulty bucket (64% / 18% / 18%)
    for diff in ["easy", "medium", "hard"]:
        target_diff_count = susp_targets[diff]
        emitted_diff = 0
        shape_idx = 0
        while emitted_diff < target_diff_count:
            rem = target_diff_count - emitted_diff
            # Bias novel shapes into medium/hard in the test split
            if is_test and diff in ("medium", "hard") and rng.random() < 0.32:
                sh = str(rng.choice(["peel_chain", "synchronized_burst_bridge"]))
            else:
                sh = shapes_list[shape_idx % len(shapes_list)]
                shape_idx += 1

            if rem <= 3:
                # Emit exact remaining single-hop or multi-leg layering steps to hit exact quota
                t_dt = start_dt + timedelta(seconds=float(rng.uniform(3600, 6 * 86400)))
                pool = eco.warm_mules_test if is_test else eco.warm_mules_train
                u, v = rng.choice(pool, size=2, replace=False)
                builder.add_payment(
                    dt=t_dt,
                    sender=str(u),
                    receiver=str(v),
                    usd_amount=float(rng.lognormal(5.5, 0.6)),
                    src_asset_id="USD_A" if diff != "easy" else "XRP",
                    dst_asset_id="USD_B" if diff == "hard" else "XRP",
                    n_legs=3 if diff == "hard" else 1,
                    success=True,
                    tx_result="tesSUCCESS",
                    dest_tag=None,
                    memo=None,
                    label=1,
                    shape=sh,
                    difficulty=diff,
                )
                emitted_diff += 1
            else:
                added = emit_pattern_instance(
                    builder=builder,
                    shape=sh,
                    label=1,
                    difficulty=diff,
                    is_test=is_test,
                    max_tx=rem,
                )
                emitted_diff += added

    # 2. Generate high-volume "Legitimate Twins" (LABEL=0) for EVERY topology (including test shapes)
    # ~14,500 transactions in structured benign motifs (payroll, merchants, DEX arb, processors, etc.)
    legit_twin_target = 14500
    emitted_twins = 0
    twin_idx = 0
    while emitted_twins < legit_twin_target:
        rem = legit_twin_target - emitted_twins
        sh = shapes_list[twin_idx % len(shapes_list)]
        twin_idx += 1
        added = emit_pattern_instance(
            builder=builder,
            shape=sh,
            label=0,
            difficulty="",
            is_test=is_test,
            max_tx=min(rem, 16),
        )
        emitted_twins += max(1, added)

    # 3. Deliberately Ambiguous Gray-Area Cases in Test (~60 transactions)
    if is_test:
        # Case A: Unlicensed P2P OTC aggregator that looks like gather_scatter and shares a merchant tag
        otc_desk = make_xrpl_address(rng, "OTCd")
        for _ in range(18):
            dt_a = start_dt + timedelta(seconds=float(rng.uniform(86400, 4 * 86400)))
            u = str(rng.choice(eco.warm_mules_test))
            builder.add_payment(
                dt=dt_a,
                sender=u,
                receiver=otc_desk,
                usd_amount=float(rng.lognormal(5.9, 0.5)),
                src_asset_id="XRP",
                dst_asset_id="USD_A",
                n_legs=2,
                success=True,
                tx_result="tesSUCCESS",
                dest_tag=808080,
                memo="OTC-SETTLE",
                label=0,
            )

    # 4. Adversarial Dust / Spam Fan-Out Campaigns (LABEL=0 nuisance dust)
    dust_campaigns = 18
    for _ in range(dust_campaigns):
        spammer = str(rng.choice(eco.spam_bots))
        t_spam = start_dt + timedelta(seconds=float(rng.uniform(1800, 6.8 * 86400)))
        targets = rng.choice(eco.peers, size=int(rng.integers(20, 40)), replace=False)
        for i, tgt in enumerate(targets):
            builder.add_payment(
                dt=t_spam + timedelta(seconds=i * 4),
                sender=spammer,
                receiver=str(tgt),
                usd_amount=0.000012,  # ~15-25 drops of XRP
                src_asset_id="XRP",
                dst_asset_id="XRP",
                n_legs=1,
                success=True,
                tx_result="tesSUCCESS",
                dest_tag=None,
                memo="CLAIM FREE XRP REWARD AT XRPL-DROP.EXAMPLE",
                label=0,
            )

    # 5. Exchange Hot-to-Cold Internal Sweeps & Liquidity Rebalancing (LABEL=0)
    for _ in range(320):
        dt_sw = start_dt + timedelta(seconds=float(rng.uniform(300, 7 * 86400 - 300)))
        idx_e = int(rng.integers(0, len(eco.exchanges)))
        aid = str(rng.choice(["XRP", "USD_A", "USD_B", "EUR", "GBP"], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
        builder.add_payment(
            dt=dt_sw,
            sender=eco.exchanges[idx_e],
            receiver=eco.exchange_hot[idx_e],
            usd_amount=float(rng.lognormal(8.2, 0.7)),
            src_asset_id=aid,
            dst_asset_id=aid,
            n_legs=1,
            success=True,
            tx_result="tesSUCCESS",
            dest_tag=int(rng.integers(1, 50)),
            memo="INTERNAL-SWEEP",
            label=0,
        )

    # 6. Fill remaining quota to exact target_total_tx with realistic everyday background activity:
    # shopping, peer-to-peer payments, exchange withdrawals/deposits, and ~4.5% failed tec... txs.
    # Importantly, warm mules actively participate in this normal life!
    active_users = eco.peers
    asset_ids = ["XRP", "USD_A", "USD_B", "EUR", "GBP"]
    failed_codes = [
        "tecPATH_DRY",
        "tecUNFUNDED_PAYMENT",
        "tecPATH_PARTIAL",
        "tecDST_TAG_NEEDED",
        "tecNO_DST_INSUF_XRP",
    ]

    while len(builder.raw_events) < target_total_tx:
        # Diurnal + weekday realism: sample hour with mild business-hour peak
        day_idx = int(rng.integers(0, 7))
        hour_of_day = int(rng.choice(24, p=_diurnal_probs()))
        sec_in_hour = float(rng.uniform(0, 3600))
        dt_bg = start_dt + timedelta(days=day_idx, hours=hour_of_day, seconds=sec_in_hour)

        roll = rng.random()
        is_success = rng.random() >= 0.045
        tx_res = "tesSUCCESS" if is_success else str(rng.choice(failed_codes))

        if roll < 0.38:
            # Retail shopping (User -> Merchant)
            u = str(rng.choice(active_users))
            m = str(rng.choice(eco.merchants))
            usd_val = float(rng.lognormal(mean=3.8, sigma=0.9))  # ~$15 - $250
            src_a = str(rng.choice(asset_ids, p=[0.55, 0.20, 0.10, 0.08, 0.07]))
            dst_a = str(rng.choice(["USD_A", "EUR", "GBP", "XRP"], p=[0.45, 0.20, 0.15, 0.20]))
            n_l = 1 if src_a == dst_a else (3 if rng.random() < 0.25 else 2)
            builder.add_payment(
                dt=dt_bg,
                sender=u,
                receiver=m,
                usd_amount=usd_val,
                src_asset_id=src_a,
                dst_asset_id=dst_a,
                n_legs=n_l,
                success=is_success,
                tx_result=tx_res,
                dest_tag=int(rng.integers(1000, 999999)) if rng.random() < 0.65 else None,
                memo=f"POS-{int(rng.integers(100000, 999999))}" if rng.random() < 0.35 else None,
                label=0,
            )
        elif roll < 0.72:
            # Peer-to-peer transfer (User -> User)
            u, v = rng.choice(active_users, size=2, replace=False)
            usd_val = float(rng.lognormal(mean=4.2, sigma=1.05))
            src_a = str(rng.choice(asset_ids, p=[0.62, 0.16, 0.10, 0.06, 0.06]))
            dst_a = src_a if rng.random() < 0.80 else str(rng.choice(asset_ids))
            n_l = 1 if src_a == dst_a else 2
            builder.add_payment(
                dt=dt_bg,
                sender=str(u),
                receiver=str(v),
                usd_amount=usd_val,
                src_asset_id=src_a,
                dst_asset_id=dst_a,
                n_legs=n_l,
                success=is_success,
                tx_result=tx_res,
                dest_tag=None,
                memo="Dinner split" if rng.random() < 0.08 else None,
                label=0,
            )
        elif roll < 0.88:
            # Exchange deposit (User -> Exchange with user's persistent DestinationTag)
            u = str(rng.choice(active_users))
            ex = str(rng.choice(eco.exchanges))
            usd_val = float(rng.lognormal(mean=5.4, sigma=1.1))
            src_a = str(rng.choice(asset_ids, p=[0.58, 0.18, 0.12, 0.06, 0.06]))
            dst_a = src_a if rng.random() < 0.75 else "USD_A"
            n_l = 1 if src_a == dst_a else 2
            builder.add_payment(
                dt=dt_bg,
                sender=u,
                receiver=ex,
                usd_amount=usd_val,
                src_asset_id=src_a,
                dst_asset_id=dst_a,
                n_legs=n_l,
                success=is_success,
                tx_result=tx_res,
                dest_tag=eco.user_tags[u],
                memo=None,
                label=0,
            )
        else:
            # Exchange withdrawal (Exchange -> User)
            u = str(rng.choice(active_users))
            ex = str(rng.choice(eco.exchanges))
            usd_val = float(rng.lognormal(mean=5.3, sigma=1.0))
            aid = str(rng.choice(asset_ids, p=[0.60, 0.18, 0.10, 0.06, 0.06]))
            builder.add_payment(
                dt=dt_bg,
                sender=ex,
                receiver=u,
                usd_amount=usd_val,
                src_asset_id=aid,
                dst_asset_id=aid,
                n_legs=1,
                success=is_success,
                tx_result=tx_res,
                dest_tag=None,
                memo=f"WD-{int(rng.integers(100000, 999999))}" if rng.random() < 0.4 else None,
                label=0,
            )

    # Sort all events chronologically
    builder.raw_events.sort(key=lambda e: (e["LEDGER_INDEX"], e["ACCOUNT"]))

    tx_rows = []
    transfer_rows = []
    used_ledgers = set()

    for idx, ev in enumerate(builder.raw_events):
        tx_hash = make_tx_hash(split_name, idx, seed)
        used_ledgers.add(ev["LEDGER_INDEX"])
        tx_rows.append(
            {
                "TRANSACTION_HASH": tx_hash,
                "LEDGER_INDEX": ev["LEDGER_INDEX"],
                "DATETIME": ev["DATETIME"],
                "ACCOUNT": ev["ACCOUNT"],
                "TRANSACTION_TYPE": ev["TRANSACTION_TYPE"],
                "SOURCE_FEE": ev["SOURCE_FEE"],
                "FEE": ev["FEE"],
                "TRANSACTION_RESULT": ev["TRANSACTION_RESULT"],
                "SUCCESS": ev["SUCCESS"],
                "DELIVERED_AMOUNT": ev["DELIVERED_AMOUNT"],
                "EXTRA_TRANSACTION_FIELDS": ev["EXTRA_TRANSACTION_FIELDS"],
                "MEMOS": ev["MEMOS"],
                "LABEL": ev["LABEL"],
                "SHAPE": ev["SHAPE"],
                "DIFFICULTY": ev["DIFFICULTY"],
            }
        )
        for leg in ev["legs"]:
            transfer_rows.append(
                {
                    "TRANSACTION_HASH": tx_hash,
                    "TRANSFER_INDEX": leg["TRANSFER_INDEX"],
                    "DATETIME": leg["DATETIME"],
                    "FROM_ADDRESS": leg["FROM_ADDRESS"],
                    "TO_ADDRESS": leg["TO_ADDRESS"],
                    "TOKEN_ADDRESS": leg["TOKEN_ADDRESS"],
                    "SYMBOL": leg["SYMBOL"],
                    "SOURCE_VALUE": leg["SOURCE_VALUE"],
                    "VALUE": leg["VALUE"],
                    "TRANSFER_TYPE": leg["TRANSFER_TYPE"],
                }
            )

    # Build full 7-day blocks DataFrame (every 4s ledger close in the week)
    ledger_indices = np.arange(
        builder.base_ledger_index,
        builder.base_ledger_index + builder.num_ledgers,
        dtype=np.int64,
    )
    close_times = pd.date_range(
        start=start_dt, periods=builder.num_ledgers, freq="4s", tz=timezone.utc
    )
    blocks_df = pd.DataFrame({"LEDGER_INDEX": ledger_indices, "CLOSE_TIME": close_times})

    tx_df = pd.DataFrame(tx_rows)
    transfers_df = pd.DataFrame(transfer_rows)

    return tx_df, transfers_df, blocks_df


def _diurnal_probs() -> np.ndarray:
    """Realistic UTC hourly activity weights."""
    weights = np.array(
        [
            0.025, 0.020, 0.018, 0.018, 0.020, 0.025,
            0.035, 0.045, 0.052, 0.058, 0.060, 0.062,
            0.064, 0.065, 0.064, 0.062, 0.058, 0.052,
            0.045, 0.040, 0.036, 0.032, 0.024, 0.020,
        ],
        dtype=np.float64,
    )
    return weights / weights.sum()


def write_split_outputs(
    out_dir: str,
    split_name: str,
    tx_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    rates_df: pd.DataFrame,
):
    """Write all required Parquet and CSV files for a split in the exact schema."""
    split_dir = os.path.join(out_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    tx_cols = [
        "TRANSACTION_HASH",
        "LEDGER_INDEX",
        "DATETIME",
        "ACCOUNT",
        "TRANSACTION_TYPE",
        "SOURCE_FEE",
        "FEE",
        "TRANSACTION_RESULT",
        "SUCCESS",
        "DELIVERED_AMOUNT",
        "EXTRA_TRANSACTION_FIELDS",
        "MEMOS",
    ]
    tx_df[tx_cols].to_parquet(os.path.join(split_dir, "transactions.parquet"), index=False)

    tr_cols = [
        "TRANSACTION_HASH",
        "TRANSFER_INDEX",
        "DATETIME",
        "FROM_ADDRESS",
        "TO_ADDRESS",
        "TOKEN_ADDRESS",
        "SYMBOL",
        "SOURCE_VALUE",
        "VALUE",
        "TRANSFER_TYPE",
    ]
    transfers_df[tr_cols].to_parquet(os.path.join(split_dir, "transfers.parquet"), index=False)

    blocks_df[["LEDGER_INDEX", "CLOSE_TIME"]].to_parquet(
        os.path.join(split_dir, "blocks.parquet"), index=False
    )

    rates_df[["HOUR", "SYMBOL", "TOKEN_ADDRESS", "USD_PRICE"]].to_csv(
        os.path.join(split_dir, "market_rates.csv"), index=False
    )

    if split_name == "train":
        tx_df[["TRANSACTION_HASH", "LABEL"]].to_csv(
            os.path.join(split_dir, "labels.csv"), index=False
        )
    else:
        tx_df[["TRANSACTION_HASH", "LABEL", "SHAPE", "DIFFICULTY"]].to_csv(
            os.path.join(out_dir, "test_labels_hidden.csv"), index=False
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial synthetic XRPL dataset for AML hackathon."
    )
    parser.add_argument("--out", type=str, required=True, help="Output directory path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    eco = Ecosystem(rng)

    train_start = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    test_start = datetime(2026, 10, 10, 0, 0, 0, tzinfo=timezone.utc)

    # Generate hourly market rates (7 days = 168 hours per split)
    train_rates_df, train_rate_lookup = generate_market_rates(
        train_start, 168, eco, rng, is_test=False
    )
    test_rates_df, test_rate_lookup = generate_market_rates(
        test_start, 168, eco, rng, is_test=True
    )

    # Generate Train Split (40,131 transactions)
    train_tx, train_tr, train_blocks = populate_split(
        split_name="train",
        target_total_tx=TRAIN_TARGET_TX,
        susp_targets=TRAIN_SUSP_COUNTS,
        shapes_list=TRAIN_SHAPES,
        start_dt=train_start,
        eco=eco,
        rate_lookup=train_rate_lookup,
        rng=rng,
        seed=args.seed,
        is_test=False,
    )
    write_split_outputs(args.out, "train", train_tx, train_tr, train_blocks, train_rates_df)

    # Generate Test Split (40,101 transactions)
    test_tx, test_tr, test_blocks = populate_split(
        split_name="test",
        target_total_tx=TEST_TARGET_TX,
        susp_targets=TEST_SUSP_COUNTS,
        shapes_list=TEST_SHAPES,
        start_dt=test_start,
        eco=eco,
        rate_lookup=test_rate_lookup,
        rng=rng,
        seed=args.seed + 1000,
        is_test=True,
    )
    write_split_outputs(args.out, "test", test_tx, test_tr, test_blocks, test_rates_df)

    print(
        f"Done. Train: {len(train_tx)} txs ({int(train_tx['LABEL'].sum())} suspicious), "
        f"{len(train_tr)} transfers | Test: {len(test_tx)} txs "
        f"({int(test_tx['LABEL'].sum())} suspicious), {len(test_tr)} transfers."
    )


if __name__ == "__main__":
    main()
