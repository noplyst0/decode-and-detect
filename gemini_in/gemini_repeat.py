#CODE
import numpy as np
import pandas as pd


def repeat_features(e: pd.DataFrame) -> pd.DataFrame:
    orig_index = e.index
    n = len(e)
    if n == 0:
        return pd.DataFrame(index=orig_index)

    # Deterministic chronological sort
    order = np.lexsort((np.arange(n, dtype=np.int32), e['TS'].values))
    inv_order = np.empty(n, dtype=np.int32)
    inv_order[order] = np.arange(n, dtype=np.int32)

    ts = e['TS'].values[order].astype(np.float64)
    u_sent = e['USD_SENT'].values[order].astype(np.float64)
    u_recv = e['USD_RECV'].values[order].astype(np.float64)

    src_raw = e['SRC'].values[order]
    dst_raw = e['DST'].values[order]
    wallets, w_inv = np.unique(np.concatenate([src_raw, dst_raw]), return_inverse=True)
    W = len(wallets)
    src_id = w_inv[:n].astype(np.int32)
    dst_id = w_inv[n:].astype(np.int32)

    w_tot_deg = (
        np.bincount(src_id, minlength=W) + np.bincount(dst_id, minlength=W)
    ).astype(np.int32)

    # 1. Build 5-minute (300 s) single-linkage bursts via sorted wallet event lists
    w_2n = np.concatenate([src_id, dst_id])
    ts_2n = np.concatenate([ts, ts])
    tx_2n = np.concatenate([np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32)])
    ev_ord = np.lexsort((tx_2n, ts_2n, w_2n))
    w_s = w_2n[ev_ord]
    ts_s = ts_2n[ev_ord]
    tx_s = tx_2n[ev_ord]

    same_w_adj = (w_s[1:] == w_s[:-1]) & ((ts_s[1:] - ts_s[:-1]) <= 300.0)
    adj_idx = np.flatnonzero(same_w_adj)
    edge_a = tx_s[adj_idx]
    edge_b = tx_s[adj_idx + 1]
    keep_e = edge_a != edge_b
    edge_a, edge_b = edge_a[keep_e], edge_b[keep_e]

    parent = np.arange(n, dtype=np.int32)
    for a, b in zip(edge_a, edge_b):
        ra, rb = int(a), int(b)
        while parent[ra] != ra:
            parent[ra] = parent[parent[ra]]
            ra = parent[ra]
        while parent[rb] != rb:
            parent[rb] = parent[parent[rb]]
            rb = parent[rb]
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r

    _, burst_id = np.unique(parent, return_inverse=True)
    burst_id = burst_id.astype(np.int32)
    B = int(burst_id.max()) + 1

    # 2. Compute burst-level aggregations and identify hub, role, and counterparty set
    b_size = np.bincount(burst_id, minlength=B).astype(np.int32)
    b_usd = np.bincount(burst_id, weights=u_sent, minlength=B).astype(np.float64)
    b_tstart = np.full(B, 1e18, dtype=np.float64)
    b_tend = np.full(B, -1e18, dtype=np.float64)
    np.minimum.at(b_tstart, burst_id, ts)
    np.maximum.at(b_tend, burst_id, ts)
    b_tmid = 0.5 * (b_tstart + b_tend)

    b_hub = np.empty(B, dtype=np.int32)
    b_role = np.empty(B, dtype=np.int8)  # 1 = sender-hub, 0 = receiver-hub
    b_star_purity = np.ones(B, dtype=np.float64)
    b_cps = [None] * B
    b_wallets = [None] * B

    # Fast vectorized initialization for size-1 bursts
    s1_mask = b_size == 1
    if np.any(s1_mask):
        s1_tx = np.empty(B, dtype=np.int32)
        s1_tx[burst_id] = np.arange(n, dtype=np.int32)
        tx1 = s1_tx[s1_mask]
        u1, v1 = src_id[tx1], dst_id[tx1]
        u_wins = w_tot_deg[u1] >= w_tot_deg[v1]
        b_hub[s1_mask] = np.where(u_wins, u1, v1)
        b_role[s1_mask] = np.where(u_wins, 1, 0).astype(np.int8)
        cp1 = np.where(u_wins, v1, u1)
        for bid, h_w, c_w in zip(np.flatnonzero(s1_mask), b_hub[s1_mask], cp1):
            b_cps[int(bid)] = {int(c_w)}
            b_wallets[int(bid)] = (int(h_w), int(c_w))

    # Group transactions for multi-tx bursts (size >= 2)
    b_ord = np.argsort(burst_id, kind='mergesort')
    b_splits = np.split(b_ord, np.cumsum(b_size)[:-1])
    multi_bids = np.flatnonzero(b_size >= 2)

    for bid in multi_bids:
        txs = b_splits[bid]
        bs = src_id[txs]
        bd = dst_id[txs]
        uniq_w = np.unique(np.concatenate([bs, bd]))
        b_wallets[bid] = uniq_w

        # Count in-burst out-degree and in-degree per participating wallet
        out_c = np.array([np.sum(bs == w) for w in uniq_w], dtype=np.int32)
        in_c = np.array([np.sum(bd == w) for w in uniq_w], dtype=np.int32)
        tot_c = out_c + in_c
        asym_c = np.abs(out_c - in_c)
        glob_c = w_tot_deg[uniq_w]

        # Break ties by: highest burst degree -> highest role asymmetry -> highest out_deg -> global degree
        best_idx = int(np.lexsort((-uniq_w, glob_c, out_c, asym_c, tot_c))[-1])
        hub_w = int(uniq_w[best_idx])
        role = 1 if out_c[best_idx] >= in_c[best_idx] else 0
        b_hub[bid] = hub_w
        b_role[bid] = role

        # Counterparty set = all non-hub wallets in the burst
        cp_set = set(int(w) for w in uniq_w if int(w) != hub_w)
        b_cps[bid] = cp_set

        # Star purity: fraction of txs touching hub minus non-hub intermediary fraction
        hub_touch = float(tot_c[best_idx]) / float(len(txs))
        non_hub_mid = float(np.sum((out_c > 0) & (in_c > 0)) - (1 if (out_c[best_idx] > 0 and in_c[best_idx] > 0) else 0))
        mid_pen = non_hub_mid / max(1.0, float(len(uniq_w) - 1))
        b_star_purity[bid] = np.clip(hub_touch * (1.0 - 0.5 * mid_pen), 0.0, 1.0)

    # 3. Index multi-tx 5m bursts by (hub, role) and compute repeat metrics
    hr_groups = {}
    for bid in multi_bids:
        key = (int(b_hub[bid]), int(b_role[bid]))
        hr_groups.setdefault(key, []).append(int(bid))

    b_same_cnt = np.zeros(B, dtype=np.float64)
    b_max_jacc = np.zeros(B, dtype=np.float64)
    b_med_space_h = np.full(B, -1.0, dtype=np.float64)
    b_size_sim = np.full(B, -1.0, dtype=np.float64)
    b_usd_sim = np.full(B, -1.0, dtype=np.float64)

    for bid in range(B):
        key = (int(b_hub[bid]), int(b_role[bid]))
        cands = hr_groups.get(key, [])
        others = [c for c in cands if c != bid]
        if not others:
            continue
        b_same_cnt[bid] = float(len(others))
        cp_self = b_cps[bid]
        t_self = b_tmid[bid]
        s_self = float(b_size[bid])
        u_self = b_usd[bid]

        max_j = 0.0
        spaces, s_sims, u_sims = [], [], []
        for oc in others:
            cp_o = b_cps[oc]
            un = len(cp_self | cp_o)
            if un > 0:
                j = len(cp_self & cp_o) / float(un)
                if j > max_j:
                    max_j = j
            spaces.append(abs(t_self - b_tmid[oc]) / 3600.0)
            s_o = float(b_size[oc])
            u_o = b_usd[oc]
            s_sims.append(min(s_self, s_o) / max(s_self, s_o, 1.0))
            u_sims.append(min(u_self, u_o) / max(u_self, u_o, 1e-6))

        b_max_jacc[bid] = max_j
        b_med_space_h[bid] = float(np.median(spaces))
        b_size_sim[bid] = float(np.mean(s_sims))
        b_usd_sim[bid] = float(np.mean(u_sims))

    # 4. Pre/post 2-hour outside-burst activity for burst wallets and hub wallet
    w_splits = np.split(ev_ord, np.cumsum(np.bincount(w_s, minlength=W))[:-1])
    w_ts_list = [ts_s[sl] for sl in w_splits]
    w_bid_list = [burst_id[tx_s[sl]] for sl in w_splits]

    b_quiet_2h_bef = np.ones(B, dtype=np.float64)
    b_quiet_2h_aft = np.ones(B, dtype=np.float64)
    b_hub_min_gap_h = np.full(B, 168.0, dtype=np.float64)

    for bid in range(B):
        t0, t1 = b_tstart[bid], b_tend[bid]
        w_arr = b_wallets[bid]
        hub_w = int(b_hub[bid])
        q_bef, q_aft = 0, 0
        for w in w_arr:
            wt = w_ts_list[w]
            wb = w_bid_list[w]
            # Check events strictly outside this burst
            lo = np.searchsorted(wt, t0 - 7200.0, side='left')
            hi = np.searchsorted(wt, t0, side='right')
            has_bef = np.any(wb[lo:hi] != bid)
            if not has_bef:
                q_bef += 1

            lo_a = np.searchsorted(wt, t1, side='left')
            hi_a = np.searchsorted(wt, t1 + 7200.0, side='right')
            has_aft = np.any(wb[lo_a:hi_a] != bid)
            if not has_aft:
                q_aft += 1

            if w == hub_w:
                out_mask = wb != bid
                if np.any(out_mask):
                    wt_out = wt[out_mask]
                    pos = np.searchsorted(wt_out, t0)
                    g_min = 1e18
                    if pos > 0:
                        g_min = min(g_min, max(0.0, t0 - wt_out[pos - 1]))
                    if pos < len(wt_out):
                        g_min = min(g_min, max(0.0, wt_out[pos] - t1))
                    b_hub_min_gap_h[bid] = g_min / 3600.0

        nw = float(len(w_arr))
        b_quiet_2h_bef[bid] = q_bef / nw
        b_quiet_2h_aft[bid] = q_aft / nw

    # 5. Build 45-minute hub-centric episodes to catch slower relay episodes & delayed collector flow
    src_counts = np.bincount(src_id, minlength=W)
    w_Outs = np.split(np.argsort(src_id, kind='mergesort'), np.cumsum(src_counts)[:-1])
    dst_counts = np.bincount(dst_id, minlength=W)
    w_Ins = np.split(np.argsort(dst_id, kind='mergesort'), np.cumsum(dst_counts)[:-1])

    w_ep45 = {0: [[] for _ in range(W)], 1: [[] for _ in range(W)]}
    for w in range(W):
        for role, tx_arr, cp_arr, u_arr in (
            (1, w_Outs[w], dst_id, u_sent),
            (0, w_Ins[w], src_id, u_recv),
        ):
            if len(tx_arr) == 0:
                continue
            t_arr = ts[tx_arr]
            splits = np.where(np.diff(t_arr) > 2700.0)[0] + 1
            for grp in np.split(tx_arr, splits):
                cps = set(int(c) for c in cp_arr[grp] if int(c) != w)
                w_ep45[role][w].append({
                    't0': float(ts[grp[0]]),
                    't1': float(ts[grp[-1]]),
                    'tmid': 0.5 * float(ts[grp[0]] + ts[grp[-1]]),
                    'usd': float(np.sum(u_arr[grp])),
                    'cps': cps,
                    'n_tx': len(grp),
                })

    b_ep45_same_cnt = np.zeros(B, dtype=np.float64)
    b_ep45_max_jacc = np.zeros(B, dtype=np.float64)
    b_cp_other_day = np.zeros(B, dtype=np.float64)
    b_delayed_flow_sim = np.zeros(B, dtype=np.float64)
    b_delayed_opp_cps = np.zeros(B, dtype=np.float64)

    for bid in range(B):
        hub_w = int(b_hub[bid])
        role = int(b_role[bid])
        t0, t1, tmid = b_tstart[bid], b_tend[bid], b_tmid[bid]
        u_self = b_usd[bid]
        cp_self = b_cps[bid]

        # Same-role 45m episodes separated by >2h from this burst
        same_eps = [
            ep for ep in w_ep45[role][hub_w]
            if abs(ep['tmid'] - tmid) > 7200.0 and (ep['n_tx'] >= 2 or len(ep['cps']) >= 2)
        ]
        b_ep45_same_cnt[bid] = float(len(same_eps))
        if same_eps and cp_self:
            b_ep45_max_jacc[bid] = max(
                len(cp_self & ep['cps']) / float(len(cp_self | ep['cps']))
                for ep in same_eps if len(cp_self | ep['cps']) > 0
            )

        # Counterparty recurrence with hub_w outside +/- 2h
        all_other_cps = set()
        for r_k in (0, 1):
            for ep in w_ep45[r_k][hub_w]:
                if ep['t1'] < t0 - 7200.0 or ep['t0'] > t1 + 7200.0:
                    all_other_cps |= ep['cps']
        if cp_self:
            b_cp_other_day[bid] = len(cp_self & all_other_cps) / float(len(cp_self))

        # Opposite-role delayed collector flow (15m to 36h before sender-hub, or after receiver-hub)
        opp_eps = w_ep45[1 - role][hub_w]
        best_sim, best_opp_cp = 0.0, 0.0
        sum_opp_usd = 0.0
        union_opp_cps = set()
        for ep in opp_eps:
            dt = (t0 - ep['t1']) if role == 1 else (ep['t0'] - t1)
            if 900.0 <= dt <= 129600.0:
                sim = min(u_self, ep['usd']) / max(u_self, ep['usd'], 1e-6)
                if sim > best_sim:
                    best_sim = sim
                    best_opp_cp = float(len(ep['cps']))
                sum_opp_usd += ep['usd']
                union_opp_cps |= ep['cps']
        if sum_opp_usd > 0:
            agg_sim = min(u_self, sum_opp_usd) / max(u_self, sum_opp_usd, 1e-6)
            if agg_sim > best_sim:
                best_sim = agg_sim
                best_opp_cp = float(len(union_opp_cps))
        b_delayed_flow_sim[bid] = best_sim
        b_delayed_opp_cps[bid] = best_opp_cp

    # Composite relay-burst false-alarm trap (high when burst is a star with repeating/collector hub)
    b_relay_trap = b_star_purity * (
        0.30 * np.tanh(b_same_cnt / 2.0)
        + 0.35 * np.tanh(b_ep45_same_cnt / 2.0)
        + 0.45 * b_cp_other_day
        + 0.45 * b_delayed_flow_sim * np.tanh(b_delayed_opp_cps / 2.0)
        + 0.30 * np.maximum(b_max_jacc, b_ep45_max_jacc)
    )

    tx_bid = burst_id[inv_order]
    out = pd.DataFrame(index=orig_index)
    # 1. Count of other 5m bursts (size>=2) in the week with the same hub and role
    out['rp_same_hub_role_cnt'] = b_same_cnt[tx_bid]
    # 2. Max counterparty Jaccard overlap with other 5m bursts of the same hub and role
    out['rp_max_cp_jaccard'] = b_max_jacc[tx_bid]
    # 3. Median spacing in hours to other 5m bursts with the same hub and role (-1 if none)
    out['rp_med_spacing_h'] = b_med_space_h[tx_bid]
    # 4. Mean burst size similarity (min/max) with other 5m bursts of same hub and role
    out['rp_size_sim'] = b_size_sim[tx_bid]
    # 5. Mean burst total-USD similarity (min/max) with other 5m bursts of same hub and role
    out['rp_usd_sim'] = b_usd_sim[tx_bid]
    # 6. Fraction of burst wallets with zero outside-burst activity in the 2h before burst
    out['rp_quiet_2h_before'] = b_quiet_2h_bef[tx_bid]
    # 7. Fraction of burst wallets with zero outside-burst activity in the 2h after burst
    out['rp_quiet_2h_after'] = b_quiet_2h_aft[tx_bid]
    # 8. Minimum time gap in hours from the burst hub wallet to nearest outside-burst tx
    out['rp_hub_min_out_gap_h'] = b_hub_min_gap_h[tx_bid]
    # 9. Count of other 45m multi-counterparty episodes (>2h away) on the same hub and role
    out['rp_ep45_same_hub_role_cnt'] = b_ep45_same_cnt[tx_bid]
    # 10. Max counterparty Jaccard with other 45m episodes on the same hub and role
    out['rp_ep45_max_cp_jaccard'] = b_ep45_max_jacc[tx_bid]
    # 11. Share of burst counterparties that transact with the hub >2h outside this burst
    out['rp_cp_other_day_share'] = b_cp_other_day[tx_bid]
    # 12. USD similarity between burst total and hub's opposite-role collector flow 15m-36h away
    out['rp_hub_delayed_flow_sim'] = b_delayed_flow_sim[tx_bid]
    # 13. Distinct opposite-role counterparties funding/draining the hub 15m-36h away
    out['rp_hub_delayed_opp_cp_cnt'] = b_delayed_opp_cps[tx_bid]
    # 14. Star topology purity of the 5m burst (1.0 = pure hub star, lower = multi-hop motif)
    out['rp_burst_star_purity'] = b_star_purity[tx_bid]
    # 15. Composite relay-burst false-alarm trap score targeting aggregate/distributed relays
    out['rp_relay_burst_trap'] = b_relay_trap[tx_bid]

    return out.astype(np.float64)


#FULLTEXT
# Gemini said
# Python
# import numpy as np
# import pandas as pd
# 
# 
# def repeat_features(e: pd.DataFrame) -> pd.DataFrame:
#     orig_index = e.index
#     n = len(e)
#     if n == 0:
#         return pd.DataFrame(index=orig_index)
# 
#     # Deterministic chronological sort
#     order = np.lexsort((np.arange(n, dtype=np.int32), e['TS'].values))
#     inv_order = np.empty(n, dtype=np.int32)
#     inv_order[order] = np.arange(n, dtype=np.int32)
# 
#     ts = e['TS'].values[order].astype(np.float64)
#     u_sent = e['USD_SENT'].values[order].astype(np.float64)
#     u_recv = e['USD_RECV'].values[order].astype(np.float64)
# 
#     src_raw = e['SRC'].values[order]
#     dst_raw = e['DST'].values[order]
#     wallets, w_inv = np.unique(np.concatenate([src_raw, dst_raw]), return_inverse=True)
#     W = len(wallets)
#     src_id = w_inv[:n].astype(np.int32)
#     dst_id = w_inv[n:].astype(np.int32)
# 
#     w_tot_deg = (
#         np.bincount(src_id, minlength=W) + np.bincount(dst_id, minlength=W)
#     ).astype(np.int32)
# 
#     # 1. Build 5-minute (300 s) single-linkage bursts via sorted wallet event lists
#     w_2n = np.concatenate([src_id, dst_id])
#     ts_2n = np.concatenate([ts, ts])
#     tx_2n = np.concatenate([np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32)])
#     ev_ord = np.lexsort((tx_2n, ts_2n, w_2n))
#     w_s = w_2n[ev_ord]
#     ts_s = ts_2n[ev_ord]
#     tx_s = tx_2n[ev_ord]
# 
#     same_w_adj = (w_s[1:] == w_s[:-1]) & ((ts_s[1:] - ts_s[:-1]) <= 300.0)
#     adj_idx = np.flatnonzero(same_w_adj)
#     edge_a = tx_s[adj_idx]
#     edge_b = tx_s[adj_idx + 1]
#     keep_e = edge_a != edge_b
#     edge_a, edge_b = edge_a[keep_e], edge_b[keep_e]
# 
#     parent = np.arange(n, dtype=np.int32)
#     for a, b in zip(edge_a, edge_b):
#         ra, rb = int(a), int(b)
#         while parent[ra] != ra:
#             parent[ra] = parent[parent[ra]]
#             ra = parent[ra]
#         while parent[rb] != rb:
#             parent[rb] = parent[parent[rb]]
#             rb = parent[rb]
#         if ra != rb:
#             if ra < rb:
#                 parent[rb] = ra
#             else:
#                 parent[ra] = rb
#     for i in range(n):
#         r = i
#         while parent[r] != r:
#             r = parent[r]
#         parent[i] = r
# 
#     _, burst_id = np.unique(parent, return_inverse=True)
#     burst_id = burst_id.astype(np.int32)
#     B = int(burst_id.max()) + 1
# 
#     # 2. Compute burst-level aggregations and identify hub, role, and counterparty set
#     b_size = np.bincount(burst_id, minlength=B).astype(np.int32)
#     b_usd = np.bincount(burst_id, weights=u_sent, minlength=B).astype(np.float64)
#     b_tstart = np.full(B, 1e18, dtype=np.float64)
#     b_tend = np.full(B, -1e18, dtype=np.float64)
#     np.minimum.at(b_tstart, burst_id, ts)
#     np.maximum.at(b_tend, burst_id, ts)
#     b_tmid = 0.5 * (b_tstart + b_tend)
# 
#     b_hub = np.empty(B, dtype=np.int32)
#     b_role = np.empty(B, dtype=np.int8)  # 1 = sender-hub, 0 = receiver-hub
#     b_star_purity = np.ones(B, dtype=np.float64)
#     b_cps = [None] * B
#     b_wallets = [None] * B
# 
#     # Fast vectorized initialization for size-1 bursts
#     s1_mask = b_size == 1
#     if np.any(s1_mask):
#         s1_tx = np.empty(B, dtype=np.int32)
#         s1_tx[burst_id] = np.arange(n, dtype=np.int32)
#         tx1 = s1_tx[s1_mask]
#         u1, v1 = src_id[tx1], dst_id[tx1]
#         u_wins = w_tot_deg[u1] >= w_tot_deg[v1]
#         b_hub[s1_mask] = np.where(u_wins, u1, v1)
#         b_role[s1_mask] = np.where(u_wins, 1, 0).astype(np.int8)
#         cp1 = np.where(u_wins, v1, u1)
#         for bid, h_w, c_w in zip(np.flatnonzero(s1_mask), b_hub[s1_mask], cp1):
#             b_cps[int(bid)] = {int(c_w)}
#             b_wallets[int(bid)] = (int(h_w), int(c_w))
# 
#     # Group transactions for multi-tx bursts (size >= 2)
#     b_ord = np.argsort(burst_id, kind='mergesort')
#     b_splits = np.split(b_ord, np.cumsum(b_size)[:-1])
#     multi_bids = np.flatnonzero(b_size >= 2)
# 
#     for bid in multi_bids:
#         txs = b_splits[bid]
#         bs = src_id[txs]
#         bd = dst_id[txs]
#         uniq_w = np.unique(np.concatenate([bs, bd]))
#         b_wallets[bid] = uniq_w
# 
#         # Count in-burst out-degree and in-degree per participating wallet
#         out_c = np.array([np.sum(bs == w) for w in uniq_w], dtype=np.int32)
#         in_c = np.array([np.sum(bd == w) for w in uniq_w], dtype=np.int32)
#         tot_c = out_c + in_c
#         asym_c = np.abs(out_c - in_c)
#         glob_c = w_tot_deg[uniq_w]
# 
#         # Break ties by: highest burst degree -> highest role asymmetry -> highest out_deg -> global degree
#         best_idx = int(np.lexsort((-uniq_w, glob_c, out_c, asym_c, tot_c))[-1])
#         hub_w = int(uniq_w[best_idx])
#         role = 1 if out_c[best_idx] >= in_c[best_idx] else 0
#         b_hub[bid] = hub_w
#         b_role[bid] = role
# 
#         # Counterparty set = all non-hub wallets in the burst
#         cp_set = set(int(w) for w in uniq_w if int(w) != hub_w)
#         b_cps[bid] = cp_set
# 
#         # Star purity: fraction of txs touching hub minus non-hub intermediary fraction
#         hub_touch = float(tot_c[best_idx]) / float(len(txs))
#         non_hub_mid = float(np.sum((out_c > 0) & (in_c > 0)) - (1 if (out_c[best_idx] > 0 and in_c[best_idx] > 0) else 0))
#         mid_pen = non_hub_mid / max(1.0, float(len(uniq_w) - 1))
#         b_star_purity[bid] = np.clip(hub_touch * (1.0 - 0.5 * mid_pen), 0.0, 1.0)
# 
#     # 3. Index multi-tx 5m bursts by (hub, role) and compute repeat metrics
#     hr_groups = {}
#     for bid in multi_bids:
#         key = (int(b_hub[bid]), int(b_role[bid]))
#         hr_groups.setdefault(key, []).append(int(bid))
# 
#     b_same_cnt = np.zeros(B, dtype=np.float64)
#     b_max_jacc = np.zeros(B, dtype=np.float64)
#     b_med_space_h = np.full(B, -1.0, dtype=np.float64)
#     b_size_sim = np.full(B, -1.0, dtype=np.float64)
#     b_usd_sim = np.full(B, -1.0, dtype=np.float64)
# 
#     for bid in range(B):
#         key = (int(b_hub[bid]), int(b_role[bid]))
#         cands = hr_groups.get(key, [])
#         others = [c for c in cands if c != bid]
#         if not others:
#             continue
#         b_same_cnt[bid] = float(len(others))
#         cp_self = b_cps[bid]
#         t_self = b_tmid[bid]
#         s_self = float(b_size[bid])
#         u_self = b_usd[bid]
# 
#         max_j = 0.0
#         spaces, s_sims, u_sims = [], [], []
#         for oc in others:
#             cp_o = b_cps[oc]
#             un = len(cp_self | cp_o)
#             if un > 0:
#                 j = len(cp_self & cp_o) / float(un)
#                 if j > max_j:
#                     max_j = j
#             spaces.append(abs(t_self - b_tmid[oc]) / 3600.0)
#             s_o = float(b_size[oc])
#             u_o = b_usd[oc]
#             s_sims.append(min(s_self, s_o) / max(s_self, s_o, 1.0))
#             u_sims.append(min(u_self, u_o) / max(u_self, u_o, 1e-6))
# 
#         b_max_jacc[bid] = max_j
#         b_med_space_h[bid] = float(np.median(spaces))
#         b_size_sim[bid] = float(np.mean(s_sims))
#         b_usd_sim[bid] = float(np.mean(u_sims))
# 
#     # 4. Pre/post 2-hour outside-burst activity for burst wallets and hub wallet
#     w_splits = np.split(ev_ord, np.cumsum(np.bincount(w_s, minlength=W))[:-1])
#     w_ts_list = [ts_s[sl] for sl in w_splits]
#     w_bid_list = [burst_id[tx_s[sl]] for sl in w_splits]
# 
#     b_quiet_2h_bef = np.ones(B, dtype=np.float64)
#     b_quiet_2h_aft = np.ones(B, dtype=np.float64)
#     b_hub_min_gap_h = np.full(B, 168.0, dtype=np.float64)
# 
#     for bid in range(B):
#         t0, t1 = b_tstart[bid], b_tend[bid]
#         w_arr = b_wallets[bid]
#         hub_w = int(b_hub[bid])
#         q_bef, q_aft = 0, 0
#         for w in w_arr:
#             wt = w_ts_list[w]
#             wb = w_bid_list[w]
#             # Check events strictly outside this burst
#             lo = np.searchsorted(wt, t0 - 7200.0, side='left')
#             hi = np.searchsorted(wt, t0, side='right')
#             has_bef = np.any(wb[lo:hi] != bid)
#             if not has_bef:
#                 q_bef += 1
# 
#             lo_a = np.searchsorted(wt, t1, side='left')
#             hi_a = np.searchsorted(wt, t1 + 7200.0, side='right')
#             has_aft = np.any(wb[lo_a:hi_a] != bid)
#             if not has_aft:
#                 q_aft += 1
# 
#             if w == hub_w:
#                 out_mask = wb != bid
#                 if np.any(out_mask):
#                     wt_out = wt[out_mask]
#                     pos = np.searchsorted(wt_out, t0)
#                     g_min = 1e18
#                     if pos > 0:
#                         g_min = min(g_min, max(0.0, t0 - wt_out[pos - 1]))
#                     if pos < len(wt_out):
#                         g_min = min(g_min, max(0.0, wt_out[pos] - t1))
#                     b_hub_min_gap_h[bid] = g_min / 3600.0
# 
#         nw = float(len(w_arr))
#         b_quiet_2h_bef[bid] = q_bef / nw
#         b_quiet_2h_aft[bid] = q_aft / nw
# 
#     # 5. Build 45-minute hub-centric episodes to catch slower relay episodes & delayed collector flow
#     src_counts = np.bincount(src_id, minlength=W)
#     w_Outs = np.split(np.argsort(src_id, kind='mergesort'), np.cumsum(src_counts)[:-1])
#     dst_counts = np.bincount(dst_id, minlength=W)
#     w_Ins = np.split(np.argsort(dst_id, kind='mergesort'), np.cumsum(dst_counts)[:-1])
# 
#     w_ep45 = {0: [[] for _ in range(W)], 1: [[] for _ in range(W)]}
#     for w in range(W):
#         for role, tx_arr, cp_arr, u_arr in (
#             (1, w_Outs[w], dst_id, u_sent),
#             (0, w_Ins[w], src_id, u_recv),
#         ):
#             if len(tx_arr) == 0:
#                 continue
#             t_arr = ts[tx_arr]
#             splits = np.where(np.diff(t_arr) > 2700.0)[0] + 1
#             for grp in np.split(tx_arr, splits):
#                 cps = set(int(c) for c in cp_arr[grp] if int(c) != w)
#                 w_ep45[role][w].append({
#                     't0': float(ts[grp[0]]),
#                     't1': float(ts[grp[-1]]),
#                     'tmid': 0.5 * float(ts[grp[0]] + ts[grp[-1]]),
#                     'usd': float(np.sum(u_arr[grp])),
#                     'cps': cps,
#                     'n_tx': len(grp),
#                 })
# 
#     b_ep45_same_cnt = np.zeros(B, dtype=np.float64)
#     b_ep45_max_jacc = np.zeros(B, dtype=np.float64)
#     b_cp_other_day = np.zeros(B, dtype=np.float64)
#     b_delayed_flow_sim = np.zeros(B, dtype=np.float64)
#     b_delayed_opp_cps = np.zeros(B, dtype=np.float64)
# 
#     for bid in range(B):
#         hub_w = int(b_hub[bid])
#         role = int(b_role[bid])
#         t0, t1, tmid = b_tstart[bid], b_tend[bid], b_tmid[bid]
#         u_self = b_usd[bid]
#         cp_self = b_cps[bid]
# 
#         # Same-role 45m episodes separated by >2h from this burst
#         same_eps = [
#             ep for ep in w_ep45[role][hub_w]
#             if abs(ep['tmid'] - tmid) > 7200.0 and (ep['n_tx'] >= 2 or len(ep['cps']) >= 2)
#         ]
#         b_ep45_same_cnt[bid] = float(len(same_eps))
#         if same_eps and cp_self:
#             b_ep45_max_jacc[bid] = max(
#                 len(cp_self & ep['cps']) / float(len(cp_self | ep['cps']))
#                 for ep in same_eps if len(cp_self | ep['cps']) > 0
#             )
# 
#         # Counterparty recurrence with hub_w outside +/- 2h
#         all_other_cps = set()
#         for r_k in (0, 1):
#             for ep in w_ep45[r_k][hub_w]:
#                 if ep['t1'] < t0 - 7200.0 or ep['t0'] > t1 + 7200.0:
#                     all_other_cps |= ep['cps']
#         if cp_self:
#             b_cp_other_day[bid] = len(cp_self & all_other_cps) / float(len(cp_self))
# 
#         # Opposite-role delayed collector flow (15m to 36h before sender-hub, or after receiver-hub)
#         opp_eps = w_ep45[1 - role][hub_w]
#         best_sim, best_opp_cp = 0.0, 0.0
#         sum_opp_usd = 0.0
#         union_opp_cps = set()
#         for ep in opp_eps:
#             dt = (t0 - ep['t1']) if role == 1 else (ep['t0'] - t1)
#             if 900.0 <= dt <= 129600.0:
#                 sim = min(u_self, ep['usd']) / max(u_self, ep['usd'], 1e-6)
#                 if sim > best_sim:
#                     best_sim = sim
#                     best_opp_cp = float(len(ep['cps']))
#                 sum_opp_usd += ep['usd']
#                 union_opp_cps |= ep['cps']
#         if sum_opp_usd > 0:
#             agg_sim = min(u_self, sum_opp_usd) / max(u_self, sum_opp_usd, 1e-6)
#             if agg_sim > best_sim:
#                 best_sim = agg_sim
#                 best_opp_cp = float(len(union_opp_cps))
#         b_delayed_flow_sim[bid] = best_sim
#         b_delayed_opp_cps[bid] = best_opp_cp
# 
#     # Composite relay-burst false-alarm trap (high when burst is a star with repeating/collector hub)
#     b_relay_trap = b_star_purity * (
#         0.30 * np.tanh(b_same_cnt / 2.0)
#         + 0.35 * np.tanh(b_ep45_same_cnt / 2.0)
#         + 0.45 * b_cp_other_day
#         + 0.45 * b_delayed_flow_sim * np.tanh(b_delayed_opp_cps / 2.0)
#         + 0.30 * np.maximum(b_max_jacc, b_ep45_max_jacc)
#     )
# 
#     tx_bid = burst_id[inv_order]
#     out = pd.DataFrame(index=orig_index)
#     # 1. Count of other 5m bursts (size>=2) in the week with the same hub and role
#     out['rp_same_hub_role_cnt'] = b_same_cnt[tx_bid]
#     # 2. Max counterparty Jaccard overlap with other 5m bursts of the same hub and role
#     out['rp_max_cp_jaccard'] = b_max_jacc[tx_bid]
#     # 3. Median spacing in hours to other 5m bursts with the same hub and role (-1 if none)
#     out['rp_med_spacing_h'] = b_med_space_h[tx_bid]
#     # 4. Mean burst size similarity (min/max) with other 5m bursts of same hub and role
#     out['rp_size_sim'] = b_size_sim[tx_bid]
#     # 5. Mean burst total-USD similarity (min/max) with other 5m bursts of same hub and role
#     out['rp_usd_sim'] = b_usd_sim[tx_bid]
#     # 6. Fraction of burst wallets with zero outside-burst activity in the 2h before burst
#     out['rp_quiet_2h_before'] = b_quiet_2h_bef[tx_bid]
#     # 7. Fraction of burst wallets with zero outside-burst activity in the 2h after burst
#     out['rp_quiet_2h_after'] = b_quiet_2h_aft[tx_bid]
#     # 8. Minimum time gap in hours from the burst hub wallet to nearest outside-burst tx
#     out['rp_hub_min_out_gap_h'] = b_hub_min_gap_h[tx_bid]
#     # 9. Count of other 45m multi-counterparty episodes (>2h away) on the same hub and role
#     out['rp_ep45_same_hub_role_cnt'] = b_ep45_same_cnt[tx_bid]
#     # 10. Max counterparty Jaccard with other 45m episodes on the same hub and role
#     out['rp_ep45_max_cp_jaccard'] = b_ep45_max_jacc[tx_bid]
#     # 11. Share of burst counterparties that transact with the hub >2h outside this burst
#     out['rp_cp_other_day_share'] = b_cp_other_day[tx_bid]
#     # 12. USD similarity between burst total and hub's opposite-role collector flow 15m-36h away
#     out['rp_hub_delayed_flow_sim'] = b_delayed_flow_sim[tx_bid]
#     # 13. Distinct opposite-role counterparties funding/draining the hub 15m-36h away
#     out['rp_hub_delayed_opp_cp_cnt'] = b_delayed_opp_cps[tx_bid]
#     # 14. Star topology purity of the 5m burst (1.0 = pure hub star, lower = multi-hop motif)
#     out['rp_burst_star_purity'] = b_star_purity[tx_bid]
#     # 15. Composite relay-burst false-alarm trap score targeting aggregate/distributed relays
#     out['rp_relay_burst_trap'] = b_relay_trap[tx_bid]
# 
#     return out.astype(np.float64)
# 
# 
# Innocent aggregate_relay payout bursts and distributed_relay collection bursts mimic suspicious fan_out (conv ~0.64 vs 0.50, USD ~116 vs 99) and fan_in (conv 0.00, USD ~201 vs 214), and because innocent relay payouts have a 50-minute median duration, a collector's other 2–3 weekly episodes often exceed 5 minutes; combining 5m and 45m same-hub-role episode repetition (rp_same_hub_role_cnt, rp_ep45_same_hub_role_cnt), multi-day counterparty recurrence (rp_cp_other_day_share), and 15m–36h opposite-role collector funding matching (rp_hub_delayed_flow_sim) will directly separate them from one-off motifs. In your pipeline, consider interaction splits between rp_relay_burst_trap and your fan_out/fan_in burst features so trees can specifically discount pure-star bursts whose hub funded itself via a multi-sender collection episode hours earlier.