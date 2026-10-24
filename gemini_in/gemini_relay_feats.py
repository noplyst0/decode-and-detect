#CODE
import numpy as np
import pandas as pd


def relay_features(e: pd.DataFrame) -> pd.DataFrame:
    orig_index = e.index
    n = len(e)
    if n == 0:
        return pd.DataFrame(index=orig_index)

    # Deterministic chronological ordering
    order = np.lexsort((np.arange(n, dtype=np.int32), e['TS'].values))
    inv_order = np.empty(n, dtype=np.int32)
    inv_order[order] = np.arange(n, dtype=np.int32)

    ts = e['TS'].values[order].astype(np.float64)
    u_sent = e['USD_SENT'].values[order].astype(np.float64)
    u_recv = e['USD_RECV'].values[order].astype(np.float64)
    v_sent = e['V_SENT'].values[order].astype(np.float64)
    is_conv = e['IS_CONV'].values[order].astype(np.float64)

    # Integer-encode wallets and assets
    src_raw = e['SRC'].values[order]
    dst_raw = e['DST'].values[order]
    wallets, w_inv = np.unique(np.concatenate([src_raw, dst_raw]), return_inverse=True)
    W = len(wallets)
    src_id = w_inv[:n].astype(np.int32)
    dst_id = w_inv[n:].astype(np.int32)

    a_sent_raw = e['ASSET_SENT'].astype(str).values[order]
    a_recv_raw = e['ASSET_RECV'].astype(str).values[order]
    _, a_inv = np.unique(np.concatenate([a_sent_raw, a_recv_raw]), return_inverse=True)
    s_asset_id = a_inv[:n].astype(np.int32)
    r_asset_id = a_inv[n:].astype(np.int32)

    # Global weekly transaction count per wallet and round-amount indicator
    w_total_tx = (
        np.bincount(src_id, minlength=W) + np.bincount(dst_id, minlength=W)
    ).astype(np.float64)
    is_round = (
        (np.abs(u_sent - np.round(u_sent)) < 0.02)
        | (np.abs(u_sent / 5.0 - np.round(u_sent / 5.0)) * 5.0 < 0.03)
        | (np.abs(v_sent - np.round(v_sent)) < 0.01)
    ).astype(np.float64)

    # Per-wallet inbound and outbound chronologically sorted transaction lists
    src_counts = np.bincount(src_id, minlength=W)
    w_Outs = np.split(np.argsort(src_id, kind='mergesort'), np.cumsum(src_counts)[:-1])
    dst_counts = np.bincount(dst_id, minlength=W)
    w_Ins = np.split(np.argsort(dst_id, kind='mergesort'), np.cumsum(dst_counts)[:-1])

    # Step 1: Flag obvious fast-motif 1-to-1 pass-throughs (<= 25 min, ratio 0.84..1.02)
    is_fast_in = np.zeros(n, dtype=bool)
    is_fast_out = np.zeros(n, dtype=bool)
    for w in range(W):
        ins, outs = w_Ins[w], w_Outs[w]
        if len(ins) == 0 or len(outs) == 0:
            continue
        t_in, t_out = ts[ins], ts[outs]
        u_in, u_out = u_recv[ins], u_sent[outs]
        used_o = np.zeros(len(outs), dtype=bool)
        for a_i, tx_i in enumerate(ins):
            lo = np.searchsorted(t_out, t_in[a_i], side='left')
            hi = np.searchsorted(t_out, t_in[a_i] + 1500.0, side='right')
            best_b, best_err = -1, 1e9
            for b_i in range(lo, hi):
                if used_o[b_i]:
                    continue
                rat = u_out[b_i] / (u_in[a_i] + 1e-6)
                if 0.84 <= rat <= 1.02:
                    err = abs(rat - 0.96) + (t_out[b_i] - t_in[a_i]) / 15000.0
                    if err < best_err:
                        best_err, best_b = err, b_i
            if best_b != -1:
                is_fast_in[tx_i] = True
                is_fast_out[outs[best_b]] = True
                used_o[best_b] = True

    # Step 2: Pre-detect same-asset 2-hop (and 3-hop) distributed relay chains (0.4h - 5.5h)
    tx_has_hop2 = np.zeros(n, dtype=bool)
    tx_hop2_delay_m = np.full(n, np.nan, dtype=np.float64)
    tx_hop2_ratio = np.full(n, np.nan, dtype=np.float64)
    tx_hop2_children = [None] * n
    is_relay_child = np.zeros(n, dtype=bool)

    for w in range(W):
        ins, outs = w_Ins[w], w_Outs[w]
        if len(ins) == 0 or len(outs) == 0:
            continue
        valid_o = outs[(~is_fast_out[outs]) & (is_conv[outs] == 0)]
        if len(valid_o) == 0:
            continue
        t_vo = ts[valid_o]
        u_vo = u_sent[valid_o]
        a_vo = s_asset_id[valid_o]

        for tx_p in ins:
            if is_fast_in[tx_p] or is_conv[tx_p] > 0.5:
                continue
            t_p = ts[tx_p]
            u_p = u_recv[tx_p]
            a_p = r_asset_id[tx_p]
            lo = np.searchsorted(t_vo, t_p + 1200.0, side='left')
            hi = np.searchsorted(t_vo, t_p + 19800.0, side='right')
            if lo >= hi:
                continue
            cand = valid_o[lo:hi]
            cand = cand[a_vo[lo:hi] == a_p]
            if len(cand) == 0:
                continue

            # Group candidate same-asset outbound txs on relay wallet W into 40m bursts
            c_ts = ts[cand]
            splits = np.where(np.diff(c_ts) > 2400.0)[0] + 1
            bursts = np.split(cand, splits)
            best_b, best_score, best_rat, best_dt_m = None, 1e9, np.nan, np.nan
            for b in bursts:
                if ts[b[-1]] - ts[b[0]] > 4800.0:
                    b = b[ts[b] - ts[b[0]] <= 4800.0]
                u_b = float(np.sum(u_sent[b]))
                rat = u_b / (u_p + 1e-6)
                if 0.68 <= rat <= 1.08:
                    dt_m = (ts[b[0]] - t_p) / 60.0
                    n_rec = len(np.unique(dst_id[b]))
                    sc = abs(rat - 0.93) + 0.04 * abs(dt_m - 122.0) / 120.0 - 0.08 * (n_rec >= 2)
                    if sc < best_score:
                        best_score, best_b, best_rat, best_dt_m = sc, b, rat, dt_m
            if best_b is not None:
                tx_has_hop2[tx_p] = True
                tx_hop2_delay_m[tx_p] = best_dt_m
                tx_hop2_ratio[tx_p] = best_rat
                tx_hop2_children[tx_p] = best_b
                is_relay_child[best_b] = True

    # Extend 2-hop chains to 3-hop if a single-tx 2nd hop itself triggers a verified hop
    for tx_p in np.flatnonzero(tx_has_hop2):
        ch = tx_hop2_children[tx_p]
        if len(ch) == 1 and tx_has_hop2[ch[0]]:
            ch2 = tx_hop2_children[ch[0]]
            tx_hop2_children[tx_p] = np.concatenate([ch, ch2])

    # Step 3: Build inbound collection clusters per wallet (>=2 senders within <=105 min)
    is_coll_in_tx = np.zeros(n, dtype=bool)
    w_clusters = [[] for _ in range(W)]

    for w in range(W):
        ins = w_Ins[w]
        outs = w_Outs[w]
        cand_in = ins[(~is_fast_in[ins]) & (~tx_has_hop2[ins])]
        if len(cand_in) < 2:
            continue
        t_in = ts[cand_in]
        splits = np.where(np.diff(t_in) > 4500.0)[0] + 1
        raw_groups = np.split(cand_in, splits)

        t_out = ts[outs] if len(outs) > 0 else np.empty(0)
        u_out = u_sent[outs] if len(outs) > 0 else np.empty(0)

        for g in raw_groups:
            if len(g) < 2:
                continue
            # Sub-split if cluster duration exceeds 105 minutes
            if ts[g[-1]] - ts[g[0]] > 6300.0:
                idx_max = int(np.argmax(np.diff(ts[g]))) + 1
                sub_gs = [g[:idx_max], g[idx_max:]]
            else:
                sub_gs = [g]

            for sg in sub_gs:
                if len(sg) < 2:
                    continue
                snds = np.unique(src_id[sg])
                if len(snds) < 2:
                    continue
                t_s, t_e = float(ts[sg[0]]), float(ts[sg[-1]])
                u_in_sum = float(np.sum(u_recv[sg]))

                # Skip fast fan_in/fan_out_in motifs that clear within < 35 minutes
                if len(t_out) > 0:
                    lo_f = np.searchsorted(t_out, t_s, side='left')
                    hi_f = np.searchsorted(t_out, t_e + 2100.0, side='right')
                    if hi_f > lo_f:
                        u_fast = float(np.sum(u_out[lo_f:hi_f]))
                        if 0.75 <= u_fast / (u_in_sum + 1e-6) <= 1.12:
                            continue

                is_coll_in_tx[sg] = True
                w_clusters[w].append({
                    'tx_in': sg,
                    't_start': t_s,
                    't_end': t_e,
                    'u_in': u_in_sum,
                    'senders': set(snds.tolist()),
                    'conv_in': float(np.mean(is_conv[sg])),
                })

    # Step 4: Match each wallet's inbound collection clusters to outbound payout bursts (1h - 42h)
    episodes = []
    for w in range(W):
        clusters = w_clusters[w]
        if len(clusters) < 2:
            continue
        outs = w_Outs[w]
        cand_out = outs[(~is_fast_out[outs]) & (~is_relay_child[outs])]
        if len(cand_out) == 0:
            continue

        # Form candidate outbound bursts on wallet W (gap <= 40m, span <= 80m)
        t_co = ts[cand_out]
        o_splits = np.where(np.diff(t_co) > 2400.0)[0] + 1
        raw_obursts = np.split(cand_out, o_splits)
        obursts = []
        for ob in raw_obursts:
            if ts[ob[-1]] - ts[ob[0]] > 4800.0:
                idx_m = int(np.argmax(np.diff(ts[ob]))) + 1
                ob_list = [ob[:idx_m], ob[idx_m:]]
            else:
                ob_list = [ob]
            for b in ob_list:
                if len(b) == 0:
                    continue
                has_h2 = bool(np.any(tx_has_hop2[b]))
                # If any tx in burst has verified 2-hop, also allow that single tx as a clean candidate
                if has_h2 and len(b) > 1:
                    for single_tx in b[tx_has_hop2[b]]:
                        obursts.append(np.array([single_tx], dtype=np.int32))
                obursts.append(b)

        n_c = len(clusters)
        n_b = len(obursts)
        if n_b == 0:
            continue

        b_tstart = np.array([float(ts[b[0]]) for b in obursts])
        b_tend = np.array([float(ts[b[-1]]) for b in obursts])
        b_uout = np.array([float(np.sum(u_sent[b])) for b in obursts])
        b_ndst = np.array([len(np.unique(dst_id[b])) for b in obursts])
        b_h2 = np.array([bool(np.any(tx_has_hop2[b])) for b in obursts])
        b_is_coll_send = np.array(
            [(len(b) == 1) and (not np.any(tx_has_hop2[b])) and bool(is_coll_in_tx[b[0]]) for b in obursts]
        )

        # Score candidate (cluster, outbound_burst) pairs
        pair_candidates = []
        for k, cl in enumerate(clusters):
            t_e = cl['t_end']
            u_in = cl['u_in']
            next_te = clusters[k + 1]['t_end'] if (k + 1 < n_c) else (t_e + 172800.0)
            for m in range(n_b):
                dt_h = (b_tstart[m] - t_e) / 3600.0
                if dt_h < 0.85 or dt_h > 42.0:
                    continue
                rat = b_uout[m] / (u_in + 1e-6)
                if rat < 0.38 or rat > 2.40:
                    continue
                cost = abs(np.log((rat + 1e-3) / 0.90))
                if b_tstart[m] > next_te:
                    cost += 0.45
                if b_is_coll_send[m]:
                    cost += 0.50
                if b_h2[m]:
                    cost -= 0.30
                elif b_ndst[m] >= 2:
                    cost -= 0.18
                cost += 0.08 * (dt_h / 24.0)
                pair_candidates.append((cost, k, m, dt_h, rat))

        if len(pair_candidates) < 2:
            continue
        pair_candidates.sort(key=lambda x: x[0])
        used_c = set()
        used_b_first_tx = set()
        w_eps = []

        for cost, k, m, dt_h, rat in pair_candidates:
            first_tx = int(obursts[m][0])
            if k in used_c or first_tx in used_b_first_tx:
                continue
            used_c.add(k)
            used_b_first_tx.add(first_tx)
            cl = clusters[k]
            b = obursts[m]

            # Gather 2-hop children and final receiver set if distributed relay
            h2_txs = np.flatnonzero(tx_has_hop2[b])
            if len(h2_txs) > 0:
                p_tx = b[h2_txs[0]]
                ch_txs = tx_hop2_children[p_tx]
                hop2_m = float(tx_hop2_delay_m[p_tx])
                sink_txs = ch_txs
                all_out_txs = np.unique(np.concatenate([b, ch_txs]))
                recvs = set(dst_id[ch_txs].tolist())
                pay_dur_m = float((ts[ch_txs[-1]] - ts[ch_txs[0]]) / 60.0)
            else:
                hop2_m = 0.0
                sink_txs = b
                all_out_txs = b
                recvs = set(dst_id[b].tolist())
                pay_dur_m = float((ts[b[-1]] - ts[b[0]]) / 60.0)

            all_ep_txs = np.unique(np.concatenate([cl['tx_in'], all_out_txs]))
            w_eps.append({
                'collector': w,
                't_start': cl['t_start'],
                't_end': cl['t_end'],
                'u_in': cl['u_in'],
                'hold_h': float(dt_h),
                'pass_ratio': float(rat),
                'hop2_m': hop2_m,
                'has_hop2': float(len(h2_txs) > 0),
                'pay_dur_m': pay_dur_m,
                'conv_rate': float(np.mean(is_conv[all_ep_txs])),
                'senders': cl['senders'],
                'recvs': recvs,
                'u_outs': u_sent[sink_txs],
                'tx_in': cl['tx_in'],
                'tx_out': all_out_txs,
                'all_txs': all_ep_txs,
            })

        if len(w_eps) < 2:
            continue
        w_eps.sort(key=lambda x: x['t_start'])

        # Compute full-wallet episode stats across all episodes on collector W
        E = len(w_eps)
        all_holds = np.array([ep['hold_h'] for ep in w_eps])
        all_pass = np.array([ep['pass_ratio'] for ep in w_eps])
        wal_hold_cv = float(np.std(all_holds) / (np.mean(all_holds) + 1e-3))
        wal_pass_std = float(np.std(all_pass))

        # Step 5: For each episode on W, select coherent scenario cohort (2..4 episodes >=6h apart)
        for idx_e, ep in enumerate(w_eps):
            if E <= 4:
                cohort = w_eps
            else:
                # Rank other episodes by hold-delay & relay-type similarity, enforcing >=6h spacing
                scored = []
                for idx_o, ep_o in enumerate(w_eps):
                    if idx_o == idx_e:
                        continue
                    d_diff = abs(ep_o['hold_h'] - ep['hold_h']) / max(ep['hold_h'], ep_o['hold_h'], 1.0)
                    t_diff = 0.5 * abs(ep_o['has_hop2'] - ep['has_hop2']) + 0.3 * abs(
                        ep_o['conv_rate'] - ep['conv_rate']
                    )
                    scored.append((d_diff + t_diff, ep_o))
                scored.sort(key=lambda x: x[0])
                cohort = [ep]
                for _, cand_ep in scored:
                    if all(abs(cand_ep['t_start'] - c_ep['t_start']) >= 21600.0 for c_ep in cohort):
                        cohort.append(cand_ep)
                        if len(cohort) == 4:
                            break
                if len(cohort) < 2:
                    cohort = w_eps[:4]
                cohort.sort(key=lambda x: x['t_start'])

            c_holds = np.array([c['hold_h'] for c in cohort])
            c_pass = np.array([c['pass_ratio'] for c in cohort])
            c_uins = np.array([c['u_in'] for c in cohort])
            c_pdur = np.array([c['pay_dur_m'] for c in cohort])
            c_hop2 = np.array([c['hop2_m'] for c in cohort])
            c_uouts = np.concatenate([c['u_outs'] for c in cohort])
            c_txs = np.unique(np.concatenate([c['all_txs'] for c in cohort]))

            scen_hold_med = float(np.median(c_holds))
            scen_hold_cv = float(np.std(c_holds) / (np.mean(c_holds) + 1e-3))
            scen_pass_med = float(np.median(c_pass))
            scen_pass_std = float(np.std(c_pass))
            scen_hop2_med = float(np.median(c_hop2))
            scen_pdur_med = float(np.median(c_pdur))
            scen_uout_cv = (
                float(np.std(c_uouts) / (np.mean(c_uouts) + 1e-3)) if len(c_uouts) >= 2 else 0.0
            )
            scen_uin_cv = float(np.std(c_uins) / (np.mean(c_uins) + 1e-3))

            jacc_l = [
                len(cohort[j]['recvs'] & cohort[j + 1]['recvs'])
                / max(1, len(cohort[j]['recvs'] | cohort[j + 1]['recvs']))
                for j in range(len(cohort) - 1)
            ]
            scen_recv_jacc = float(np.mean(jacc_l)) if jacc_l else 0.0

            peer_w = list(ep['senders'] | ep['recvs'])
            peer_bg_med = float(np.median(w_total_tx[peer_w])) if peer_w else np.nan
            scen_round = float(np.mean(is_round[c_txs]))
            coll_tot = float(w_total_tx[w])

            # Unified scripted-relay score pooling Aggregate (~4.33h) and Distributed (~2.08h + ~122m hop)
            d_agg = abs(scen_hold_med - 4.35) / 0.55
            d_dist = abs(scen_hold_med - 2.08) / 0.35 + 0.7 * abs(scen_hop2_med - 122.0) / 25.0
            band_closeness = np.maximum(np.exp(-0.5 * d_agg * d_agg), np.exp(-0.5 * d_dist * d_dist))
            scripted_score = (
                2.2 * np.exp(-scen_hold_cv / 0.18)
                + 1.8 * np.exp(-scen_pass_std / 0.30)
                + 1.6 * band_closeness
                + 0.6 * np.tanh((scen_uout_cv - 0.70) / 0.30)
                - 0.5 * np.tanh((coll_tot - 105.0) / 20.0)
            )

            ep['stats'] = (
                ep['hold_h'],
                scen_hold_med,
                scen_hold_cv,
                wal_hold_cv,
                ep['pass_ratio'],
                scen_pass_med,
                scen_pass_std,
                wal_pass_std,
                ep['hop2_m'] if ep['hop2_m'] > 0 else (
                    float(tx_hop2_delay_m[ep['tx_out'][0]])
                    if (len(ep['tx_out']) > 0 and tx_has_hop2[ep['tx_out'][0]])
                    else 0.0
                ),
                scen_hop2_med,
                scen_pdur_med,
                scen_uout_cv,
                scen_uin_cv,
                scen_recv_jacc,
                coll_tot,
                peer_bg_med,
                scen_round,
                float(scripted_score),
            )
            ep['reg_priority'] = scen_hold_cv + 0.5 * scen_pass_std - 0.15 * band_closeness
            episodes.append(ep)

    # Map reconstructed episode statistics onto every transaction (preferring most coherent episode on ties)
    n_feats = 18
    feat_mat = np.full((n, n_feats), np.nan, dtype=np.float64)
    best_prio = np.full(n, 1e9, dtype=np.float64)

    for ep in episodes:
        pr = ep['reg_priority']
        st = ep['stats']
        for idx_tx in ep['all_txs']:
            if pr < best_prio[idx_tx]:
                best_prio[idx_tx] = pr
                feat_mat[idx_tx] = st

    # Populate standalone 2-hop relay delay if a 2-hop chain was found outside a >=2 episode cluster
    for tx_p in np.flatnonzero(tx_has_hop2):
        if np.isnan(feat_mat[tx_p, 8]):
            feat_mat[tx_p, 8] = tx_hop2_delay_m[tx_p]
        for ch_tx in tx_hop2_children[tx_p]:
            if np.isnan(feat_mat[ch_tx, 8]):
                feat_mat[ch_tx, 8] = tx_hop2_delay_m[tx_p]

    feat_mat = feat_mat[inv_order]
    out = pd.DataFrame(index=orig_index)
    # 1. Episode hold delay in hours from collection cluster end to first payout
    out['rl_ep_hold_delay_h'] = feat_mat[:, 0]
    # 2. Median hold delay in hours across the reconstructed scenario's episodes (4.33h agg / 2.08h dist)
    out['rl_scen_hold_med_h'] = feat_mat[:, 1]
    # 3. CV of hold delays across the scenario's episodes (AUC 0.175-0.230: suspicious ~0.08 vs 0.71)
    out['rl_scen_hold_cv'] = feat_mat[:, 2]
    # 4. Un-subsetted wallet-wide CV of hold delays across all episodes on the collector wallet
    out['rl_wal_hold_cv'] = feat_mat[:, 3]
    # 5. Episode outbound-to-inbound USD pass-through ratio
    out['rl_ep_pass_ratio'] = feat_mat[:, 4]
    # 6. Median per-episode USD pass-through ratio across the scenario's episodes (AUC 0.260 in agg)
    out['rl_scen_pass_med'] = feat_mat[:, 5]
    # 7. Std of per-episode USD pass-through ratios across the scenario (AUC 0.109 in agg: 0.20 vs 0.64)
    out['rl_scen_pass_std'] = feat_mat[:, 6]
    # 8. Un-subsetted wallet-wide std of per-episode USD pass-through ratios on the collector wallet
    out['rl_wal_pass_std'] = feat_mat[:, 7]
    # 9. 2-hop relay delay in minutes from collector->relay to relay->receivers (122m in dist)
    out['rl_hop2_delay_m'] = feat_mat[:, 8]
    # 10. Median 2-hop relay delay in minutes across the scenario's episodes (AUC 0.679 in dist)
    out['rl_scen_hop2_med_m'] = feat_mat[:, 9]
    # 11. Median payout burst duration in minutes across the scenario's episodes (23m vs 50m in agg)
    out['rl_scen_pay_dur_med_m'] = feat_mat[:, 10]
    # 12. CV of outbound payout USD amounts across the scenario (AUC 0.772 in agg: 0.97 vs 0.65)
    out['rl_scen_usd_out_cv'] = feat_mat[:, 11]
    # 13. CV of total inbound USD per episode across the scenario (AUC 0.682 in dist: 0.71 vs 0.54)
    out['rl_scen_in_usd_cv'] = feat_mat[:, 12]
    # 14. Mean Jaccard overlap of receiver sets across consecutive episodes (AUC 0.293 in agg)
    out['rl_scen_recv_jacc'] = feat_mat[:, 13]
    # 15. Total weekly transaction count of the scenario's collector wallet (AUC 0.184 in dist)
    out['rl_coll_total_tx'] = feat_mat[:, 14]
    # 16. Median weekly transaction count of the episode's sender and receiver peer wallets
    out['rl_peer_bg_med_tx'] = feat_mat[:, 15]
    # 17. Share of round-USD or round-native-unit transactions in the scenario (AUC 0.752 in dist)
    out['rl_scen_round_share'] = feat_mat[:, 16]
    # 18. Unified scripted-relay log-odds score pooling both relay families above min_child_samples=60
    out['rl_scripted_relay_score'] = feat_mat[:, 17]

    return out.astype(np.float64)


#FULLTEXT
# Gemini said
# Python
# import numpy as np
# import pandas as pd
# 
# 
# def relay_features(e: pd.DataFrame) -> pd.DataFrame:
#     orig_index = e.index
#     n = len(e)
#     if n == 0:
#         return pd.DataFrame(index=orig_index)
# 
#     # Deterministic chronological ordering
#     order = np.lexsort((np.arange(n, dtype=np.int32), e['TS'].values))
#     inv_order = np.empty(n, dtype=np.int32)
#     inv_order[order] = np.arange(n, dtype=np.int32)
# 
#     ts = e['TS'].values[order].astype(np.float64)
#     u_sent = e['USD_SENT'].values[order].astype(np.float64)
#     u_recv = e['USD_RECV'].values[order].astype(np.float64)
#     v_sent = e['V_SENT'].values[order].astype(np.float64)
#     is_conv = e['IS_CONV'].values[order].astype(np.float64)
# 
#     # Integer-encode wallets and assets
#     src_raw = e['SRC'].values[order]
#     dst_raw = e['DST'].values[order]
#     wallets, w_inv = np.unique(np.concatenate([src_raw, dst_raw]), return_inverse=True)
#     W = len(wallets)
#     src_id = w_inv[:n].astype(np.int32)
#     dst_id = w_inv[n:].astype(np.int32)
# 
#     a_sent_raw = e['ASSET_SENT'].astype(str).values[order]
#     a_recv_raw = e['ASSET_RECV'].astype(str).values[order]
#     _, a_inv = np.unique(np.concatenate([a_sent_raw, a_recv_raw]), return_inverse=True)
#     s_asset_id = a_inv[:n].astype(np.int32)
#     r_asset_id = a_inv[n:].astype(np.int32)
# 
#     # Global weekly transaction count per wallet and round-amount indicator
#     w_total_tx = (
#         np.bincount(src_id, minlength=W) + np.bincount(dst_id, minlength=W)
#     ).astype(np.float64)
#     is_round = (
#         (np.abs(u_sent - np.round(u_sent)) < 0.02)
#         | (np.abs(u_sent / 5.0 - np.round(u_sent / 5.0)) * 5.0 < 0.03)
#         | (np.abs(v_sent - np.round(v_sent)) < 0.01)
#     ).astype(np.float64)
# 
#     # Per-wallet inbound and outbound chronologically sorted transaction lists
#     src_counts = np.bincount(src_id, minlength=W)
#     w_Outs = np.split(np.argsort(src_id, kind='mergesort'), np.cumsum(src_counts)[:-1])
#     dst_counts = np.bincount(dst_id, minlength=W)
#     w_Ins = np.split(np.argsort(dst_id, kind='mergesort'), np.cumsum(dst_counts)[:-1])
# 
#     # Step 1: Flag obvious fast-motif 1-to-1 pass-throughs (<= 25 min, ratio 0.84..1.02)
#     is_fast_in = np.zeros(n, dtype=bool)
#     is_fast_out = np.zeros(n, dtype=bool)
#     for w in range(W):
#         ins, outs = w_Ins[w], w_Outs[w]
#         if len(ins) == 0 or len(outs) == 0:
#             continue
#         t_in, t_out = ts[ins], ts[outs]
#         u_in, u_out = u_recv[ins], u_sent[outs]
#         used_o = np.zeros(len(outs), dtype=bool)
#         for a_i, tx_i in enumerate(ins):
#             lo = np.searchsorted(t_out, t_in[a_i], side='left')
#             hi = np.searchsorted(t_out, t_in[a_i] + 1500.0, side='right')
#             best_b, best_err = -1, 1e9
#             for b_i in range(lo, hi):
#                 if used_o[b_i]:
#                     continue
#                 rat = u_out[b_i] / (u_in[a_i] + 1e-6)
#                 if 0.84 <= rat <= 1.02:
#                     err = abs(rat - 0.96) + (t_out[b_i] - t_in[a_i]) / 15000.0
#                     if err < best_err:
#                         best_err, best_b = err, b_i
#             if best_b != -1:
#                 is_fast_in[tx_i] = True
#                 is_fast_out[outs[best_b]] = True
#                 used_o[best_b] = True
# 
#     # Step 2: Pre-detect same-asset 2-hop (and 3-hop) distributed relay chains (0.4h - 5.5h)
#     tx_has_hop2 = np.zeros(n, dtype=bool)
#     tx_hop2_delay_m = np.full(n, np.nan, dtype=np.float64)
#     tx_hop2_ratio = np.full(n, np.nan, dtype=np.float64)
#     tx_hop2_children = [None] * n
#     is_relay_child = np.zeros(n, dtype=bool)
# 
#     for w in range(W):
#         ins, outs = w_Ins[w], w_Outs[w]
#         if len(ins) == 0 or len(outs) == 0:
#             continue
#         valid_o = outs[(~is_fast_out[outs]) & (is_conv[outs] == 0)]
#         if len(valid_o) == 0:
#             continue
#         t_vo = ts[valid_o]
#         u_vo = u_sent[valid_o]
#         a_vo = s_asset_id[valid_o]
# 
#         for tx_p in ins:
#             if is_fast_in[tx_p] or is_conv[tx_p] > 0.5:
#                 continue
#             t_p = ts[tx_p]
#             u_p = u_recv[tx_p]
#             a_p = r_asset_id[tx_p]
#             lo = np.searchsorted(t_vo, t_p + 1200.0, side='left')
#             hi = np.searchsorted(t_vo, t_p + 19800.0, side='right')
#             if lo >= hi:
#                 continue
#             cand = valid_o[lo:hi]
#             cand = cand[a_vo[lo:hi] == a_p]
#             if len(cand) == 0:
#                 continue
# 
#             # Group candidate same-asset outbound txs on relay wallet W into 40m bursts
#             c_ts = ts[cand]
#             splits = np.where(np.diff(c_ts) > 2400.0)[0] + 1
#             bursts = np.split(cand, splits)
#             best_b, best_score, best_rat, best_dt_m = None, 1e9, np.nan, np.nan
#             for b in bursts:
#                 if ts[b[-1]] - ts[b[0]] > 4800.0:
#                     b = b[ts[b] - ts[b[0]] <= 4800.0]
#                 u_b = float(np.sum(u_sent[b]))
#                 rat = u_b / (u_p + 1e-6)
#                 if 0.68 <= rat <= 1.08:
#                     dt_m = (ts[b[0]] - t_p) / 60.0
#                     n_rec = len(np.unique(dst_id[b]))
#                     sc = abs(rat - 0.93) + 0.04 * abs(dt_m - 122.0) / 120.0 - 0.08 * (n_rec >= 2)
#                     if sc < best_score:
#                         best_score, best_b, best_rat, best_dt_m = sc, b, rat, dt_m
#             if best_b is not None:
#                 tx_has_hop2[tx_p] = True
#                 tx_hop2_delay_m[tx_p] = best_dt_m
#                 tx_hop2_ratio[tx_p] = best_rat
#                 tx_hop2_children[tx_p] = best_b
#                 is_relay_child[best_b] = True
# 
#     # Extend 2-hop chains to 3-hop if a single-tx 2nd hop itself triggers a verified hop
#     for tx_p in np.flatnonzero(tx_has_hop2):
#         ch = tx_hop2_children[tx_p]
#         if len(ch) == 1 and tx_has_hop2[ch[0]]:
#             ch2 = tx_hop2_children[ch[0]]
#             tx_hop2_children[tx_p] = np.concatenate([ch, ch2])
# 
#     # Step 3: Build inbound collection clusters per wallet (>=2 senders within <=105 min)
#     is_coll_in_tx = np.zeros(n, dtype=bool)
#     w_clusters = [[] for _ in range(W)]
# 
#     for w in range(W):
#         ins = w_Ins[w]
#         outs = w_Outs[w]
#         cand_in = ins[(~is_fast_in[ins]) & (~tx_has_hop2[ins])]
#         if len(cand_in) < 2:
#             continue
#         t_in = ts[cand_in]
#         splits = np.where(np.diff(t_in) > 4500.0)[0] + 1
#         raw_groups = np.split(cand_in, splits)
# 
#         t_out = ts[outs] if len(outs) > 0 else np.empty(0)
#         u_out = u_sent[outs] if len(outs) > 0 else np.empty(0)
# 
#         for g in raw_groups:
#             if len(g) < 2:
#                 continue
#             # Sub-split if cluster duration exceeds 105 minutes
#             if ts[g[-1]] - ts[g[0]] > 6300.0:
#                 idx_max = int(np.argmax(np.diff(ts[g]))) + 1
#                 sub_gs = [g[:idx_max], g[idx_max:]]
#             else:
#                 sub_gs = [g]
# 
#             for sg in sub_gs:
#                 if len(sg) < 2:
#                     continue
#                 snds = np.unique(src_id[sg])
#                 if len(snds) < 2:
#                     continue
#                 t_s, t_e = float(ts[sg[0]]), float(ts[sg[-1]])
#                 u_in_sum = float(np.sum(u_recv[sg]))
# 
#                 # Skip fast fan_in/fan_out_in motifs that clear within < 35 minutes
#                 if len(t_out) > 0:
#                     lo_f = np.searchsorted(t_out, t_s, side='left')
#                     hi_f = np.searchsorted(t_out, t_e + 2100.0, side='right')
#                     if hi_f > lo_f:
#                         u_fast = float(np.sum(u_out[lo_f:hi_f]))
#                         if 0.75 <= u_fast / (u_in_sum + 1e-6) <= 1.12:
#                             continue
# 
#                 is_coll_in_tx[sg] = True
#                 w_clusters[w].append({
#                     'tx_in': sg,
#                     't_start': t_s,
#                     't_end': t_e,
#                     'u_in': u_in_sum,
#                     'senders': set(snds.tolist()),
#                     'conv_in': float(np.mean(is_conv[sg])),
#                 })
# 
#     # Step 4: Match each wallet's inbound collection clusters to outbound payout bursts (1h - 42h)
#     episodes = []
#     for w in range(W):
#         clusters = w_clusters[w]
#         if len(clusters) < 2:
#             continue
#         outs = w_Outs[w]
#         cand_out = outs[(~is_fast_out[outs]) & (~is_relay_child[outs])]
#         if len(cand_out) == 0:
#             continue
# 
#         # Form candidate outbound bursts on wallet W (gap <= 40m, span <= 80m)
#         t_co = ts[cand_out]
#         o_splits = np.where(np.diff(t_co) > 2400.0)[0] + 1
#         raw_obursts = np.split(cand_out, o_splits)
#         obursts = []
#         for ob in raw_obursts:
#             if ts[ob[-1]] - ts[ob[0]] > 4800.0:
#                 idx_m = int(np.argmax(np.diff(ts[ob]))) + 1
#                 ob_list = [ob[:idx_m], ob[idx_m:]]
#             else:
#                 ob_list = [ob]
#             for b in ob_list:
#                 if len(b) == 0:
#                     continue
#                 has_h2 = bool(np.any(tx_has_hop2[b]))
#                 # If any tx in burst has verified 2-hop, also allow that single tx as a clean candidate
#                 if has_h2 and len(b) > 1:
#                     for single_tx in b[tx_has_hop2[b]]:
#                         obursts.append(np.array([single_tx], dtype=np.int32))
#                 obursts.append(b)
# 
#         n_c = len(clusters)
#         n_b = len(obursts)
#         if n_b == 0:
#             continue
# 
#         b_tstart = np.array([float(ts[b[0]]) for b in obursts])
#         b_tend = np.array([float(ts[b[-1]]) for b in obursts])
#         b_uout = np.array([float(np.sum(u_sent[b])) for b in obursts])
#         b_ndst = np.array([len(np.unique(dst_id[b])) for b in obursts])
#         b_h2 = np.array([bool(np.any(tx_has_hop2[b])) for b in obursts])
#         b_is_coll_send = np.array(
#             [(len(b) == 1) and (not np.any(tx_has_hop2[b])) and bool(is_coll_in_tx[b[0]]) for b in obursts]
#         )
# 
#         # Score candidate (cluster, outbound_burst) pairs
#         pair_candidates = []
#         for k, cl in enumerate(clusters):
#             t_e = cl['t_end']
#             u_in = cl['u_in']
#             next_te = clusters[k + 1]['t_end'] if (k + 1 < n_c) else (t_e + 172800.0)
#             for m in range(n_b):
#                 dt_h = (b_tstart[m] - t_e) / 3600.0
#                 if dt_h < 0.85 or dt_h > 42.0:
#                     continue
#                 rat = b_uout[m] / (u_in + 1e-6)
#                 if rat < 0.38 or rat > 2.40:
#                     continue
#                 cost = abs(np.log((rat + 1e-3) / 0.90))
#                 if b_tstart[m] > next_te:
#                     cost += 0.45
#                 if b_is_coll_send[m]:
#                     cost += 0.50
#                 if b_h2[m]:
#                     cost -= 0.30
#                 elif b_ndst[m] >= 2:
#                     cost -= 0.18
#                 cost += 0.08 * (dt_h / 24.0)
#                 pair_candidates.append((cost, k, m, dt_h, rat))
# 
#         if len(pair_candidates) < 2:
#             continue
#         pair_candidates.sort(key=lambda x: x[0])
#         used_c = set()
#         used_b_first_tx = set()
#         w_eps = []
# 
#         for cost, k, m, dt_h, rat in pair_candidates:
#             first_tx = int(obursts[m][0])
#             if k in used_c or first_tx in used_b_first_tx:
#                 continue
#             used_c.add(k)
#             used_b_first_tx.add(first_tx)
#             cl = clusters[k]
#             b = obursts[m]
# 
#             # Gather 2-hop children and final receiver set if distributed relay
#             h2_txs = np.flatnonzero(tx_has_hop2[b])
#             if len(h2_txs) > 0:
#                 p_tx = b[h2_txs[0]]
#                 ch_txs = tx_hop2_children[p_tx]
#                 hop2_m = float(tx_hop2_delay_m[p_tx])
#                 sink_txs = ch_txs
#                 all_out_txs = np.unique(np.concatenate([b, ch_txs]))
#                 recvs = set(dst_id[ch_txs].tolist())
#                 pay_dur_m = float((ts[ch_txs[-1]] - ts[ch_txs[0]]) / 60.0)
#             else:
#                 hop2_m = 0.0
#                 sink_txs = b
#                 all_out_txs = b
#                 recvs = set(dst_id[b].tolist())
#                 pay_dur_m = float((ts[b[-1]] - ts[b[0]]) / 60.0)
# 
#             all_ep_txs = np.unique(np.concatenate([cl['tx_in'], all_out_txs]))
#             w_eps.append({
#                 'collector': w,
#                 't_start': cl['t_start'],
#                 't_end': cl['t_end'],
#                 'u_in': cl['u_in'],
#                 'hold_h': float(dt_h),
#                 'pass_ratio': float(rat),
#                 'hop2_m': hop2_m,
#                 'has_hop2': float(len(h2_txs) > 0),
#                 'pay_dur_m': pay_dur_m,
#                 'conv_rate': float(np.mean(is_conv[all_ep_txs])),
#                 'senders': cl['senders'],
#                 'recvs': recvs,
#                 'u_outs': u_sent[sink_txs],
#                 'tx_in': cl['tx_in'],
#                 'tx_out': all_out_txs,
#                 'all_txs': all_ep_txs,
#             })
# 
#         if len(w_eps) < 2:
#             continue
#         w_eps.sort(key=lambda x: x['t_start'])
# 
#         # Compute full-wallet episode stats across all episodes on collector W
#         E = len(w_eps)
#         all_holds = np.array([ep['hold_h'] for ep in w_eps])
#         all_pass = np.array([ep['pass_ratio'] for ep in w_eps])
#         wal_hold_cv = float(np.std(all_holds) / (np.mean(all_holds) + 1e-3))
#         wal_pass_std = float(np.std(all_pass))
# 
#         # Step 5: For each episode on W, select coherent scenario cohort (2..4 episodes >=6h apart)
#         for idx_e, ep in enumerate(w_eps):
#             if E <= 4:
#                 cohort = w_eps
#             else:
#                 # Rank other episodes by hold-delay & relay-type similarity, enforcing >=6h spacing
#                 scored = []
#                 for idx_o, ep_o in enumerate(w_eps):
#                     if idx_o == idx_e:
#                         continue
#                     d_diff = abs(ep_o['hold_h'] - ep['hold_h']) / max(ep['hold_h'], ep_o['hold_h'], 1.0)
#                     t_diff = 0.5 * abs(ep_o['has_hop2'] - ep['has_hop2']) + 0.3 * abs(
#                         ep_o['conv_rate'] - ep['conv_rate']
#                     )
#                     scored.append((d_diff + t_diff, ep_o))
#                 scored.sort(key=lambda x: x[0])
#                 cohort = [ep]
#                 for _, cand_ep in scored:
#                     if all(abs(cand_ep['t_start'] - c_ep['t_start']) >= 21600.0 for c_ep in cohort):
#                         cohort.append(cand_ep)
#                         if len(cohort) == 4:
#                             break
#                 if len(cohort) < 2:
#                     cohort = w_eps[:4]
#                 cohort.sort(key=lambda x: x['t_start'])
# 
#             c_holds = np.array([c['hold_h'] for c in cohort])
#             c_pass = np.array([c['pass_ratio'] for c in cohort])
#             c_uins = np.array([c['u_in'] for c in cohort])
#             c_pdur = np.array([c['pay_dur_m'] for c in cohort])
#             c_hop2 = np.array([c['hop2_m'] for c in cohort])
#             c_uouts = np.concatenate([c['u_outs'] for c in cohort])
#             c_txs = np.unique(np.concatenate([c['all_txs'] for c in cohort]))
# 
#             scen_hold_med = float(np.median(c_holds))
#             scen_hold_cv = float(np.std(c_holds) / (np.mean(c_holds) + 1e-3))
#             scen_pass_med = float(np.median(c_pass))
#             scen_pass_std = float(np.std(c_pass))
#             scen_hop2_med = float(np.median(c_hop2))
#             scen_pdur_med = float(np.median(c_pdur))
#             scen_uout_cv = (
#                 float(np.std(c_uouts) / (np.mean(c_uouts) + 1e-3)) if len(c_uouts) >= 2 else 0.0
#             )
#             scen_uin_cv = float(np.std(c_uins) / (np.mean(c_uins) + 1e-3))
# 
#             jacc_l = [
#                 len(cohort[j]['recvs'] & cohort[j + 1]['recvs'])
#                 / max(1, len(cohort[j]['recvs'] | cohort[j + 1]['recvs']))
#                 for j in range(len(cohort) - 1)
#             ]
#             scen_recv_jacc = float(np.mean(jacc_l)) if jacc_l else 0.0
# 
#             peer_w = list(ep['senders'] | ep['recvs'])
#             peer_bg_med = float(np.median(w_total_tx[peer_w])) if peer_w else np.nan
#             scen_round = float(np.mean(is_round[c_txs]))
#             coll_tot = float(w_total_tx[w])
# 
#             # Unified scripted-relay score pooling Aggregate (~4.33h) and Distributed (~2.08h + ~122m hop)
#             d_agg = abs(scen_hold_med - 4.35) / 0.55
#             d_dist = abs(scen_hold_med - 2.08) / 0.35 + 0.7 * abs(scen_hop2_med - 122.0) / 25.0
#             band_closeness = np.maximum(np.exp(-0.5 * d_agg * d_agg), np.exp(-0.5 * d_dist * d_dist))
#             scripted_score = (
#                 2.2 * np.exp(-scen_hold_cv / 0.18)
#                 + 1.8 * np.exp(-scen_pass_std / 0.30)
#                 + 1.6 * band_closeness
#                 + 0.6 * np.tanh((scen_uout_cv - 0.70) / 0.30)
#                 - 0.5 * np.tanh((coll_tot - 105.0) / 20.0)
#             )
# 
#             ep['stats'] = (
#                 ep['hold_h'],
#                 scen_hold_med,
#                 scen_hold_cv,
#                 wal_hold_cv,
#                 ep['pass_ratio'],
#                 scen_pass_med,
#                 scen_pass_std,
#                 wal_pass_std,
#                 ep['hop2_m'] if ep['hop2_m'] > 0 else (
#                     float(tx_hop2_delay_m[ep['tx_out'][0]])
#                     if (len(ep['tx_out']) > 0 and tx_has_hop2[ep['tx_out'][0]])
#                     else 0.0
#                 ),
#                 scen_hop2_med,
#                 scen_pdur_med,
#                 scen_uout_cv,
#                 scen_uin_cv,
#                 scen_recv_jacc,
#                 coll_tot,
#                 peer_bg_med,
#                 scen_round,
#                 float(scripted_score),
#             )
#             ep['reg_priority'] = scen_hold_cv + 0.5 * scen_pass_std - 0.15 * band_closeness
#             episodes.append(ep)
# 
#     # Map reconstructed episode statistics onto every transaction (preferring most coherent episode on ties)
#     n_feats = 18
#     feat_mat = np.full((n, n_feats), np.nan, dtype=np.float64)
#     best_prio = np.full(n, 1e9, dtype=np.float64)
# 
#     for ep in episodes:
#         pr = ep['reg_priority']
#         st = ep['stats']
#         for idx_tx in ep['all_txs']:
#             if pr < best_prio[idx_tx]:
#                 best_prio[idx_tx] = pr
#                 feat_mat[idx_tx] = st
# 
#     # Populate standalone 2-hop relay delay if a 2-hop chain was found outside a >=2 episode cluster
#     for tx_p in np.flatnonzero(tx_has_hop2):
#         if np.isnan(feat_mat[tx_p, 8]):
#             feat_mat[tx_p, 8] = tx_hop2_delay_m[tx_p]
#         for ch_tx in tx_hop2_children[tx_p]:
#             if np.isnan(feat_mat[ch_tx, 8]):
#                 feat_mat[ch_tx, 8] = tx_hop2_delay_m[tx_p]
# 
#     feat_mat = feat_mat[inv_order]
#     out = pd.DataFrame(index=orig_index)
#     # 1. Episode hold delay in hours from collection cluster end to first payout
#     out['rl_ep_hold_delay_h'] = feat_mat[:, 0]
#     # 2. Median hold delay in hours across the reconstructed scenario's episodes (4.33h agg / 2.08h dist)
#     out['rl_scen_hold_med_h'] = feat_mat[:, 1]
#     # 3. CV of hold delays across the scenario's episodes (AUC 0.175-0.230: suspicious ~0.08 vs 0.71)
#     out['rl_scen_hold_cv'] = feat_mat[:, 2]
#     # 4. Un-subsetted wallet-wide CV of hold delays across all episodes on the collector wallet
#     out['rl_wal_hold_cv'] = feat_mat[:, 3]
#     # 5. Episode outbound-to-inbound USD pass-through ratio
#     out['rl_ep_pass_ratio'] = feat_mat[:, 4]
#     # 6. Median per-episode USD pass-through ratio across the scenario's episodes (AUC 0.260 in agg)
#     out['rl_scen_pass_med'] = feat_mat[:, 5]
#     # 7. Std of per-episode USD pass-through ratios across the scenario (AUC 0.109 in agg: 0.20 vs 0.64)
#     out['rl_scen_pass_std'] = feat_mat[:, 6]
#     # 8. Un-subsetted wallet-wide std of per-episode USD pass-through ratios on the collector wallet
#     out['rl_wal_pass_std'] = feat_mat[:, 7]
#     # 9. 2-hop relay delay in minutes from collector->relay to relay->receivers (122m in dist)
#     out['rl_hop2_delay_m'] = feat_mat[:, 8]
#     # 10. Median 2-hop relay delay in minutes across the scenario's episodes (AUC 0.679 in dist)
#     out['rl_scen_hop2_med_m'] = feat_mat[:, 9]
#     # 11. Median payout burst duration in minutes across the scenario's episodes (23m vs 50m in agg)
#     out['rl_scen_pay_dur_med_m'] = feat_mat[:, 10]
#     # 12. CV of outbound payout USD amounts across the scenario (AUC 0.772 in agg: 0.97 vs 0.65)
#     out['rl_scen_usd_out_cv'] = feat_mat[:, 11]
#     # 13. CV of total inbound USD per episode across the scenario (AUC 0.682 in dist: 0.71 vs 0.54)
#     out['rl_scen_in_usd_cv'] = feat_mat[:, 12]
#     # 14. Mean Jaccard overlap of receiver sets across consecutive episodes (AUC 0.293 in agg)
#     out['rl_scen_recv_jacc'] = feat_mat[:, 13]
#     # 15. Total weekly transaction count of the scenario's collector wallet (AUC 0.184 in dist)
#     out['rl_coll_total_tx'] = feat_mat[:, 14]
#     # 16. Median weekly transaction count of the episode's sender and receiver peer wallets
#     out['rl_peer_bg_med_tx'] = feat_mat[:, 15]
#     # 17. Share of round-USD or round-native-unit transactions in the scenario (AUC 0.752 in dist)
#     out['rl_scen_round_share'] = feat_mat[:, 16]
#     # 18. Unified scripted-relay log-odds score pooling both relay families above min_child_samples=60
#     out['rl_scripted_relay_score'] = feat_mat[:, 17]
# 
#     return out.astype(np.float64)
# 
# 
# I expect rl_scripted_relay_score, rl_scen_hold_cv, and rl_scen_pass_std (alongside rl_scen_hop2_med_m and rl_scen_hold_med_h) to drive the largest lift on hard cases because stripping <25m fast-motif pass-throughs and selecting coherent 
# ≥
# 6
# h
# ≥6h-spaced episode cohorts prevents multi-scenario wallet reuse from washing out the tight hold-clock regularity (CV ~0.08) and propagates the collector's signature to all collection, 1st-hop, and 2-hop relay transactions. Crucially, because each 4-fold training split has only ~80 positives per relay family and your LightGBM uses min_child_samples=60, rl_scripted_relay_score pools both families (~160 positives) into a single monotonic feature so trees can split without hitting the leaf-size floor.