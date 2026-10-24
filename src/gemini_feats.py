import numpy as np
import pandas as pd


def _compute_motif_arrays(
    comp_raw: np.ndarray,
    ts: np.ndarray,
    src_id: np.ndarray,
    dst_id: np.ndarray,
    u_sent: np.ndarray,
    u_recv: np.ndarray,
    is_conv: np.ndarray,
    same_curr_swap: np.ndarray,
    tx_iss_mask: np.ndarray,
    tx_cur_mask: np.ndarray,
    round_score: np.ndarray,
    W: int,
):
    """Vectorized O(N log N) component topology, flow conservation, and family priors."""
    n = len(comp_raw)
    _, comp_id = np.unique(comp_raw, return_inverse=True)
    K = int(comp_id.max()) + 1

    c_size = np.bincount(comp_id, minlength=K).astype(np.float64)
    c_conv = np.bincount(comp_id, weights=is_conv, minlength=K) / np.maximum(c_size, 1.0)
    c_swap = np.bincount(comp_id, weights=same_curr_swap, minlength=K) / np.maximum(c_size, 1.0)
    c_round = np.bincount(comp_id, weights=round_score, minlength=K) / np.maximum(c_size, 1.0)

    # Exact median and normalized IQR of USD_SENT per component via lexsort
    u_ord = np.lexsort((u_sent, comp_id))
    u_sorted = u_sent[u_ord]
    c_sz_int = c_size.astype(np.int32)
    c_starts = np.concatenate([[0], np.cumsum(c_sz_int)[:-1]])
    c_usd_med = 0.5 * (
        u_sorted[c_starts + (c_sz_int - 1) // 2] + u_sorted[c_starts + c_sz_int // 2]
    )
    c_usd_q25 = u_sorted[c_starts + (c_sz_int - 1) // 4]
    c_usd_q75 = u_sorted[c_starts + (3 * (c_sz_int - 1)) // 4]
    c_usd_iqr_norm = (c_usd_q75 - c_usd_q25) / (c_usd_med + 1.0)

    # Chronological gaps, monotone decay, and conversion sequence inside each component
    t_ord = np.argsort(comp_id, kind='mergesort')
    c_sorted = comp_id[t_ord]
    ts_sorted = ts[t_ord]
    us_sorted = u_sent[t_ord]
    ur_sorted = u_recv[t_ord]
    cv_sorted = is_conv[t_ord]

    same_c = c_sorted[1:] == c_sorted[:-1]
    valid_c = c_sorted[:-1][same_c]
    dts = ts_sorted[1:][same_c] - ts_sorted[:-1][same_c]

    n_gaps = np.maximum(c_size - 1.0, 1.0)
    gap_mean = np.bincount(valid_c, weights=dts, minlength=K) / n_gaps
    gap_sq_mean = np.bincount(valid_c, weights=dts * dts, minlength=K) / n_gaps
    gap_std = np.sqrt(np.maximum(0.0, gap_sq_mean - gap_mean * gap_mean))
    c_gap_cv = np.where(c_size >= 3, gap_std / (gap_mean + 1.0), -1.0)

    # Exact median gap per component via lexsort on (valid_c, dts)
    c_gap_med = np.full(K, -1.0, dtype=np.float64)
    if len(dts) > 0:
        g_ord = np.lexsort((dts, valid_c))
        vc_s = valid_c[g_ord]
        dt_s = dts[g_ord]
        g_cnt = np.bincount(vc_s, minlength=K)
        has_g = np.flatnonzero(g_cnt > 0)
        g_starts = np.concatenate([[0], np.cumsum(g_cnt)[:-1]])[has_g]
        g_sz = g_cnt[has_g]
        c_gap_med[has_g] = 0.5 * (
            dt_s[g_starts + (g_sz - 1) // 2] + dt_s[g_starts + g_sz // 2]
        )

    # Monotone step decay and conversion state alternation along time-sorted motif
    step_ratios = us_sorted[1:][same_c] / (ur_sorted[:-1][same_c] + 1e-6)
    is_decay = ((step_ratios >= 0.84) & (step_ratios <= 1.005)).astype(np.float64)
    c_monotone = np.where(
        c_size >= 2,
        np.bincount(valid_c, weights=is_decay, minlength=K) / n_gaps,
        0.0,
    )
    conv_flip = (cv_sorted[1:][same_c] != cv_sorted[:-1][same_c]).astype(np.float64)
    c_conv_alt = np.where(
        c_size >= 2,
        np.bincount(valid_c, weights=conv_flip, minlength=K) / n_gaps,
        0.0,
    )

    # Mean normalized temporal position of conversions in motif (-0.5 = early, +0.5 = late)
    pos_in_c = np.arange(n, dtype=np.float64) - np.repeat(c_starts, c_sz_int)
    norm_pos = pos_in_c / np.maximum(c_size[c_sorted] - 1.0, 1.0) - 0.5
    conv_pos_sum = np.bincount(c_sorted, weights=norm_pos * cv_sorted, minlength=K)
    conv_cnt = np.bincount(c_sorted, weights=cv_sorted, minlength=K)
    c_conv_pos_skew = np.where(conv_cnt > 0, conv_pos_sum / np.maximum(conv_cnt, 1.0), 0.0)

    # Unique senders, unique receivers, max in/out degree, and intermediary pass-through
    cw_src = comp_id.astype(np.int64) * W + src_id.astype(np.int64)
    cw_dst = comp_id.astype(np.int64) * W + dst_id.astype(np.int64)

    s_uniq, s_idx, s_cnt = np.unique(cw_src, return_index=True, return_counts=True)
    s_comp = comp_id[s_idx]
    c_n_src = np.bincount(s_comp, minlength=K).astype(np.float64)
    c_max_out = np.zeros(K, dtype=np.float64)
    np.maximum.at(c_max_out, s_comp, s_cnt.astype(np.float64))

    d_uniq, d_idx, d_cnt = np.unique(cw_dst, return_index=True, return_counts=True)
    d_comp = comp_id[d_idx]
    c_n_dst = np.bincount(d_comp, minlength=K).astype(np.float64)
    c_max_in = np.zeros(K, dtype=np.float64)
    np.maximum.at(c_max_in, d_comp, d_cnt.astype(np.float64))

    # Intermediary wallets (appear in BOTH cw_src and cw_dst inside the same component)
    cw_all = np.concatenate([cw_src, cw_dst])
    all_uniq = np.unique(cw_all)
    c_n_wallets = np.bincount((all_uniq // W).astype(np.int32), minlength=K).astype(np.float64)

    # Sum USD_SENT per (comp, src) and USD_RECV per (comp, dst) to get exact intermediary retention
    s_inv = np.searchsorted(s_uniq, cw_src)
    s_usd_sum = np.bincount(s_inv, weights=u_sent, minlength=len(s_uniq))
    d_inv = np.searchsorted(d_uniq, cw_dst)
    d_usd_sum = np.bincount(d_inv, weights=u_recv, minlength=len(d_uniq))

    mid_mask_s = np.isin(s_uniq, d_uniq, assume_unique=True)
    mid_cw = s_uniq[mid_mask_s]
    mid_comp = (mid_cw // W).astype(np.int32)
    c_n_mid = np.bincount(mid_comp, minlength=K).astype(np.float64)

    c_retain_med = np.full(K, -1.0, dtype=np.float64)
    c_retain_std = np.full(K, -1.0, dtype=np.float64)
    if len(mid_cw) > 0:
        mid_d_pos = np.searchsorted(d_uniq, mid_cw)
        mid_ratios = np.clip(s_usd_sum[mid_mask_s] / (d_usd_sum[mid_d_pos] + 1e-6), 0.0, 2.5)
        m_ord = np.lexsort((mid_ratios, mid_comp))
        mc_s = mid_comp[m_ord]
        mr_s = mid_ratios[m_ord]
        m_cnt = np.bincount(mc_s, minlength=K)
        has_m = np.flatnonzero(m_cnt > 0)
        m_starts = np.concatenate([[0], np.cumsum(m_cnt)[:-1]])[has_m]
        m_sz = m_cnt[has_m]
        c_retain_med[has_m] = 0.5 * (
            mr_s[m_starts + (m_sz - 1) // 2] + mr_s[m_starts + m_sz // 2]
        )
        m_mean = np.bincount(mc_s, weights=mr_s, minlength=K)[has_m] / m_sz
        m_sq = np.bincount(mc_s, weights=mr_s * mr_s, minlength=K)[has_m] / m_sz
        c_retain_std[has_m] = np.sqrt(np.maximum(0.0, m_sq - m_mean * m_mean))

    # Bitmask popcount for distinct issuers and non-XRP currencies per component
    c_iss_mask = np.zeros(K, dtype=np.int64)
    c_cur_mask = np.zeros(K, dtype=np.int64)
    np.bitwise_or.at(c_iss_mask, comp_id, tx_iss_mask)
    np.bitwise_or.at(c_cur_mask, comp_id, tx_cur_mask)
    c_n_iss = np.zeros(K, dtype=np.float64)
    c_n_cur = np.zeros(K, dtype=np.float64)
    for bit in range(16):
        c_n_iss += ((c_iss_mask >> bit) & 1).astype(np.float64)
        c_n_cur += ((c_cur_mask >> bit) & 1).astype(np.float64)
    c_iss_per_cur = c_n_iss / np.maximum(c_n_cur, 1.0)

    # Soft topological family weights for components of size >= 3
    active = (c_size >= 3).astype(np.float64)
    mid_ratio = c_n_mid / np.maximum(c_n_wallets, 1.0)

    # 1. Cycle: nearly all wallets both send & receive, max degree ~1-2
    w_cycle = active * np.clip((mid_ratio - 0.65) / 0.35, 0.0, 1.0) * (c_max_out <= 2) * (c_max_in <= 2)
    # 2. Chain: linear serial path, max_in==1, max_out==1, n_wallets == size + 1
    w_chain = active * (1.0 - w_cycle) * (c_max_out == 1) * (c_max_in == 1) * (c_n_mid >= 1)
    # 3. Bipartite: no intermediaries (n_mid == 0), multiple senders and multiple receivers
    w_bip = active * (c_n_mid == 0) * (c_n_src >= 2) * (c_n_dst >= 2)
    # 4. Fan-out: no intermediaries, 1 sender, >=3 receivers
    w_fout = active * (c_n_mid == 0) * (c_n_src == 1) * (c_n_dst >= 3)
    # 5. Fan-in: no intermediaries, 1 receiver, >=3 senders
    w_fin = active * (c_n_mid == 0) * (c_n_dst == 1) * (c_n_src >= 3)
    # 6. Fan-out-in vs Stack (multi-stage with intermediaries and branching degree >= 2)
    w_branched_mid = active * (c_n_mid >= 1) * ((c_max_out >= 2) | (c_max_in >= 2))
    is_scatter_gather = ((c_n_src - c_n_mid) <= 1.5) | ((c_n_dst - c_n_mid) <= 1.5)
    w_foi = w_branched_mid * is_scatter_gather.astype(np.float64)
    w_stack = w_branched_mid * (1.0 - is_scatter_gather.astype(np.float64))

    # Empirical log-odds per family from GENERATOR FACTS (suspicious vs look-alike medians)
    s_cycle = (0.30 - c_conv) * 2.5 + np.tanh((c_usd_med - 160.0) / 70.0)
    s_chain = (0.25 - c_conv) * 3.0
    s_bip = (c_conv - 0.25) * 3.0
    s_fout = (c_conv - 0.20) * 2.5 + np.tanh((c_usd_med - 68.0) / 30.0)
    s_fin = np.tanh((c_usd_med - 150.0) / 55.0) * 1.8
    s_foi = (c_conv - 0.25) * 3.0
    s_stack = np.tanh((c_size - 7.5) / 3.0) * 2.0

    c_family_logodds = (
        w_cycle * s_cycle
        + w_chain * s_chain
        + w_bip * s_bip
        + w_fout * s_fout
        + w_fin * s_fin
        + w_foi * s_foi
        + w_stack * s_stack
    )

    # Signed XOR interaction between branching-vs-serial topology and conversion share
    branch_minus_serial = (w_bip + w_fout + w_foi) - (w_cycle + w_chain)
    c_xor_conv = branch_minus_serial * (c_conv - 0.25)

    return {
        'size': c_size[comp_id],
        'conv': c_conv[comp_id],
        'swap': c_swap[comp_id],
        'round': c_round[comp_id],
        'usd_med': c_usd_med[comp_id],
        'usd_iqr_norm': c_usd_iqr_norm[comp_id],
        'gap_cv': c_gap_cv[comp_id],
        'gap_med': c_gap_med[comp_id],
        'monotone': c_monotone[comp_id],
        'conv_alt': c_conv_alt[comp_id],
        'conv_pos_skew': c_conv_pos_skew[comp_id],
        'retain_med': c_retain_med[comp_id],
        'retain_std': c_retain_std[comp_id],
        'mid_ratio': mid_ratio[comp_id],
        'bip_score': (w_bip * (c_size / np.maximum(c_n_src + c_n_dst, 1.0)))[comp_id],
        'cycle_score': w_cycle[comp_id],
        'fan_asym': ((c_max_out - c_max_in) / np.maximum(c_size, 1.0))[comp_id],
        'iss_per_cur': c_iss_per_cur[comp_id],
        'family_logodds': c_family_logodds[comp_id],
        'xor_conv': c_xor_conv[comp_id],
        'comp_id': comp_id,
    }


def gemini_features(e: pd.DataFrame) -> pd.DataFrame:
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
    v_sent = e['V_SENT'].values[order].astype(np.float64)
    is_conv = e['IS_CONV'].values[order].astype(np.float64)

    # Fast integer encoding for wallets
    src_raw = e['SRC'].values[order]
    dst_raw = e['DST'].values[order]
    wallets, w_inv = np.unique(np.concatenate([src_raw, dst_raw]), return_inverse=True)
    W = len(wallets)
    src_id = w_inv[:n].astype(np.int32)
    dst_id = w_inv[n:].astype(np.int32)

    # Fast lookup table for asset, issuer, and currency IDs
    a_sent_raw = e['ASSET_SENT'].astype(str).values[order]
    a_recv_raw = e['ASSET_RECV'].astype(str).values[order]
    uniq_assets, a_inv = np.unique(np.concatenate([a_sent_raw, a_recv_raw]), return_inverse=True)
    s_asset_id = a_inv[:n].astype(np.int32)
    r_asset_id = a_inv[n:].astype(np.int32)

    u_iss, u_cur = [], []
    for s in uniq_assets:
        if ':' in s:
            iss, cur = s.split(':', 1)
        else:
            iss, cur = 'XRP', 'XRP'
        u_iss.append(iss)
        u_cur.append(cur)
    _, iss_map = np.unique(u_iss, return_inverse=True)
    uniq_cur, cur_map = np.unique(u_cur, return_inverse=True)
    s_iss_id = iss_map[s_asset_id]
    r_iss_id = iss_map[r_asset_id]
    s_cur_id = cur_map[s_asset_id]
    r_cur_id = cur_map[r_asset_id]

    same_curr_swap = ((is_conv > 0.5) & (s_cur_id == r_cur_id)).astype(np.float64)
    tx_iss_mask = (1 << np.minimum(s_iss_id, 15)) | (1 << np.minimum(r_iss_id, 15))
    tx_cur_mask = (1 << np.minimum(s_cur_id, 15)) | (1 << np.minimum(r_cur_id, 15))

    # Round USD / native unit target detection
    d_us1 = np.abs(u_sent - np.round(u_sent))
    d_us10 = np.abs(u_sent / 10.0 - np.round(u_sent / 10.0)) * 10.0
    d_vs1 = np.abs(v_sent - np.round(v_sent))
    round_score = (
        0.4 * (d_us1 < 0.02) + 0.3 * (d_us10 < 0.05) + 0.3 * (d_vs1 < 0.01)
    ).astype(np.float64)

    # Group chronologically sorted tx indices by wallet
    src_counts = np.bincount(src_id, minlength=W)
    src_order = np.argsort(src_id, kind='mergesort')
    w_Outs = np.split(src_order, np.cumsum(src_counts)[:-1])

    dst_counts = np.bincount(dst_id, minlength=W)
    dst_order = np.argsort(dst_id, kind='mergesort')
    w_Ins = np.split(dst_order, np.cumsum(dst_counts)[:-1])

    # Per-wallet multi-tx flow conservation, collector traps, and 1-to-1 value relays
    src_in_usd_90m = np.zeros(n, dtype=np.float64)
    src_out_usd_90m = np.zeros(n, dtype=np.float64)
    dst_in_usd_90m = np.zeros(n, dtype=np.float64)
    dst_out_usd_90m = np.zeros(n, dtype=np.float64)
    src_in_cnt_3d_2h = np.zeros(n, dtype=np.float64)
    src_in_cnt_2h = np.zeros(n, dtype=np.float64)
    src_out_cnt_30m = np.zeros(n, dtype=np.float64)

    best_prev_tx = np.full(n, -1, dtype=np.int32)
    best_prev_score = np.full(n, 1e9, dtype=np.float64)
    best_next_tx = np.full(n, -1, dtype=np.int32)
    best_next_score = np.full(n, 1e9, dtype=np.float64)

    for w in range(W):
        ins = w_Ins[w]
        outs = w_Outs[w]
        n_i, n_o = len(ins), len(outs)
        if n_o > 0:
            t_out = ts[outs]
            u_out = u_sent[outs]
            cs_out = np.concatenate([[0.0], np.cumsum(u_out)])
            lo90 = np.searchsorted(t_out, t_out - 2700.0, side='left')
            hi90 = np.searchsorted(t_out, t_out + 2700.0, side='right')
            src_out_usd_90m[outs] = cs_out[hi90] - cs_out[lo90]
            lo15 = np.searchsorted(t_out, t_out - 900.0, side='left')
            hi15 = np.searchsorted(t_out, t_out + 900.0, side='right')
            src_out_cnt_30m[outs] = (hi15 - lo15).astype(np.float64)

        if n_i > 0:
            t_in = ts[ins]
            u_in = u_recv[ins]
            cs_in = np.concatenate([[0.0], np.cumsum(u_in)])
            li90 = np.searchsorted(t_in, t_in - 2700.0, side='left')
            ri90 = np.searchsorted(t_in, t_in + 2700.0, side='right')
            dst_in_usd_90m[ins] = cs_in[ri90] - cs_in[li90]

        if n_i > 0 and n_o > 0:
            l90 = np.searchsorted(t_in, t_out - 5400.0, side='left')
            r90 = np.searchsorted(t_in, t_out, side='right')
            src_in_usd_90m[outs] = cs_in[r90] - cs_in[l90]

            l3d = np.searchsorted(t_in, t_out - 259200.0, side='left')
            r2h = np.searchsorted(t_in, t_out - 7200.0, side='right')
            src_in_cnt_3d_2h[outs] = np.maximum(0, r2h - l3d).astype(np.float64)
            src_in_cnt_2h[outs] = np.maximum(0, r90 - r2h).astype(np.float64)

            l90_o = np.searchsorted(t_out, t_in, side='left')
            r90_o = np.searchsorted(t_out, t_in + 5400.0, side='right')
            dst_out_usd_90m[ins] = cs_out[r90_o] - cs_out[l90_o]

            # Match 1-to-1 relays in [0, 5400s] with pass-through ratio in [0.78, 1.03]
            for a_idx in np.flatnonzero(r90_o > l90_o):
                p_tx = ins[a_idx]
                sl = slice(l90_o[a_idx], r90_o[a_idx])
                q_cands = outs[sl]
                dt_c = t_out[sl] - t_in[a_idx]
                ratios = u_out[sl] / (u_in[a_idx] + 1e-6)
                valid = (ratios >= 0.78) & (ratios <= 1.03)
                if np.any(valid):
                    a_Pen = np.where(s_asset_id[q_cands] == r_asset_id[p_tx], 0.0, 0.04)
                    sc = np.abs(ratios - 0.95) + a_Pen + (dt_c / 5400.0) * 0.04
                    sc[~valid] = 1e9
                    best_k = int(np.argmin(sc))
                    best_sc = sc[best_k]
                    q_tx = q_cands[best_k]
                    if best_sc < best_next_score[p_tx]:
                        best_next_score[p_tx] = best_sc
                        best_next_tx[p_tx] = q_tx
                    if best_sc < best_prev_score[q_tx]:
                        best_prev_score[q_tx] = best_sc
                        best_prev_tx[q_tx] = p_tx

    # Trace value-matched relay paths (up to 8 hops back and forward)
    vchain_len = np.ones(n, dtype=np.float64)
    vchain_dt_cv = np.full(n, -1.0, dtype=np.float64)
    vchain_dt_med = np.full(n, -1.0, dtype=np.float64)
    vchain_ratio_mean = np.full(n, -1.0, dtype=np.float64)
    vchain_ratio_std = np.full(n, -1.0, dtype=np.float64)
    vchain_asset_cont = np.full(n, -1.0, dtype=np.float64)
    vchain_is_cycle = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if best_prev_tx[i] == -1 and best_next_tx[i] == -1:
            continue
        dts_l, rat_l, ast_l = [], [], []
        vis = {int(src_id[i]), int(dst_id[i])}
        cur = i
        for _ in range(8):
            prv = int(best_prev_tx[cur])
            if prv == -1 or prv == cur:
                break
            dts_l.append(ts[cur] - ts[prv])
            rat_l.append(u_sent[cur] / (u_recv[prv] + 1e-6))
            ast_l.append(1.0 if s_asset_id[cur] == r_asset_id[prv] else 0.0)
            if int(src_id[prv]) in vis:
                vchain_is_cycle[i] = 1.0
                break
            vis.add(int(src_id[prv]))
            cur = prv
        cur = i
        for _ in range(8):
            nxt = int(best_next_tx[cur])
            if nxt == -1 or nxt == cur:
                break
            dts_l.append(ts[nxt] - ts[cur])
            rat_l.append(u_sent[nxt] / (u_recv[cur] + 1e-6))
            ast_l.append(1.0 if s_asset_id[nxt] == r_asset_id[cur] else 0.0)
            if int(dst_id[nxt]) in vis:
                vchain_is_cycle[i] = 1.0
                break
            vis.add(int(dst_id[nxt]))
            cur = nxt

        nh = len(dts_l)
        if nh > 0:
            vchain_len[i] = 1.0 + nh
            arr_dt = np.array(dts_l, dtype=np.float64)
            arr_rt = np.array(rat_l, dtype=np.float64)
            vchain_dt_med[i] = float(np.median(arr_dt))
            vchain_dt_cv[i] = float(np.std(arr_dt) / (np.mean(arr_dt) + 1.0)) if nh >= 2 else 0.0
            vchain_ratio_mean[i] = float(np.mean(arr_rt))
            vchain_ratio_std[i] = float(np.std(arr_rt)) if nh >= 2 else 0.0
            vchain_asset_cont[i] = float(np.mean(ast_l))

    # Build endpoint event table (2N) to generate candidate motif links and wallet quietness
    w_2n = np.concatenate([src_id, dst_id])
    ts_2n = np.concatenate([ts, ts])
    tx_2n = np.concatenate([np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32)])
    role_2n = np.concatenate([np.zeros(n, dtype=np.int8), np.ones(n, dtype=np.int8)])
    usd_2n = np.concatenate([u_sent, u_recv])

    ev_ord = np.lexsort((tx_2n, ts_2n, w_2n))
    w_s = w_2n[ev_ord]
    ts_s = ts_2n[ev_ord]
    tx_s = tx_2n[ev_ord]
    role_s = role_2n[ev_ord]
    usd_s = usd_2n[ev_ord]

    # Collect local candidate edges (step = 1..6 on same wallet within 2700s)
    src_conserv = src_out_usd_90m / (src_in_usd_90m + 1e-6)
    dst_conserv = dst_out_usd_90m / (dst_in_usd_90m + 1e-6)
    collector_trap = src_in_cnt_3d_2h * src_out_cnt_30m / (1.0 + src_in_cnt_2h)

    eu_l, ev_l, edt_l, ecost_l, efast_l = [], [], [], [], []
    for step in range(1, 7):
        same_w = (w_s[:-step] == w_s[step:]) & (tx_s[:-step] != tx_s[step:])
        dt_step = ts_s[step:] - ts_s[:-step]
        mask = same_w & (dt_step <= 2700.0)
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        ta = tx_s[idx]
        tb = tx_s[idx + step]
        dt_ab = dt_step[idx]
        ra = role_s[idx]
        rb = role_s[idx + step]
        ua = usd_s[idx]
        ub = usd_s[idx + step]

        u_ratio = np.maximum(ua, ub) / (np.minimum(ua, ub) + 1e-6)
        is_relay = ra != rb
        flow_ok = (
            ((src_conserv[tb] >= 0.75) & (src_conserv[tb] <= 1.08))
            | ((dst_conserv[ta] >= 0.75) & (dst_conserv[ta] <= 1.08))
            | (u_ratio <= 1.30)
        )
        not_hub = (collector_trap[ta] < 12.0) & (collector_trap[tb] < 12.0) & (
            src_out_cnt_30m[ta] <= 18
        ) & (src_out_cnt_30m[tb] <= 18)

        keep_med = (is_relay & flow_ok & not_hub) | ((~is_relay) & (u_ratio <= 3.2) & not_hub)
        if np.any(keep_med):
            eu_l.append(ta[keep_med])
            ev_l.append(tb[keep_med])
            dt_k = dt_ab[keep_med]
            ur_k = u_ratio[keep_med]
            edt_l.append(dt_k)
            ecost_l.append(dt_k * (1.0 + 0.25 * np.minimum(ur_k - 1.0, 4.0)))
            efast_l.append(dt_k <= 600.0)

    # Also include 1-to-1 value-matched relay links (up to 3000s) in medium motif candidates
    has_nxt = np.flatnonzero((best_next_tx != -1) & ((ts[best_next_tx] - ts) <= 3000.0))
    if len(has_nxt) > 0:
        q_nxt = best_next_tx[has_nxt]
        dt_nxt = ts[q_nxt] - ts[has_nxt]
        eu_l.append(has_nxt.astype(np.int32))
        ev_l.append(q_nxt.astype(np.int32))
        edt_l.append(dt_nxt)
        ecost_l.append(dt_nxt * 0.7)
        efast_l.append(dt_nxt <= 600.0)

    eu = np.concatenate(eu_l) if eu_l else np.empty(0, dtype=np.int32)
    ev = np.concatenate(ev_l) if ev_l else np.empty(0, dtype=np.int32)
    edt = np.concatenate(edt_l) if edt_l else np.empty(0, dtype=np.float64)
    ecost = np.concatenate(ecost_l) if ecost_l else np.empty(0, dtype=np.float64)
    efast = np.concatenate(efast_l) if efast_l else np.empty(0, dtype=bool)

    def _kruskal_components(u_arr, v_arr, cost_arr, dt_arr, max_sz=18):
        parent = np.arange(n, dtype=np.int32)
        sz = np.ones(n, dtype=np.int32)
        if len(u_arr) > 0:
            s_ord = np.argsort(cost_arr, kind='mergesort')
            u_s, v_s, dt_s = u_arr[s_ord], v_arr[s_ord], dt_arr[s_ord]
            for a, b, d in zip(u_s, v_s, dt_s):
                ra, rb = int(a), int(b)
                while parent[ra] != ra:
                    parent[ra] = parent[parent[ra]]
                    ra = parent[ra]
                while parent[rb] != rb:
                    parent[rb] = parent[parent[rb]]
                    rb = parent[rb]
                if ra != rb:
                    if sz[ra] + sz[rb] <= max_sz or (d <= 60.0 and sz[ra] + sz[rb] <= 22):
                        if sz[ra] < sz[rb]:
                            ra, rb = rb, ra
                        parent[rb] = ra
                        sz[ra] += sz[rb]
        for i in range(n):
            r = i
            while parent[r] != r:
                r = parent[r]
            parent[i] = r
        return parent

    comp_fast = _kruskal_components(eu[efast], ev[efast], ecost[efast], edt[efast], max_sz=18)
    comp_med = _kruskal_components(eu, ev, ecost, edt, max_sz=18)

    st_fast = _compute_motif_arrays(
        comp_fast, ts, src_id, dst_id, u_sent, u_recv,
        is_conv, same_curr_swap, tx_iss_mask, tx_cur_mask, round_score, W
    )
    st_med = _compute_motif_arrays(
        comp_med, ts, src_id, dst_id, u_sent, u_recv,
        is_conv, same_curr_swap, tx_iss_mask, tx_cur_mask, round_score, W
    )

    # Wallet activity isolation outside the reconstructed 45m motif
    med_cid = st_med['comp_id']
    ev_cid = med_cid[tx_s]
    outside_gap = np.full(n, 86400.0, dtype=np.float64)
    for step in (1, 2, 3):
        valid_b = (w_s[step:] == w_s[:-step]) & (ev_cid[step:] != ev_cid[:-step])
        idx_b = np.flatnonzero(valid_b)
        dt_b = ts_s[idx_b + step] - ts_s[idx_b]
        np.minimum.at(outside_gap, tx_s[idx_b + step], dt_b)
        np.minimum.at(outside_gap, tx_s[idx_b], dt_b)

    log_outside_gap = np.log1p(np.maximum(0.0, outside_gap))
    K_med = int(med_cid.max()) + 1
    c_iso_min = np.full(K_med, np.log1p(86400.0), dtype=np.float64)
    np.minimum.at(c_iso_min, med_cid, log_outside_gap)
    med_boundary_iso = c_iso_min[med_cid]

    # Target pass-through error for medium motif intermediaries
    med_ret = st_med['retain_med']
    med_ret_err = np.where(
        med_ret >= 0,
        np.abs(med_ret - 0.92) + 0.5 * np.maximum(0.0, st_med['retain_std']),
        -1.0,
    )

    # Assemble all 38 features mapped back to original DataFrame order and index
    out = pd.DataFrame(index=orig_index)
    # 1. Family-conditioned log-odds on 10m motif (separates fast look-alikes from easy suspicious)
    out['gm_fast_family_logodds'] = st_fast['family_logodds'][inv_order]
    # 2. Family-conditioned log-odds on 45m Kruskal motif (lifts medium scenarios across all 7 families)
    out['gm_med_family_logodds'] = st_med['family_logodds'][inv_order]
    # 3. Signed XOR interaction of topology (branching vs serial) and conversion share in 10m motif
    out['gm_fast_xor_conv_topo'] = st_fast['xor_conv'][inv_order]
    # 4. Signed XOR interaction of topology (branching vs serial) and conversion share in 45m motif
    out['gm_med_xor_conv_topo'] = st_med['xor_conv'][inv_order]
    # 5. Size of size-capped 45m Kruskal motif (isolates medium scenarios and size-12 stacks)
    out['gm_med_size'] = st_med['size'][inv_order]
    # 6. Conversion share inside 45m Kruskal motif (crucial discriminator missing above 15m)
    out['gm_med_conv_share'] = st_med['conv'][inv_order]
    # 7. Median USD_SENT inside 45m Kruskal motif (separates high-USD cycle/fan_in/fan_out from look-alikes)
    out['gm_med_usd_med'] = st_med['usd_med'][inv_order]
    # 8. Normalized IQR of USD_SENT in 45m motif (measures intra-motif amount coherence)
    out['gm_med_usd_iqr_norm'] = st_med['usd_iqr_norm'][inv_order]
    # 9. CV of consecutive timestamps in 45m motif (medium suspicious 0.41 vs look-alike 0.52)
    out['gm_med_gap_cv'] = st_med['gap_cv'][inv_order]
    # 10. Median consecutive time gap in seconds inside 45m motif (identifies ~18 min medium cadence)
    out['gm_med_gap_med'] = st_med['gap_med'][inv_order]
    # 11. Median multi-edge USD pass-through ratio across intermediaries in 45m motif
    out['gm_med_intermed_retain_med'] = med_ret[inv_order]
    # 12. Intermediary pass-through deviation from suspicious retention band plus dispersion in 45m motif
    out['gm_med_intermed_retain_err'] = med_ret_err[inv_order]
    # 13. Median multi-edge USD pass-through ratio across intermediaries in 10m motif
    out['gm_fast_intermed_retain_med'] = st_fast['retain_med'][inv_order]
    # 14. Standard deviation of intermediary pass-through ratios in 10m motif (uniform vs noisy decay)
    out['gm_fast_intermed_retain_std'] = st_fast['retain_std'][inv_order]
    # 15. Fraction of wallets in 45m motif that both send and receive inside the motif
    out['gm_med_mid_wallet_ratio'] = st_med['mid_ratio'][inv_order]
    # 16. Structural bipartite motif score in 45m window (zero intermediaries, multi-src, multi-dst)
    out['gm_med_bipartite_score'] = st_med['bip_score'][inv_order]
    # 17. Structural cycle motif score in 45m window (closed loop topology)
    out['gm_med_cycle_score'] = st_med['cycle_score'][inv_order]
    # 18. Degree asymmetry (max_out - max_in)/size in 45m motif (fan_out vs fan_in separation)
    out['gm_med_fan_asymmetry'] = st_med['fan_asym'][inv_order]
    # 19. Fraction of consecutive tx pairs in 45m motif showing monotone hop decay (0.84-1.005)
    out['gm_med_monotone_decay'] = st_med['monotone'][inv_order]
    # 20. Conversion state alternation rate across consecutive transactions in 45m motif
    out['gm_med_conv_alternation'] = st_med['conv_alt'][inv_order]
    # 21. Mean normalized temporal position of conversions within 45m motif (early vs late conversions)
    out['gm_med_conv_pos_skew'] = st_med['conv_pos_skew'][inv_order]
    # 22. Log minimum time gap from 45m motif wallets to nearest outside-motif transaction
    out['gm_med_boundary_isolation'] = med_boundary_iso[inv_order]
    # 23. Total length of 90m value-matched (0.78-1.03) 1-to-1 relay path through this tx
    out['gm_vchain_len'] = vchain_len[inv_order]
    # 24. CV of hop-to-hop waiting times along reconstructed value-matched relay path
    out['gm_vchain_dt_cv'] = vchain_dt_cv[inv_order]
    # 25. Median hop-to-hop waiting time (seconds) along value-matched relay path
    out['gm_vchain_dt_med'] = vchain_dt_med[inv_order]
    # 26. Mean hop-to-hop USD pass-through ratio along value-matched relay path
    out['gm_vchain_ratio_mean'] = vchain_ratio_mean[inv_order]
    # 27. Standard deviation of hop-to-hop pass-through ratios along value-matched path
    out['gm_vchain_ratio_std'] = vchain_ratio_std[inv_order]
    # 28. Fraction of hops along value-matched path keeping exact same asset and issuer
    out['gm_vchain_asset_continuity'] = vchain_asset_cont[inv_order]
    # 29. Binary indicator that the value-matched relay path forms a closed wallet cycle
    out['gm_vchain_is_cycle'] = vchain_is_cycle[inv_order]
    # 30. Multi-tx flow conservation ratio at SRC over 90m (captures 1-to-many splits summing to input)
    out['gm_src_split_conserv_90m'] = np.clip(src_conserv, 0.0, 3.0)[inv_order]
    # 31. Multi-tx flow conservation ratio at DST over 90m (captures many-to-1 merges summing to output)
    out['gm_dst_merge_conserv_90m'] = np.clip(dst_conserv, 0.0, 3.0)[inv_order]
    # 32. Multi-day collector burst score (flags 20k innocent relay collector payouts vs true motifs)
    out['gm_collector_burst_trap'] = np.log1p(collector_trap)[inv_order]
    # 33. Indicator for same-currency issuer swap (e.g. IssuerA:USD to IssuerB:USD)
    out['gm_same_curr_issuer_swap'] = same_curr_swap[inv_order]
    # 34. Share of transactions in 45m motif performing a same-currency issuer swap
    out['gm_med_issuer_swap_share'] = st_med['swap'][inv_order]
    # 35. Ratio of distinct issuers to distinct currencies inside 45m motif
    out['gm_med_issuer_per_curr'] = st_med['iss_per_cur'][inv_order]
    # 36. Composite round-number score on USD_SENT and V_SENT
    out['gm_round_amount_score'] = round_score[inv_order]
    # 37. Mean round-number score across transactions in the 45m motif
    out['gm_med_round_share'] = st_med['round'][inv_order]
    # 38. Log time gap from this tx's endpoints to nearest outside-motif activity
    out['gm_wallet_quiet_min_log'] = log_outside_gap[inv_order]

    return out.astype(np.float64)


