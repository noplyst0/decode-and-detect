#CODE
import numpy as np
import pandas as pd


def medium_report(e: pd.DataFrame, lab: pd.DataFrame) -> str:
    # Align edge and label tables on TRANSACTION_HASH
    e_df = e.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in e.columns else e.copy()
    l_df = lab.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in lab.columns else lab.copy()

    # Precompute global weekly wallet activity and pair recurrence on full 40k tx
    e_sorted = e_df.sort_values('TS')
    all_src = e_sorted['SRC'].values
    all_dst = e_sorted['DST'].values
    all_ts = e_sorted['TS'].values.astype(np.float64)

    w_all = pd.concat([e_df['SRC'], e_df['DST']])
    global_w_cnt = w_all.value_counts().to_dict()

    # Build sorted timestamp list per wallet across the entire week
    w_ev_df = pd.DataFrame({
        'w': np.concatenate([all_src, all_dst]),
        'ts': np.concatenate([all_ts, all_ts]),
    }).sort_values('ts', kind='mergesort')
    w_keys, w_counts = np.unique(w_ev_df['w'].values, return_counts=True)
    w_ord = np.argsort(w_ev_df['w'].values, kind='mergesort')
    w_ts_splits = np.split(w_ev_df['ts'].values[w_ord], np.cumsum(w_counts)[:-1])
    w_ts_map = dict(zip(w_keys, w_ts_splits))

    # Undirected wallet-pair weekly counts
    u_min = np.minimum(all_src, all_dst)
    u_max = np.maximum(all_src, all_dst)
    pair_keys = pd.Series(list(zip(u_min, u_max))).value_counts().to_dict()

    # Filter to non-relay, non-background motif scenarios
    exclude_fams = {'aggregate_relay', 'distributed_relay', 'background', 'none', ''}
    valid_mask = l_df['scenario_id'].notna() & (
        ~l_df['family'].astype(str).str.lower().isin(exclude_fams)
    )
    l_sub = l_df[valid_mask].copy()
    df = e_df.join(l_sub[['is_suspicious', 'family', 'difficulty', 'scenario_id']], how='inner')
    df['is_suspicious'] = df['is_suspicious'].astype(int)

    # Parse currencies and issuers
    s_sp = df['ASSET_SENT'].astype(str).str.split(':', n=1, expand=True)
    r_sp = df['ASSET_RECV'].astype(str).str.split(':', n=1, expand=True)
    df['s_cur'] = s_sp[1].fillna('XRP') if s_sp.shape[1] > 1 else 'XRP'
    df['r_cur'] = r_sp[1].fillna('XRP') if r_sp.shape[1] > 1 else 'XRP'
    df['same_cur_swap'] = ((df['IS_CONV'] > 0) & (df['s_cur'] == df['r_cur'])).astype(float)

    us_all = df['USD_SENT'].values.astype(float)
    ur_all = df['USD_RECV'].values.astype(float)
    vs_all = df['V_SENT'].values.astype(float)
    df['is_round'] = (
        (np.abs(us_all - np.round(us_all)) < 0.02)
        | (np.abs(us_all / 10.0 - np.round(us_all / 10.0)) * 10.0 < 0.05)
        | (np.abs(ur_all - np.round(ur_all)) < 0.02)
        | (np.abs(vs_all - np.round(vs_all)) < 0.01)
    ).astype(float)

    records = []
    for (fam, sc_id), g in df.groupby(['family', 'scenario_id'], sort=False):
        g = g.sort_values('TS', kind='mergesort')
        n_tx = len(g)
        if n_tx < 2:
            continue

        susp = int(g['is_suspicious'].iloc[0])
        diff = str(g['difficulty'].iloc[0]).lower()
        if susp == 1 and 'med' in diff:
            grp_type = 'med'
        elif susp == 1 and 'eas' in diff:
            grp_type = 'easy'
        elif susp == 0:
            grp_type = 'inno'
        else:
            continue

        ts = g['TS'].values.astype(float)
        u_s = g['USD_SENT'].values.astype(float)
        u_r = g['USD_RECV'].values.astype(float)
        conv = g['IS_CONV'].values.astype(float)
        n_legs = g['N_LEGS'].values.astype(float)
        sendmax = g['HAS_SENDMAX'].values.astype(float)
        a_s = g['ASSET_SENT'].astype(str).values
        a_r = g['ASSET_RECV'].astype(str).values
        src = g['SRC'].values
        dst = g['DST'].values

        # 1. Timing & Gap distribution
        dur_s = max(1.0, ts[-1] - ts[0])
        dts = np.diff(ts)
        gap_med_m = float(np.median(dts) / 60.0)
        gap_cv = float(np.std(dts) / (np.mean(dts) + 1e-3))
        gap_max_sh = float(np.max(dts) / dur_s)
        gap_adj_rel = (
            float(np.median(np.abs(dts[1:] - dts[:-1]) / (dts[1:] + dts[:-1] + 1.0)))
            if len(dts) >= 2
            else 0.0
        )
        mod60_err = float(np.median(np.minimum(dts % 60.0, 60.0 - (dts % 60.0))))

        # 2. Node degrees, intermediaries, causal ordering, and running balance
        in_cnt = pd.Series(dst).value_counts()
        out_cnt = pd.Series(src).value_counts()
        w_tab = pd.DataFrame({'in_c': in_cnt, 'out_c': out_cnt}).fillna(0)
        all_w = w_tab.index.tolist()
        mid_w = set(w_tab[(w_tab['in_c'] > 0) & (w_tab['out_c'] > 0)].index)
        pure_src = set(w_tab[w_tab['in_c'] == 0].index)
        pure_dst = set(w_tab[w_tab['out_c'] == 0].index)

        first_in_ts = {}
        for i, d_w in enumerate(dst):
            if d_w not in first_in_ts:
                first_in_ts[d_w] = ts[i]

        causal_checks = [
            1.0 if ts[i] >= first_in_ts.get(src[i], 1e18) else 0.0
            for i in range(n_tx)
            if src[i] in mid_w
        ]
        causal_ok_rate = float(np.mean(causal_checks)) if causal_checks else 1.0

        # Running balance & intermediary total pass-through
        run_bal = {w: 0.0 for w in mid_w}
        prefund_viol = {w: False for w in mid_w}
        for i in range(n_tx):
            s_w, d_w = src[i], dst[i]
            if s_w in mid_w:
                if run_bal[s_w] * 1.05 + 1.0 < u_s[i]:
                    prefund_viol[s_w] = True
                run_bal[s_w] -= u_s[i]
            if d_w in mid_w:
                run_bal[d_w] += u_r[i]
        prefund_ok_rate = (
            1.0 - float(np.mean(list(prefund_viol.values()))) if mid_w else 1.0
        )

        mid_ratios = []
        for w in mid_w:
            sum_in = float(np.sum(u_r[dst == w]))
            sum_out = float(np.sum(u_s[src == w]))
            if sum_in > 0:
                mid_ratios.append(sum_out / sum_in)
        mid_pass_med = float(np.median(mid_ratios)) if mid_ratios else 1.0
        mid_pass_iqr = (
            float(np.percentile(mid_ratios, 75) - np.percentile(mid_ratios, 25))
            if len(mid_ratios) >= 2
            else 0.0
        )

        # 3. Causal hop-by-hop matching (waiting times, 1-to-1 fee cuts, asset handoff)
        hop_waits, hop_1to1_rats, hop_asset_match = [], [], []
        for w in mid_w:
            in_idx = np.flatnonzero(dst == w)
            out_idx = np.flatnonzero(src == w)
            for o_i in out_idx:
                prev_ins = in_idx[ts[in_idx] <= ts[o_i]]
                if len(prev_ins) > 0:
                    last_in = prev_ins[-1]
                    hop_waits.append((ts[o_i] - ts[last_in]) / 60.0)
                    hop_asset_match.append(1.0 if a_s[o_i] == a_r[last_in] else 0.0)
                    if len(in_idx) == 1 and len(out_idx) == 1:
                        hop_1to1_rats.append(u_s[o_i] / (u_r[last_in] + 1e-6))
                    else:
                        # Match closest-amount prior incoming tx on this node
                        rats = u_s[o_i] / (u_r[prev_ins] + 1e-6)
                        best_k = int(np.argmin(np.abs(rats - 0.90)))
                        if 0.65 <= rats[best_k] <= 1.05:
                            hop_1to1_rats.append(float(rats[best_k]))

        hop_wait_med_m = float(np.median(hop_waits)) if hop_waits else 0.0
        hop_wait_cv = (
            float(np.std(hop_waits) / (np.mean(hop_waits) + 1e-3))
            if len(hop_waits) >= 2
            else 0.0
        )
        hop1to1_med = float(np.median(hop_1to1_rats)) if hop_1to1_rats else mid_pass_med
        hop1to1_std = float(np.std(hop_1to1_rats)) if len(hop_1to1_rats) >= 2 else 0.0
        asset_handoff = float(np.mean(hop_asset_match)) if hop_asset_match else 1.0

        # 4. End-to-end conservation, monotone decay, split evenness, and rounding
        src_u_sum = float(np.sum(u_s[np.isin(src, list(pure_src))])) if pure_src else float(u_s[0])
        dst_u_sum = float(np.sum(u_r[np.isin(dst, list(pure_dst))])) if pure_dst else float(u_r[-1])
        sink_src_rat = dst_u_sum / (src_u_sum + 1e-6)
        last_first_rat = float(u_s[-1] / (u_s[0] + 1e-6))

        rk_u = pd.Series(u_s).rank().values
        rk_t = np.arange(n_tx, dtype=float)
        time_usd_corr = (
            float(np.corrcoef(rk_t, rk_u)[0, 1]) if np.std(rk_u) > 0 else 0.0
        )

        branch_cvs = []
        for w in all_w:
            if w_tab.loc[w, 'out_c'] >= 2:
                vals = u_s[src == w]
                branch_cvs.append(float(np.std(vals) / (np.mean(vals) + 1e-6)))
            if w_tab.loc[w, 'in_c'] >= 2:
                vals = u_r[dst == w]
                branch_cvs.append(float(np.std(vals) / (np.mean(vals) + 1e-6)))
        branch_split_cv = float(np.mean(branch_cvs)) if branch_cvs else 0.0

        src_sum_round = float(
            min(src_u_sum % 10.0, 10.0 - (src_u_sum % 10.0)) / 5.0
        )

        # 5. Conversions, spreads, and asset diversity
        conv_share = float(np.mean(conv))
        conv_alt_rate = float(np.mean(conv[1:] != conv[:-1]))
        n_assets = float(len(set(a_s) | set(a_r)))
        c_mask = conv > 0.5
        spread_bps = (
            float(np.median(1e4 * (1.0 - u_r[c_mask] / (u_s[c_mask] + 1e-6))))
            if np.any(c_mask)
            else 0.0
        )

        # 6. Weekly pair recurrence & concurrent/adjacent background wallet activity
        scen_pairs = list(zip(np.minimum(src, dst), np.maximum(src, dst)))
        scen_pair_cnt = pd.Series(scen_pairs).value_counts().to_dict()
        reused_pairs = sum(
            1 for p, c in scen_pair_cnt.items() if pair_keys.get(p, 0) > c
        )
        pair_reused_wk = float(reused_pairs) / max(1.0, float(len(scen_pair_cnt)))

        t0, t1 = ts[0], ts[-1]
        concur_bg, quiet_2h, wk_bg_cnts = [], [], []
        for w in all_w:
            w_scen_c = int(w_tab.loc[w, 'in_c'] + w_tab.loc[w, 'out_c'])
            w_ts = w_ts_map.get(w, np.empty(0))
            in_win = int(
                np.searchsorted(w_ts, t1, 'right') - np.searchsorted(w_ts, t0, 'left')
            )
            in_2h = int(
                np.searchsorted(w_ts, t1 + 7200.0, 'right')
                - np.searchsorted(w_ts, t0 - 7200.0, 'left')
            )
            concur_bg.append(max(0, in_win - w_scen_c))
            quiet_2h.append(1.0 if max(0, in_2h - w_scen_c) == 0 else 0.0)
            wk_bg_cnts.append(max(0, global_w_cnt.get(w, 0) - w_scen_c))

        src_wk_med = (
            float(np.median([global_w_cnt.get(w, 0) for w in pure_src]))
            if pure_src
            else float(np.median(wk_bg_cnts))
        )
        dst_wk_med = (
            float(np.median([global_w_cnt.get(w, 0) for w in pure_dst]))
            if pure_dst
            else float(np.median(wk_bg_cnts))
        )

        records.append({
            'family': str(fam),
            'grp': grp_type,
            '01_n_tx': float(n_tx),
            '02_dur_min': dur_s / 60.0,
            '03_gap_med_m': gap_med_m,
            '04_gap_cv': gap_cv,
            '05_gap_max_sh': gap_max_sh,
            '06_gap_adj_rel': gap_adj_rel,
            '07_gap_mod60_err': mod60_err,
            '08_causal_ok': causal_ok_rate,
            '09_prefund_ok': prefund_ok_rate,
            '10_hop_wait_m': hop_wait_med_m,
            '11_hop_wait_cv': hop_wait_cv,
            '12_mid_pass_med': mid_pass_med,
            '13_mid_pass_iqr': mid_pass_iqr,
            '14_hop1to1_med': hop1to1_med,
            '15_hop1to1_std': hop1to1_std,
            '16_sink_src_rat': sink_src_rat,
            '17_last_first_u': last_first_rat,
            '18_time_u_corr': time_usd_corr,
            '19_usd_med': float(np.median(u_s)),
            '20_usd_cv': float(np.std(u_s) / (np.mean(u_s) + 1e-6)),
            '21_split_cv': branch_split_cv,
            '22_round_share': float(g['is_round'].mean()),
            '23_src_u_round': src_sum_round,
            '24_conv_share': conv_share,
            '25_same_cur_swp': float(g['same_cur_swap'].mean()),
            '26_asset_hndoff': asset_handoff,
            '27_conv_alt': conv_alt_rate,
            '28_n_assets': n_assets,
            '29_spread_bps': spread_bps,
            '30_n_legs_mean': float(np.mean(n_legs)),
            '31_sendmax_sh': float(np.mean(sendmax)),
            '32_pair_reused': pair_reused_wk,
            '33_concur_bg_tx': float(np.mean(concur_bg)),
            '34_quiet_2h_sh': float(np.mean(quiet_2h)),
            '35_wk_bg_tx_med': float(np.median(wk_bg_cnts)),
        })

    res = pd.DataFrame(records)
    stat_cols = [c for c in res.columns if c not in ('family', 'grp')]

    def _calc_auc(v1: np.ndarray, v0: np.ndarray) -> float:
        if len(v1) == 0 or len(v0) == 0:
            return 0.5
        all_v = np.concatenate([v1, v0])
        rks = pd.Series(all_v).rank(method='average').values
        n1 = len(v1)
        return float((np.sum(rks[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * len(v0)))

    m_df = res[res['grp'] == 'med']
    i_df = res[res['grp'] == 'inno']
    e_df_s = res[res['grp'] == 'easy']

    lines = [
        f'=== MEDIUM vs LOOK-ALIKE vs EASY (med={len(m_df)}, inno={len(i_df)}, easy={len(e_df_s)}) ===',
        f"{'Stat':<15}|{'Med p50(p10,p90)':<20}|{'Inno p50(p10,p90)':<20}|{'Easy50':<7}|AUC",
        '-' * 70,
    ]

    global_aucs = {}
    fam_aucs = {c: {} for c in stat_cols}
    fams = sorted(res['family'].unique())

    for col in stat_cols:
        vm = m_df[col].dropna().values
        vi = i_df[col].dropna().values
        ve = e_df_s[col].dropna().values
        auc_g = _calc_auc(vm, vi)
        global_aucs[col] = auc_g

        pm50, pm10, pm90 = np.percentile(vm, [50, 10, 90]) if len(vm) else (0, 0, 0)
        pi50, pi10, pi90 = np.percentile(vi, [50, 10, 90]) if len(vi) else (0, 0, 0)
        pe50 = float(np.median(ve)) if len(ve) else 0.0

        sm = f'{pm50:.2f}({pm10:.2f},{pm90:.2f})'
        si = f'{pi50:.2f}({pi10:.2f},{pi90:.2f})'
        lines.append(f'{col:<15}|{sm:<20}|{si:<20}|{pe50:<7.2f}|{auc_g:.3f}')

        for f in fams:
            v_fm = m_df[m_df['family'] == f][col].dropna().values
            v_fi = i_df[i_df['family'] == f][col].dropna().values
            fam_aucs[col][f] = _calc_auc(v_fm, v_fi)

    # Top 5 by global |AUC - 0.5| plus Top 5 by mean within-family |AUC - 0.5|
    top_global = sorted(stat_cols, key=lambda c: abs(global_aucs[c] - 0.5), reverse=True)[:5]
    mean_fam_dev = {
        c: float(np.mean([abs(fam_aucs[c][f] - 0.5) for f in fams])) for c in stat_cols
    }
    top_fam = sorted(stat_cols, key=lambda c: mean_fam_dev[c], reverse=True)[:5]
    show_cols = list(dict.fromkeys(top_global + top_fam))

    short_f = {f: f[:4] for f in fams}
    short_f['fan_out_in'] = 'foi'
    short_f['fan_out'] = 'fout'
    short_f['fan_in'] = 'fin'
    short_f['multi_stage'] = 'mstg'

    lines.append('')
    cnt_str = ' '.join(
        f"{short_f.get(f, f[:4])}:{len(m_df[m_df['family']==f])}/{len(i_df[i_df['family']==f])}"
        for f in fams
    )
    lines.append(f'=== PER-FAMILY AUC (top global & within-family stats) [{cnt_str}] ===')
    hdr = f"{'Stat':<15}|" + '|'.join(f'{short_f.get(f, f[:4]):<5}' for f in fams) + '|MeanD'
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for col in show_cols:
        row_str = '|'.join(f'{fam_aucs[col][f]:.2f} ' for f in fams)
        lines.append(f'{col:<15}|{row_str}|{mean_fam_dev[col]:.3f}')

    return '\n'.join(lines)[:5950]


#FULLTEXT
# Gemini said
# Python
# import numpy as np
# import pandas as pd
# 
# 
# def medium_report(e: pd.DataFrame, lab: pd.DataFrame) -> str:
#     # Align edge and label tables on TRANSACTION_HASH
#     e_df = e.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in e.columns else e.copy()
#     l_df = lab.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in lab.columns else lab.copy()
# 
#     # Precompute global weekly wallet activity and pair recurrence on full 40k tx
#     e_sorted = e_df.sort_values('TS')
#     all_src = e_sorted['SRC'].values
#     all_dst = e_sorted['DST'].values
#     all_ts = e_sorted['TS'].values.astype(np.float64)
# 
#     w_all = pd.concat([e_df['SRC'], e_df['DST']])
#     global_w_cnt = w_all.value_counts().to_dict()
# 
#     # Build sorted timestamp list per wallet across the entire week
#     w_ev_df = pd.DataFrame({
#         'w': np.concatenate([all_src, all_dst]),
#         'ts': np.concatenate([all_ts, all_ts]),
#     }).sort_values('ts', kind='mergesort')
#     w_keys, w_counts = np.unique(w_ev_df['w'].values, return_counts=True)
#     w_ord = np.argsort(w_ev_df['w'].values, kind='mergesort')
#     w_ts_splits = np.split(w_ev_df['ts'].values[w_ord], np.cumsum(w_counts)[:-1])
#     w_ts_map = dict(zip(w_keys, w_ts_splits))
# 
#     # Undirected wallet-pair weekly counts
#     u_min = np.minimum(all_src, all_dst)
#     u_max = np.maximum(all_src, all_dst)
#     pair_keys = pd.Series(list(zip(u_min, u_max))).value_counts().to_dict()
# 
#     # Filter to non-relay, non-background motif scenarios
#     exclude_fams = {'aggregate_relay', 'distributed_relay', 'background', 'none', ''}
#     valid_mask = l_df['scenario_id'].notna() & (
#         ~l_df['family'].astype(str).str.lower().isin(exclude_fams)
#     )
#     l_sub = l_df[valid_mask].copy()
#     df = e_df.join(l_sub[['is_suspicious', 'family', 'difficulty', 'scenario_id']], how='inner')
#     df['is_suspicious'] = df['is_suspicious'].astype(int)
# 
#     # Parse currencies and issuers
#     s_sp = df['ASSET_SENT'].astype(str).str.split(':', n=1, expand=True)
#     r_sp = df['ASSET_RECV'].astype(str).str.split(':', n=1, expand=True)
#     df['s_cur'] = s_sp[1].fillna('XRP') if s_sp.shape[1] > 1 else 'XRP'
#     df['r_cur'] = r_sp[1].fillna('XRP') if r_sp.shape[1] > 1 else 'XRP'
#     df['same_cur_swap'] = ((df['IS_CONV'] > 0) & (df['s_cur'] == df['r_cur'])).astype(float)
# 
#     us_all = df['USD_SENT'].values.astype(float)
#     ur_all = df['USD_RECV'].values.astype(float)
#     vs_all = df['V_SENT'].values.astype(float)
#     df['is_round'] = (
#         (np.abs(us_all - np.round(us_all)) < 0.02)
#         | (np.abs(us_all / 10.0 - np.round(us_all / 10.0)) * 10.0 < 0.05)
#         | (np.abs(ur_all - np.round(ur_all)) < 0.02)
#         | (np.abs(vs_all - np.round(vs_all)) < 0.01)
#     ).astype(float)
# 
#     records = []
#     for (fam, sc_id), g in df.groupby(['family', 'scenario_id'], sort=False):
#         g = g.sort_values('TS', kind='mergesort')
#         n_tx = len(g)
#         if n_tx < 2:
#             continue
# 
#         susp = int(g['is_suspicious'].iloc[0])
#         diff = str(g['difficulty'].iloc[0]).lower()
#         if susp == 1 and 'med' in diff:
#             grp_type = 'med'
#         elif susp == 1 and 'eas' in diff:
#             grp_type = 'easy'
#         elif susp == 0:
#             grp_type = 'inno'
#         else:
#             continue
# 
#         ts = g['TS'].values.astype(float)
#         u_s = g['USD_SENT'].values.astype(float)
#         u_r = g['USD_RECV'].values.astype(float)
#         conv = g['IS_CONV'].values.astype(float)
#         n_legs = g['N_LEGS'].values.astype(float)
#         sendmax = g['HAS_SENDMAX'].values.astype(float)
#         a_s = g['ASSET_SENT'].astype(str).values
#         a_r = g['ASSET_RECV'].astype(str).values
#         src = g['SRC'].values
#         dst = g['DST'].values
# 
#         # 1. Timing & Gap distribution
#         dur_s = max(1.0, ts[-1] - ts[0])
#         dts = np.diff(ts)
#         gap_med_m = float(np.median(dts) / 60.0)
#         gap_cv = float(np.std(dts) / (np.mean(dts) + 1e-3))
#         gap_max_sh = float(np.max(dts) / dur_s)
#         gap_adj_rel = (
#             float(np.median(np.abs(dts[1:] - dts[:-1]) / (dts[1:] + dts[:-1] + 1.0)))
#             if len(dts) >= 2
#             else 0.0
#         )
#         mod60_err = float(np.median(np.minimum(dts % 60.0, 60.0 - (dts % 60.0))))
# 
#         # 2. Node degrees, intermediaries, causal ordering, and running balance
#         in_cnt = pd.Series(dst).value_counts()
#         out_cnt = pd.Series(src).value_counts()
#         w_tab = pd.DataFrame({'in_c': in_cnt, 'out_c': out_cnt}).fillna(0)
#         all_w = w_tab.index.tolist()
#         mid_w = set(w_tab[(w_tab['in_c'] > 0) & (w_tab['out_c'] > 0)].index)
#         pure_src = set(w_tab[w_tab['in_c'] == 0].index)
#         pure_dst = set(w_tab[w_tab['out_c'] == 0].index)
# 
#         first_in_ts = {}
#         for i, d_w in enumerate(dst):
#             if d_w not in first_in_ts:
#                 first_in_ts[d_w] = ts[i]
# 
#         causal_checks = [
#             1.0 if ts[i] >= first_in_ts.get(src[i], 1e18) else 0.0
#             for i in range(n_tx)
#             if src[i] in mid_w
#         ]
#         causal_ok_rate = float(np.mean(causal_checks)) if causal_checks else 1.0
# 
#         # Running balance & intermediary total pass-through
#         run_bal = {w: 0.0 for w in mid_w}
#         prefund_viol = {w: False for w in mid_w}
#         for i in range(n_tx):
#             s_w, d_w = src[i], dst[i]
#             if s_w in mid_w:
#                 if run_bal[s_w] * 1.05 + 1.0 < u_s[i]:
#                     prefund_viol[s_w] = True
#                 run_bal[s_w] -= u_s[i]
#             if d_w in mid_w:
#                 run_bal[d_w] += u_r[i]
#         prefund_ok_rate = (
#             1.0 - float(np.mean(list(prefund_viol.values()))) if mid_w else 1.0
#         )
# 
#         mid_ratios = []
#         for w in mid_w:
#             sum_in = float(np.sum(u_r[dst == w]))
#             sum_out = float(np.sum(u_s[src == w]))
#             if sum_in > 0:
#                 mid_ratios.append(sum_out / sum_in)
#         mid_pass_med = float(np.median(mid_ratios)) if mid_ratios else 1.0
#         mid_pass_iqr = (
#             float(np.percentile(mid_ratios, 75) - np.percentile(mid_ratios, 25))
#             if len(mid_ratios) >= 2
#             else 0.0
#         )
# 
#         # 3. Causal hop-by-hop matching (waiting times, 1-to-1 fee cuts, asset handoff)
#         hop_waits, hop_1to1_rats, hop_asset_match = [], [], []
#         for w in mid_w:
#             in_idx = np.flatnonzero(dst == w)
#             out_idx = np.flatnonzero(src == w)
#             for o_i in out_idx:
#                 prev_ins = in_idx[ts[in_idx] <= ts[o_i]]
#                 if len(prev_ins) > 0:
#                     last_in = prev_ins[-1]
#                     hop_waits.append((ts[o_i] - ts[last_in]) / 60.0)
#                     hop_asset_match.append(1.0 if a_s[o_i] == a_r[last_in] else 0.0)
#                     if len(in_idx) == 1 and len(out_idx) == 1:
#                         hop_1to1_rats.append(u_s[o_i] / (u_r[last_in] + 1e-6))
#                     else:
#                         # Match closest-amount prior incoming tx on this node
#                         rats = u_s[o_i] / (u_r[prev_ins] + 1e-6)
#                         best_k = int(np.argmin(np.abs(rats - 0.90)))
#                         if 0.65 <= rats[best_k] <= 1.05:
#                             hop_1to1_rats.append(float(rats[best_k]))
# 
#         hop_wait_med_m = float(np.median(hop_waits)) if hop_waits else 0.0
#         hop_wait_cv = (
#             float(np.std(hop_waits) / (np.mean(hop_waits) + 1e-3))
#             if len(hop_waits) >= 2
#             else 0.0
#         )
#         hop1to1_med = float(np.median(hop_1to1_rats)) if hop_1to1_rats else mid_pass_med
#         hop1to1_std = float(np.std(hop_1to1_rats)) if len(hop_1to1_rats) >= 2 else 0.0
#         asset_handoff = float(np.mean(hop_asset_match)) if hop_asset_match else 1.0
# 
#         # 4. End-to-end conservation, monotone decay, split evenness, and rounding
#         src_u_sum = float(np.sum(u_s[np.isin(src, list(pure_src))])) if pure_src else float(u_s[0])
#         dst_u_sum = float(np.sum(u_r[np.isin(dst, list(pure_dst))])) if pure_dst else float(u_r[-1])
#         sink_src_rat = dst_u_sum / (src_u_sum + 1e-6)
#         last_first_rat = float(u_s[-1] / (u_s[0] + 1e-6))
# 
#         rk_u = pd.Series(u_s).rank().values
#         rk_t = np.arange(n_tx, dtype=float)
#         time_usd_corr = (
#             float(np.corrcoef(rk_t, rk_u)[0, 1]) if np.std(rk_u) > 0 else 0.0
#         )
# 
#         branch_cvs = []
#         for w in all_w:
#             if w_tab.loc[w, 'out_c'] >= 2:
#                 vals = u_s[src == w]
#                 branch_cvs.append(float(np.std(vals) / (np.mean(vals) + 1e-6)))
#             if w_tab.loc[w, 'in_c'] >= 2:
#                 vals = u_r[dst == w]
#                 branch_cvs.append(float(np.std(vals) / (np.mean(vals) + 1e-6)))
#         branch_split_cv = float(np.mean(branch_cvs)) if branch_cvs else 0.0
# 
#         src_sum_round = float(
#             min(src_u_sum % 10.0, 10.0 - (src_u_sum % 10.0)) / 5.0
#         )
# 
#         # 5. Conversions, spreads, and asset diversity
#         conv_share = float(np.mean(conv))
#         conv_alt_rate = float(np.mean(conv[1:] != conv[:-1]))
#         n_assets = float(len(set(a_s) | set(a_r)))
#         c_mask = conv > 0.5
#         spread_bps = (
#             float(np.median(1e4 * (1.0 - u_r[c_mask] / (u_s[c_mask] + 1e-6))))
#             if np.any(c_mask)
#             else 0.0
#         )
# 
#         # 6. Weekly pair recurrence & concurrent/adjacent background wallet activity
#         scen_pairs = list(zip(np.minimum(src, dst), np.maximum(src, dst)))
#         scen_pair_cnt = pd.Series(scen_pairs).value_counts().to_dict()
#         reused_pairs = sum(
#             1 for p, c in scen_pair_cnt.items() if pair_keys.get(p, 0) > c
#         )
#         pair_reused_wk = float(reused_pairs) / max(1.0, float(len(scen_pair_cnt)))
# 
#         t0, t1 = ts[0], ts[-1]
#         concur_bg, quiet_2h, wk_bg_cnts = [], [], []
#         for w in all_w:
#             w_scen_c = int(w_tab.loc[w, 'in_c'] + w_tab.loc[w, 'out_c'])
#             w_ts = w_ts_map.get(w, np.empty(0))
#             in_win = int(
#                 np.searchsorted(w_ts, t1, 'right') - np.searchsorted(w_ts, t0, 'left')
#             )
#             in_2h = int(
#                 np.searchsorted(w_ts, t1 + 7200.0, 'right')
#                 - np.searchsorted(w_ts, t0 - 7200.0, 'left')
#             )
#             concur_bg.append(max(0, in_win - w_scen_c))
#             quiet_2h.append(1.0 if max(0, in_2h - w_scen_c) == 0 else 0.0)
#             wk_bg_cnts.append(max(0, global_w_cnt.get(w, 0) - w_scen_c))
# 
#         src_wk_med = (
#             float(np.median([global_w_cnt.get(w, 0) for w in pure_src]))
#             if pure_src
#             else float(np.median(wk_bg_cnts))
#         )
#         dst_wk_med = (
#             float(np.median([global_w_cnt.get(w, 0) for w in pure_dst]))
#             if pure_dst
#             else float(np.median(wk_bg_cnts))
#         )
# 
#         records.append({
#             'family': str(fam),
#             'grp': grp_type,
#             '01_n_tx': float(n_tx),
#             '02_dur_min': dur_s / 60.0,
#             '03_gap_med_m': gap_med_m,
#             '04_gap_cv': gap_cv,
#             '05_gap_max_sh': gap_max_sh,
#             '06_gap_adj_rel': gap_adj_rel,
#             '07_gap_mod60_err': mod60_err,
#             '08_causal_ok': causal_ok_rate,
#             '09_prefund_ok': prefund_ok_rate,
#             '10_hop_wait_m': hop_wait_med_m,
#             '11_hop_wait_cv': hop_wait_cv,
#             '12_mid_pass_med': mid_pass_med,
#             '13_mid_pass_iqr': mid_pass_iqr,
#             '14_hop1to1_med': hop1to1_med,
#             '15_hop1to1_std': hop1to1_std,
#             '16_sink_src_rat': sink_src_rat,
#             '17_last_first_u': last_first_rat,
#             '18_time_u_corr': time_usd_corr,
#             '19_usd_med': float(np.median(u_s)),
#             '20_usd_cv': float(np.std(u_s) / (np.mean(u_s) + 1e-6)),
#             '21_split_cv': branch_split_cv,
#             '22_round_share': float(g['is_round'].mean()),
#             '23_src_u_round': src_sum_round,
#             '24_conv_share': conv_share,
#             '25_same_cur_swp': float(g['same_cur_swap'].mean()),
#             '26_asset_hndoff': asset_handoff,
#             '27_conv_alt': conv_alt_rate,
#             '28_n_assets': n_assets,
#             '29_spread_bps': spread_bps,
#             '30_n_legs_mean': float(np.mean(n_legs)),
#             '31_sendmax_sh': float(np.mean(sendmax)),
#             '32_pair_reused': pair_reused_wk,
#             '33_concur_bg_tx': float(np.mean(concur_bg)),
#             '34_quiet_2h_sh': float(np.mean(quiet_2h)),
#             '35_wk_bg_tx_med': float(np.median(wk_bg_cnts)),
#         })
# 
#     res = pd.DataFrame(records)
#     stat_cols = [c for c in res.columns if c not in ('family', 'grp')]
# 
#     def _calc_auc(v1: np.ndarray, v0: np.ndarray) -> float:
#         if len(v1) == 0 or len(v0) == 0:
#             return 0.5
#         all_v = np.concatenate([v1, v0])
#         rks = pd.Series(all_v).rank(method='average').values
#         n1 = len(v1)
#         return float((np.sum(rks[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * len(v0)))
# 
#     m_df = res[res['grp'] == 'med']
#     i_df = res[res['grp'] == 'inno']
#     e_df_s = res[res['grp'] == 'easy']
# 
#     lines = [
#         f'=== MEDIUM vs LOOK-ALIKE vs EASY (med={len(m_df)}, inno={len(i_df)}, easy={len(e_df_s)}) ===',
#         f"{'Stat':<15}|{'Med p50(p10,p90)':<20}|{'Inno p50(p10,p90)':<20}|{'Easy50':<7}|AUC",
#         '-' * 70,
#     ]
# 
#     global_aucs = {}
#     fam_aucs = {c: {} for c in stat_cols}
#     fams = sorted(res['family'].unique())
# 
#     for col in stat_cols:
#         vm = m_df[col].dropna().values
#         vi = i_df[col].dropna().values
#         ve = e_df_s[col].dropna().values
#         auc_g = _calc_auc(vm, vi)
#         global_aucs[col] = auc_g
# 
#         pm50, pm10, pm90 = np.percentile(vm, [50, 10, 90]) if len(vm) else (0, 0, 0)
#         pi50, pi10, pi90 = np.percentile(vi, [50, 10, 90]) if len(vi) else (0, 0, 0)
#         pe50 = float(np.median(ve)) if len(ve) else 0.0
# 
#         sm = f'{pm50:.2f}({pm10:.2f},{pm90:.2f})'
#         si = f'{pi50:.2f}({pi10:.2f},{pi90:.2f})'
#         lines.append(f'{col:<15}|{sm:<20}|{si:<20}|{pe50:<7.2f}|{auc_g:.3f}')
# 
#         for f in fams:
#             v_fm = m_df[m_df['family'] == f][col].dropna().values
#             v_fi = i_df[i_df['family'] == f][col].dropna().values
#             fam_aucs[col][f] = _calc_auc(v_fm, v_fi)
# 
#     # Top 5 by global |AUC - 0.5| plus Top 5 by mean within-family |AUC - 0.5|
#     top_global = sorted(stat_cols, key=lambda c: abs(global_aucs[c] - 0.5), reverse=True)[:5]
#     mean_fam_dev = {
#         c: float(np.mean([abs(fam_aucs[c][f] - 0.5) for f in fams])) for c in stat_cols
#     }
#     top_fam = sorted(stat_cols, key=lambda c: mean_fam_dev[c], reverse=True)[:5]
#     show_cols = list(dict.fromkeys(top_global + top_fam))
# 
#     short_f = {f: f[:4] for f in fams}
#     short_f['fan_out_in'] = 'foi'
#     short_f['fan_out'] = 'fout'
#     short_f['fan_in'] = 'fin'
#     short_f['multi_stage'] = 'mstg'
# 
#     lines.append('')
#     cnt_str = ' '.join(
#         f"{short_f.get(f, f[:4])}:{len(m_df[m_df['family']==f])}/{len(i_df[i_df['family']==f])}"
#         for f in fams
#     )
#     lines.append(f'=== PER-FAMILY AUC (top global & within-family stats) [{cnt_str}] ===')
#     hdr = f"{'Stat':<15}|" + '|'.join(f'{short_f.get(f, f[:4]):<5}' for f in fams) + '|MeanD'
#     lines.append(hdr)
#     lines.append('-' * len(hdr))
#     for col in show_cols:
#         row_str = '|'.join(f'{fam_aucs[col][f]:.2f} ' for f in fams)
#         lines.append(f'{col:<15}|{row_str}|{mean_fam_dev[col]:.3f}')
# 
#     return '\n'.join(lines)[:5950]
# 