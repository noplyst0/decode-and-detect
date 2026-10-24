#CODE
import numpy as np
import pandas as pd


def relay_report(e: pd.DataFrame, lab: pd.DataFrame) -> str:
    # Align edge table and label table on TRANSACTION_HASH
    e_df = e.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in e.columns else e.copy()
    l_df = lab.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in lab.columns else lab.copy()

    # Global wallet transaction counts across the entire week (all 40k tx)
    w_all = pd.concat([e_df['SRC'], e_df['DST']])
    global_w_cnt = w_all.value_counts()

    # Restrict to the two relay families
    relay_fams = ['aggregate_relay', 'distributed_relay']
    l_rel = l_df[l_df['family'].isin(relay_fams)].copy()
    df = e_df.join(l_rel[['is_suspicious', 'family', 'scenario_id']], how='inner')
    df['is_suspicious'] = df['is_suspicious'].astype(int)

    # Parse currency and issuer codes
    s_split = df['ASSET_SENT'].astype(str).str.split(':', n=1, expand=True)
    r_split = df['ASSET_RECV'].astype(str).str.split(':', n=1, expand=True)
    df['s_cur'] = s_split[1].fillna('XRP') if s_split.shape[1] > 1 else 'XRP'
    df['r_cur'] = r_split[1].fillna('XRP') if r_split.shape[1] > 1 else 'XRP'
    df['same_cur_swap'] = ((df['IS_CONV'] > 0) & (df['s_cur'] == df['r_cur'])).astype(float)

    # Round USD / native unit check
    us = df['USD_SENT'].values.astype(float)
    vs = df['V_SENT'].values.astype(float)
    df['is_round'] = (
        (np.abs(us - np.round(us)) < 0.02)
        | (np.abs(us / 5.0 - np.round(us / 5.0)) * 5.0 < 0.03)
        | (np.abs(vs - np.round(vs)) < 0.01)
    ).astype(float)

    records = []
    for (fam, sc_id), g in df.groupby(['family', 'scenario_id'], sort=False):
        g = g.sort_values('TS')
        n_tx = len(g)
        susp = int(g['is_suspicious'].iloc[0])
        ts = g['TS'].values.astype(float)
        u_s = g['USD_SENT'].values.astype(float)
        u_r = g['USD_RECV'].values.astype(float)
        conv = g['IS_CONV'].values.astype(float)
        src = g['SRC'].values
        dst = g['DST'].values

        # Identify collector wallet: highest in-degree among wallets that also send
        in_s = pd.Series(dst).value_counts()
        out_s = pd.Series(src).value_counts()
        w_stats = pd.DataFrame({'in_c': in_s, 'out_c': out_s}).fillna(0)
        mid_w = w_stats[(w_stats['in_c'] > 0) & (w_stats['out_c'] > 0)]
        if len(mid_w) > 0:
            collector = mid_w.sort_values(['in_c', 'out_c'], ascending=False).index[0]
        else:
            collector = w_stats.sort_values('in_c', ascending=False).index[0]

        # Classify scenario transactions: collection (into C), sink (to final receivers), mid-hop
        sink_wallets = set(w_stats[w_stats['out_c'] == 0].index) - {collector}
        mid_wallets = set(mid_w.index) - {collector}
        is_coll = dst == collector
        is_sink = np.array([d in sink_wallets for d in dst], dtype=bool)
        if not np.any(is_sink):
            is_sink = ~is_coll
        is_mid_hop = (src == collector) & np.array([d in mid_wallets for d in dst], dtype=bool)

        # Segment into episodes using >4h (14,400s) gaps between collection txs
        c_ts = ts[is_coll]
        c_us = u_s[is_coll]
        c_ur = u_r[is_coll]
        if len(c_ts) > 0:
            ep_breaks = np.where(np.diff(c_ts) > 14400.0)[0] + 1
            ep_c_idx = np.split(np.arange(len(c_ts)), ep_breaks)
            ep_starts = np.array([c_ts[idx[0]] for idx in ep_c_idx])
        else:
            ep_c_idx = [np.array([], dtype=int)]
            ep_starts = np.array([ts[0]])

        n_ep = len(ep_starts)
        tx_ep = np.clip(np.searchsorted(ep_starts, ts, side='right') - 1, 0, n_ep - 1)

        ep_in_usd, ep_out_usd, ep_ratios = [], [], []
        ep_hold_h, ep_cdur_m, ep_odur_m, ep_cgap_cv, ep_hop_m = [], [], [], [], []
        ep_senders, ep_recvs = [], []

        for k in range(n_ep):
            m_k = tx_ep == k
            m_in = m_k & is_coll
            m_out = m_k & is_sink
            m_mid = m_k & is_mid_hop

            in_sum = float(np.sum(u_r[m_in]))
            out_sum = float(np.sum(u_s[m_out]))
            if in_sum > 0:
                ep_in_usd.append(in_sum)
                if out_sum > 0:
                    ep_out_usd.append(out_sum)
                    ep_ratios.append(out_sum / in_sum)

            t_in = ts[m_in]
            t_out = ts[m_out]
            t_pay = ts[m_k & (~is_coll)]
            if len(t_in) > 0:
                ep_cdur_m.append((t_in[-1] - t_in[0]) / 60.0)
                ep_senders.append( set(src[m_in]) )
                if len(t_in) >= 3:
                    dts = np.diff(t_in)
                    ep_cgap_cv.append(float(np.std(dts) / (np.mean(dts) + 1.0)))
            if len(t_out) > 0:
                ep_odur_m.append((t_out[-1] - t_out[0]) / 60.0)
                ep_recvs.append( set(dst[m_out]) )
            if len(t_in) > 0 and len(t_pay) > 0:
                ep_hold_h.append(max(0.0, (t_pay[0] - t_in[-1]) / 3600.0))
            if np.any(m_mid) and len(t_out) > 0:
                ep_hop_m.append(max(0.0, (t_out[0] - ts[m_mid][0]) / 60.0))

        # Pairwise consecutive episode Jaccard overlap for senders & receivers
        s_jacc = [
            len(ep_senders[i] & ep_senders[i + 1]) / max(1, len(ep_senders[i] | ep_senders[i + 1]))
            for i in range(len(ep_senders) - 1)
        ]
        r_jacc = [
            len(ep_recvs[i] & ep_recvs[i + 1]) / max(1, len(ep_recvs[i] | ep_recvs[i + 1]))
            for i in range(len(ep_recvs) - 1)
        ]

        # Intermediary (relay node) pass-through ratio
        if len(mid_wallets) > 0:
            m_in_mid = np.array([d in mid_wallets for d in dst], dtype=bool)
            m_out_mid = np.array([s in mid_wallets for s in src], dtype=bool)
            mid_pass = float(np.sum(u_s[m_out_mid]) / (np.sum(u_r[m_in_mid]) + 1e-6))
        else:
            mid_pass = float(np.sum(u_s[src == collector]) / (np.sum(u_r[is_coll]) + 1e-6))

        # Episode-to-episode growth slope
        if len(ep_in_usd) >= 2:
            x_idx = np.arange(len(ep_in_usd), dtype=float)
            y_val = np.array(ep_in_usd, dtype=float)
            slope = float(np.polyfit(x_idx, y_val / (np.mean(y_val) + 1e-6), 1)[0])
        else:
            slope = 0.0

        ep_gaps_h = np.diff(ep_starts) / 3600.0 if n_ep >= 2 else np.array([0.0])
        ep_gap_cv = (
            float(np.std(ep_gaps_h) / (np.mean(ep_gaps_h) + 1e-3)) if n_ep >= 3 else 0.0
        )

        # Circadian concentration (mean resultant length R in [0,1]) of collection txs
        angles = 2.0 * np.pi * ((c_ts if len(c_ts) > 0 else ts) % 86400.0) / 86400.0
        tod_r = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))

        # Conversion spread in bps
        c_mask = conv > 0.5
        spread_bps = (
            float(np.median(1e4 * (1.0 - u_r[c_mask] / (u_s[c_mask] + 1e-6))))
            if np.any(c_mask)
            else float(np.median(1e4 * (1.0 - u_r / (u_s + 1e-6))))
        )

        # Background activity outside this scenario in the full week
        coll_scen_tx = int(w_stats.loc[collector, 'in_c'] + w_stats.loc[collector, 'out_c'])
        coll_bg_tx = float(max(0, int(global_w_cnt.get(collector, 0)) - coll_scen_tx))
        sink_bg = [
            max(0, int(global_w_cnt.get(w, 0)) - int(w_stats.loc[w, 'in_c']))
            for w in sink_wallets
        ]
        send_wallets = set(src[is_coll])
        send_bg = [
            max(0, int(global_w_cnt.get(w, 0)) - int(w_stats.loc[w, 'out_c']))
            for w in send_wallets
        ]

        u_in_arr = u_s[is_coll] if np.any(is_coll) else u_s
        u_out_arr = u_s[is_sink] if np.any(is_sink) else u_s

        records.append({
            'family': fam,
            'is_suspicious': susp,
            '01_usd_med': float(np.median(u_s)),
            '02_usd_in_cv': float(np.std(u_in_arr) / (np.mean(u_in_arr) + 1e-6)),
            '03_usd_out_cv': float(np.std(u_out_arr) / (np.mean(u_out_arr) + 1e-6)),
            '04_wk_pass_ratio': float(np.sum(u_s[is_sink]) / (np.sum(u_r[is_coll]) + 1e-6)),
            '05_coll_pass_ratio': float(
                np.sum(u_s[src == collector]) / (np.sum(u_r[is_coll]) + 1e-6)
            ),
            '06_mid_pass_ratio': mid_pass,
            '07_ep_pass_med': float(np.median(ep_ratios)) if ep_ratios else 0.0,
            '08_ep_pass_std': float(np.std(ep_ratios)) if len(ep_ratios) >= 2 else 0.0,
            '09_ep_cnt': float(n_ep),
            '10_ep_in_usd_cv': (
                float(np.std(ep_in_usd) / (np.mean(ep_in_usd) + 1e-6))
                if len(ep_in_usd) >= 2
                else 0.0
            ),
            '11_ep_in_slope': slope,
            '12_ep_gap_med_h': float(np.median(ep_gaps_h)),
            '13_ep_gap_cv': ep_gap_cv,
            '14_hold_delay_med_h': float(np.median(ep_hold_h)) if ep_hold_h else 0.0,
            '15_hold_delay_cv': (
                float(np.std(ep_hold_h) / (np.mean(ep_hold_h) + 1e-3))
                if len(ep_hold_h) >= 2
                else 0.0
            ),
            '16_coll_dur_med_m': float(np.median(ep_cdur_m)) if ep_cdur_m else 0.0,
            '17_coll_gap_cv': float(np.median(ep_cgap_cv)) if ep_cgap_cv else 0.0,
            '18_pay_dur_med_m': float(np.median(ep_odur_m)) if ep_odur_m else 0.0,
            '19_relay_hop_med_m': float(np.median(ep_hop_m)) if ep_hop_m else 0.0,
            '20_send_rep_rate': 1.0 - len(send_wallets) / max(1.0, float(np.sum(is_coll))),
            '21_recv_rep_rate': 1.0 - len(sink_wallets) / max(1.0, float(np.sum(is_sink))),
            '22_send_ep_jacc': float(np.mean(s_jacc)) if s_jacc else 0.0,
            '23_recv_ep_jacc': float(np.mean(r_jacc)) if r_jacc else 0.0,
            '24_mid_wallet_cnt': float(len(mid_wallets)),
            '25_conv_out_m_in': float(
                np.mean(conv[~is_coll]) - np.mean(conv[is_coll])
                if (np.any(is_coll) and np.any(~is_coll))
                else 0.0
            ),
            '26_same_cur_swap': float(g['same_cur_swap'].mean()),
            '27_spread_med_bps': spread_bps,
            '28_round_share': float(g['is_round'].mean()),
            '29_tod_conc_r': tod_r,
            '30_coll_bg_tx': coll_bg_tx,
            '31_sink_bg_med': float(np.median(sink_bg)) if sink_bg else 0.0,
            '32_send_bg_med': float(np.median(send_bg)) if send_bg else 0.0,
        })

    res_df = pd.DataFrame(records)
    stat_cols = [c for c in res_df.columns if c not in ('family', 'is_suspicious')]

    lines = []
    for fam in relay_fams:
        sub = res_df[res_df['family'] == fam]
        s1 = sub[sub['is_suspicious'] == 1]
        s0 = sub[sub['is_suspicious'] == 0]
        n1, n0 = len(s1), len(s0)
        lines.append(f'=== {fam.upper()} (susp={n1}, inno={n0}) ===')
        lines.append(f"{'Statistic':<19} | {'Susp p50(p10,p90)':<21} | {'Inno p50(p10,p90)':<21} | AUC")
        lines.append('-' * 72)

        for col in stat_cols:
            v1 = s1[col].dropna().values
            v0 = s0[col].dropna().values
            if len(v1) > 0 and len(v0) > 0:
                p50_1, p10_1, p90_1 = np.percentile(v1, [50, 10, 90])
                p50_0, p10_0, p90_0 = np.percentile(v0, [50, 10, 90])
                all_v = np.concatenate([v1, v0])
                ranks = pd.Series(all_v).rank(method='average').values
                auc = (np.sum(ranks[: len(v1)]) - len(v1) * (len(v1) + 1) / 2.0) / (
                    len(v1) * len(v0)
                )
            else:
                p50_1 = p10_1 = p90_1 = p50_0 = p10_0 = p90_0 = 0.0
                auc = 0.5

            s1_str = f'{p50_1:.2f}({p10_1:.2f},{p90_1:.2f})'
            s0_str = f'{p50_0:.2f}({p10_0:.2f},{p90_0:.2f})'
            lines.append(f'{col:<19} | {s1_str:<21} | {s0_str:<21} | {auc:.3f}')
        lines.append('')

    return '\n'.join(lines)[:5950]


#FULLTEXT
# Gemini said
# Python
# import numpy as np
# import pandas as pd
# 
# 
# def relay_report(e: pd.DataFrame, lab: pd.DataFrame) -> str:
#     # Align edge table and label table on TRANSACTION_HASH
#     e_df = e.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in e.columns else e.copy()
#     l_df = lab.set_index('TRANSACTION_HASH') if 'TRANSACTION_HASH' in lab.columns else lab.copy()
# 
#     # Global wallet transaction counts across the entire week (all 40k tx)
#     w_all = pd.concat([e_df['SRC'], e_df['DST']])
#     global_w_cnt = w_all.value_counts()
# 
#     # Restrict to the two relay families
#     relay_fams = ['aggregate_relay', 'distributed_relay']
#     l_rel = l_df[l_df['family'].isin(relay_fams)].copy()
#     df = e_df.join(l_rel[['is_suspicious', 'family', 'scenario_id']], how='inner')
#     df['is_suspicious'] = df['is_suspicious'].astype(int)
# 
#     # Parse currency and issuer codes
#     s_split = df['ASSET_SENT'].astype(str).str.split(':', n=1, expand=True)
#     r_split = df['ASSET_RECV'].astype(str).str.split(':', n=1, expand=True)
#     df['s_cur'] = s_split[1].fillna('XRP') if s_split.shape[1] > 1 else 'XRP'
#     df['r_cur'] = r_split[1].fillna('XRP') if r_split.shape[1] > 1 else 'XRP'
#     df['same_cur_swap'] = ((df['IS_CONV'] > 0) & (df['s_cur'] == df['r_cur'])).astype(float)
# 
#     # Round USD / native unit check
#     us = df['USD_SENT'].values.astype(float)
#     vs = df['V_SENT'].values.astype(float)
#     df['is_round'] = (
#         (np.abs(us - np.round(us)) < 0.02)
#         | (np.abs(us / 5.0 - np.round(us / 5.0)) * 5.0 < 0.03)
#         | (np.abs(vs - np.round(vs)) < 0.01)
#     ).astype(float)
# 
#     records = []
#     for (fam, sc_id), g in df.groupby(['family', 'scenario_id'], sort=False):
#         g = g.sort_values('TS')
#         n_tx = len(g)
#         susp = int(g['is_suspicious'].iloc[0])
#         ts = g['TS'].values.astype(float)
#         u_s = g['USD_SENT'].values.astype(float)
#         u_r = g['USD_RECV'].values.astype(float)
#         conv = g['IS_CONV'].values.astype(float)
#         src = g['SRC'].values
#         dst = g['DST'].values
# 
#         # Identify collector wallet: highest in-degree among wallets that also send
#         in_s = pd.Series(dst).value_counts()
#         out_s = pd.Series(src).value_counts()
#         w_stats = pd.DataFrame({'in_c': in_s, 'out_c': out_s}).fillna(0)
#         mid_w = w_stats[(w_stats['in_c'] > 0) & (w_stats['out_c'] > 0)]
#         if len(mid_w) > 0:
#             collector = mid_w.sort_values(['in_c', 'out_c'], ascending=False).index[0]
#         else:
#             collector = w_stats.sort_values('in_c', ascending=False).index[0]
# 
#         # Classify scenario transactions: collection (into C), sink (to final receivers), mid-hop
#         sink_wallets = set(w_stats[w_stats['out_c'] == 0].index) - {collector}
#         mid_wallets = set(mid_w.index) - {collector}
#         is_coll = dst == collector
#         is_sink = np.array([d in sink_wallets for d in dst], dtype=bool)
#         if not np.any(is_sink):
#             is_sink = ~is_coll
#         is_mid_hop = (src == collector) & np.array([d in mid_wallets for d in dst], dtype=bool)
# 
#         # Segment into episodes using >4h (14,400s) gaps between collection txs
#         c_ts = ts[is_coll]
#         c_us = u_s[is_coll]
#         c_ur = u_r[is_coll]
#         if len(c_ts) > 0:
#             ep_breaks = np.where(np.diff(c_ts) > 14400.0)[0] + 1
#             ep_c_idx = np.split(np.arange(len(c_ts)), ep_breaks)
#             ep_starts = np.array([c_ts[idx[0]] for idx in ep_c_idx])
#         else:
#             ep_c_idx = [np.array([], dtype=int)]
#             ep_starts = np.array([ts[0]])
# 
#         n_ep = len(ep_starts)
#         tx_ep = np.clip(np.searchsorted(ep_starts, ts, side='right') - 1, 0, n_ep - 1)
# 
#         ep_in_usd, ep_out_usd, ep_ratios = [], [], []
#         ep_hold_h, ep_cdur_m, ep_odur_m, ep_cgap_cv, ep_hop_m = [], [], [], [], []
#         ep_senders, ep_recvs = [], []
# 
#         for k in range(n_ep):
#             m_k = tx_ep == k
#             m_in = m_k & is_coll
#             m_out = m_k & is_sink
#             m_mid = m_k & is_mid_hop
# 
#             in_sum = float(np.sum(u_r[m_in]))
#             out_sum = float(np.sum(u_s[m_out]))
#             if in_sum > 0:
#                 ep_in_usd.append(in_sum)
#                 if out_sum > 0:
#                     ep_out_usd.append(out_sum)
#                     ep_ratios.append(out_sum / in_sum)
# 
#             t_in = ts[m_in]
#             t_out = ts[m_out]
#             t_pay = ts[m_k & (~is_coll)]
#             if len(t_in) > 0:
#                 ep_cdur_m.append((t_in[-1] - t_in[0]) / 60.0)
#                 ep_senders.append( set(src[m_in]) )
#                 if len(t_in) >= 3:
#                     dts = np.diff(t_in)
#                     ep_cgap_cv.append(float(np.std(dts) / (np.mean(dts) + 1.0)))
#             if len(t_out) > 0:
#                 ep_odur_m.append((t_out[-1] - t_out[0]) / 60.0)
#                 ep_recvs.append( set(dst[m_out]) )
#             if len(t_in) > 0 and len(t_pay) > 0:
#                 ep_hold_h.append(max(0.0, (t_pay[0] - t_in[-1]) / 3600.0))
#             if np.any(m_mid) and len(t_out) > 0:
#                 ep_hop_m.append(max(0.0, (t_out[0] - ts[m_mid][0]) / 60.0))
# 
#         # Pairwise consecutive episode Jaccard overlap for senders & receivers
#         s_jacc = [
#             len(ep_senders[i] & ep_senders[i + 1]) / max(1, len(ep_senders[i] | ep_senders[i + 1]))
#             for i in range(len(ep_senders) - 1)
#         ]
#         r_jacc = [
#             len(ep_recvs[i] & ep_recvs[i + 1]) / max(1, len(ep_recvs[i] | ep_recvs[i + 1]))
#             for i in range(len(ep_recvs) - 1)
#         ]
# 
#         # Intermediary (relay node) pass-through ratio
#         if len(mid_wallets) > 0:
#             m_in_mid = np.array([d in mid_wallets for d in dst], dtype=bool)
#             m_out_mid = np.array([s in mid_wallets for s in src], dtype=bool)
#             mid_pass = float(np.sum(u_s[m_out_mid]) / (np.sum(u_r[m_in_mid]) + 1e-6))
#         else:
#             mid_pass = float(np.sum(u_s[src == collector]) / (np.sum(u_r[is_coll]) + 1e-6))
# 
#         # Episode-to-episode growth slope
#         if len(ep_in_usd) >= 2:
#             x_idx = np.arange(len(ep_in_usd), dtype=float)
#             y_val = np.array(ep_in_usd, dtype=float)
#             slope = float(np.polyfit(x_idx, y_val / (np.mean(y_val) + 1e-6), 1)[0])
#         else:
#             slope = 0.0
# 
#         ep_gaps_h = np.diff(ep_starts) / 3600.0 if n_ep >= 2 else np.array([0.0])
#         ep_gap_cv = (
#             float(np.std(ep_gaps_h) / (np.mean(ep_gaps_h) + 1e-3)) if n_ep >= 3 else 0.0
#         )
# 
#         # Circadian concentration (mean resultant length R in [0,1]) of collection txs
#         angles = 2.0 * np.pi * ((c_ts if len(c_ts) > 0 else ts) % 86400.0) / 86400.0
#         tod_r = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))
# 
#         # Conversion spread in bps
#         c_mask = conv > 0.5
#         spread_bps = (
#             float(np.median(1e4 * (1.0 - u_r[c_mask] / (u_s[c_mask] + 1e-6))))
#             if np.any(c_mask)
#             else float(np.median(1e4 * (1.0 - u_r / (u_s + 1e-6))))
#         )
# 
#         # Background activity outside this scenario in the full week
#         coll_scen_tx = int(w_stats.loc[collector, 'in_c'] + w_stats.loc[collector, 'out_c'])
#         coll_bg_tx = float(max(0, int(global_w_cnt.get(collector, 0)) - coll_scen_tx))
#         sink_bg = [
#             max(0, int(global_w_cnt.get(w, 0)) - int(w_stats.loc[w, 'in_c']))
#             for w in sink_wallets
#         ]
#         send_wallets = set(src[is_coll])
#         send_bg = [
#             max(0, int(global_w_cnt.get(w, 0)) - int(w_stats.loc[w, 'out_c']))
#             for w in send_wallets
#         ]
# 
#         u_in_arr = u_s[is_coll] if np.any(is_coll) else u_s
#         u_out_arr = u_s[is_sink] if np.any(is_sink) else u_s
# 
#         records.append({
#             'family': fam,
#             'is_suspicious': susp,
#             '01_usd_med': float(np.median(u_s)),
#             '02_usd_in_cv': float(np.std(u_in_arr) / (np.mean(u_in_arr) + 1e-6)),
#             '03_usd_out_cv': float(np.std(u_out_arr) / (np.mean(u_out_arr) + 1e-6)),
#             '04_wk_pass_ratio': float(np.sum(u_s[is_sink]) / (np.sum(u_r[is_coll]) + 1e-6)),
#             '05_coll_pass_ratio': float(
#                 np.sum(u_s[src == collector]) / (np.sum(u_r[is_coll]) + 1e-6)
#             ),
#             '06_mid_pass_ratio': mid_pass,
#             '07_ep_pass_med': float(np.median(ep_ratios)) if ep_ratios else 0.0,
#             '08_ep_pass_std': float(np.std(ep_ratios)) if len(ep_ratios) >= 2 else 0.0,
#             '09_ep_cnt': float(n_ep),
#             '10_ep_in_usd_cv': (
#                 float(np.std(ep_in_usd) / (np.mean(ep_in_usd) + 1e-6))
#                 if len(ep_in_usd) >= 2
#                 else 0.0
#             ),
#             '11_ep_in_slope': slope,
#             '12_ep_gap_med_h': float(np.median(ep_gaps_h)),
#             '13_ep_gap_cv': ep_gap_cv,
#             '14_hold_delay_med_h': float(np.median(ep_hold_h)) if ep_hold_h else 0.0,
#             '15_hold_delay_cv': (
#                 float(np.std(ep_hold_h) / (np.mean(ep_hold_h) + 1e-3))
#                 if len(ep_hold_h) >= 2
#                 else 0.0
#             ),
#             '16_coll_dur_med_m': float(np.median(ep_cdur_m)) if ep_cdur_m else 0.0,
#             '17_coll_gap_cv': float(np.median(ep_cgap_cv)) if ep_cgap_cv else 0.0,
#             '18_pay_dur_med_m': float(np.median(ep_odur_m)) if ep_odur_m else 0.0,
#             '19_relay_hop_med_m': float(np.median(ep_hop_m)) if ep_hop_m else 0.0,
#             '20_send_rep_rate': 1.0 - len(send_wallets) / max(1.0, float(np.sum(is_coll))),
#             '21_recv_rep_rate': 1.0 - len(sink_wallets) / max(1.0, float(np.sum(is_sink))),
#             '22_send_ep_jacc': float(np.mean(s_jacc)) if s_jacc else 0.0,
#             '23_recv_ep_jacc': float(np.mean(r_jacc)) if r_jacc else 0.0,
#             '24_mid_wallet_cnt': float(len(mid_wallets)),
#             '25_conv_out_m_in': float(
#                 np.mean(conv[~is_coll]) - np.mean(conv[is_coll])
#                 if (np.any(is_coll) and np.any(~is_coll))
#                 else 0.0
#             ),
#             '26_same_cur_swap': float(g['same_cur_swap'].mean()),
#             '27_spread_med_bps': spread_bps,
#             '28_round_share': float(g['is_round'].mean()),
#             '29_tod_conc_r': tod_r,
#             '30_coll_bg_tx': coll_bg_tx,
#             '31_sink_bg_med': float(np.median(sink_bg)) if sink_bg else 0.0,
#             '32_send_bg_med': float(np.median(send_bg)) if send_bg else 0.0,
#         })
# 
#     res_df = pd.DataFrame(records)
#     stat_cols = [c for c in res_df.columns if c not in ('family', 'is_suspicious')]
# 
#     lines = []
#     for fam in relay_fams:
#         sub = res_df[res_df['family'] == fam]
#         s1 = sub[sub['is_suspicious'] == 1]
#         s0 = sub[sub['is_suspicious'] == 0]
#         n1, n0 = len(s1), len(s0)
#         lines.append(f'=== {fam.upper()} (susp={n1}, inno={n0}) ===')
#         lines.append(f"{'Statistic':<19} | {'Susp p50(p10,p90)':<21} | {'Inno p50(p10,p90)':<21} | AUC")
#         lines.append('-' * 72)
# 
#         for col in stat_cols:
#             v1 = s1[col].dropna().values
#             v0 = s0[col].dropna().values
#             if len(v1) > 0 and len(v0) > 0:
#                 p50_1, p10_1, p90_1 = np.percentile(v1, [50, 10, 90])
#                 p50_0, p10_0, p90_0 = np.percentile(v0, [50, 10, 90])
#                 all_v = np.concatenate([v1, v0])
#                 ranks = pd.Series(all_v).rank(method='average').values
#                 auc = (np.sum(ranks[: len(v1)]) - len(v1) * (len(v1) + 1) / 2.0) / (
#                     len(v1) * len(v0)
#                 )
#             else:
#                 p50_1 = p10_1 = p90_1 = p50_0 = p10_0 = p90_0 = 0.0
#                 auc = 0.5
# 
#             s1_str = f'{p50_1:.2f}({p10_1:.2f},{p90_1:.2f})'
#             s0_str = f'{p50_0:.2f}({p10_0:.2f},{p90_0:.2f})'
#             lines.append(f'{col:<19} | {s1_str:<21} | {s0_str:<21} | {auc:.3f}')
#         lines.append('')
# 
#     return '\n'.join(lines)[:5950]
# 
# 
# I expect 07_ep_pass_med / 04_wk_pass_ratio (per-episode and full-week USD pass-through conservation from senders through the collector/relay to final receivers, alongside 14_hold_delay_med_h and 25_conv_out_m_in) to win because the generator consistently enforces higher intermediary pass-through (~95–97% vs ~88–90%) and distinct stage-level conversion/holding rules on suspicious money flows even when scenario size, duration, and wallet counts are matched.