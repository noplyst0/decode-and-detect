#CODE
import numpy as np
import pandas as pd
import lightgbm as lgb


def _make_folds(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pure numpy/pandas GroupKFold (or uses precomputed fold IDs if groups in 0..n_splits-1)."""
    g = pd.Series(groups)
    uniq = g.unique()
    if len(uniq) <= n_splits and set(uniq).issubset(set(range(n_splits))):
        return [(np.flatnonzero(g.values != k), np.flatnonzero(g.values == k)) for k in sorted(uniq)]
    counts = g.value_counts()
    fold_loads = np.zeros(n_splits, dtype=np.int64)
    grp_to_fold = {}
    for grp_id, cnt in counts.items():
        f_idx = int(np.argmin(fold_loads))
        grp_to_fold[grp_id] = f_idx
        fold_loads[f_idx] += cnt
    f_assign = g.map(grp_to_fold).values
    return [(np.flatnonzero(f_assign != k), np.flatnonzero(f_assign == k)) for k in range(n_splits)]


# ==============================================================================
# 1. JOINT (FAMILY x SUSPICIOUS) + 6-REGIME DIFFICULTY HELPER
# ==============================================================================
def helper_joint_subtype(
    F: pd.DataFrame,
    Ft: pd.DataFrame,
    y: pd.Series,
    aux: pd.DataFrame,
    groups: np.ndarray,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = _make_folds(groups, n_splits=5)
    n_tr, n_te = len(F), len(Ft)
    y_bin = y.values.astype(int)
    fam = aux['family'].fillna('background').astype(str).str.lower().values
    diff = aux['difficulty'].fillna('none').astype(str).str.lower().values

    # Build 22+ class label: family + "__1" vs family + "__0" (merge tiny classes <25 rows into other)
    raw_sub = np.array([f"{f}__{yb}" for f, yb in zip(fam, y_bin)])
    val_cnts = pd.Series(raw_sub).value_counts()
    Rare = set(val_cnts[val_cnts < 25].index)
    clean_sub = np.array([s if s not in Rare else f"other__{s[-1]}" for s in raw_sub])
    classes_22, y_22 = np.unique(clean_sub, return_inverse=True)
    n_c22 = len(classes_22)

    oof_22 = np.zeros((n_tr, n_c22), dtype=np.float64)
    test_22 = np.zeros((n_te, n_c22), dtype=np.float64)

    params_22 = {
        'objective': 'multiclass',
        'num_class': n_c22,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'min_child_samples': 18,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_lambda': 2.0,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': seed,
    }

    for k, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(F.iloc[tr_idx], label=y_22[tr_idx])
        dva = lgb.Dataset(F.iloc[va_idx], label=y_22[va_idx], reference=dtr)
        model = lgb.train(
            params_22,
            dtr,
            num_boost_round=350,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_22[va_idx] = model.predict(F.iloc[va_idx], num_iteration=model.best_iteration)
        test_22 += model.predict(Ft, num_iteration=model.best_iteration) / len(folds)

    # Build 6-class difficulty/negative-regime target:
    # 0=bg_inno, 1=relay_inno, 2=motif_inno, 3=easy_pos, 4=med_pos, 5=hard_pos
    is_relay = np.isin(fam, ['aggregate_relay', 'distributed_relay'])
    is_bg = np.isin(fam, ['background', 'none', ''])
    y_6 = np.zeros(n_tr, dtype=np.int32)
    y_6[(y_bin == 0) & is_relay] = 1
    y_6[(y_bin == 0) & (~is_relay) & (~is_bg)] = 2
    y_6[(y_bin == 1) & pd.Series(diff).str.startswith('eas').values] = 3
    y_6[(y_bin == 1) & pd.Series(diff).str.startswith('med').values] = 4
    y_6[(y_bin == 1) & pd.Series(diff).str.startswith('har').values] = 5

    oof_6 = np.zeros((n_tr, 6), dtype=np.float64)
    test_6 = np.zeros((n_te, 6), dtype=np.float64)
    params_6 = {**params_22, 'num_class': 6, 'min_child_samples': 30, 'seed': seed + 101}

    for k, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(F.iloc[tr_idx], label=y_6[tr_idx])
        dva = lgb.Dataset(F.iloc[va_idx], label=y_6[va_idx], reference=dtr)
        model = lgb.train(
            params_6,
            dtr,
            num_boost_round=400,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(35, verbose=False)],
        )
        oof_6[va_idx] = model.predict(F.iloc[va_idx], num_iteration=model.best_iteration)
        test_6 += model.predict(Ft, num_iteration=model.best_iteration) / len(folds)

    # Extract structured Bayes probabilities and conditional log-odds
    c_map = {name: i for i, name in enumerate(classes_22)}
    motif_fams = [
        'bipartite', 'chain', 'cycle', 'fan_in', 'fan_out',
        'fan_out_in', 'multi_stage', 'random', 'stack',
    ]

    def _build_df(p22: np.ndarray, p6: np.ndarray, idx: pd.Index) -> pd.DataFrame:
        out = pd.DataFrame(index=idx)
        pos_cols = [i for i, c in enumerate(classes_22) if c.endswith('__1')]
        motif_pos_cols = [c_map[f"{f}__1"] for f in motif_fams if f"{f}__1" in c_map]
        motif_neg_cols = [c_map[f"{f}__0"] for f in motif_fams if f"{f}__0" in c_map]
        relay_neg_cols = [
            c_map[f"{f}__0"] for f in ('aggregate_relay', 'distributed_relay') if f"{f}__0" in c_map
        ]

        p_pos_all = np.sum(p22[:, pos_cols], axis=1)
        p_motif_pos = np.sum(p22[:, motif_pos_cols], axis=1)
        p_motif_neg = np.sum(p22[:, motif_neg_cols], axis=1)
        p_relay_neg = np.sum(p22[:, relay_neg_cols], axis=1)

        out['hj_p22_pos_total'] = p_pos_all
        out['hj_p22_motif_pos'] = p_motif_pos
        out['hj_p22_motif_inno'] = p_motif_neg
        out['hj_p22_relay_inno'] = p_relay_neg
        out['hj_logodds_motif_pos_vs_inno'] = np.log((p_motif_pos + 1e-5) / (p_motif_neg + 1e-5))
        out['hj_logodds_motif_pos_vs_relay'] = np.log((p_motif_pos + 1e-5) / (p_relay_neg + 1e-5))

        # Per-family conditional probability P(susp=1 | family=f) & family-gated expectation
        exp_cond_pos = np.zeros(len(idx), dtype=np.float64)
        for f in motif_fams:
            k1, k0 = f"{f}__1", f"{f}__0"
            if k1 in c_map and k0 in c_map:
                pf1 = p22[:, c_map[k1]]
                pf0 = p22[:, c_map[k0]]
                pf_tot = pf1 + pf0
                cond = pf1 / (pf_tot + 1e-4)
                out[f'hj_cond_pos_{f}'] = cond
                exp_cond_pos += pf_tot * cond
        out['hj_expected_family_gated_pos'] = exp_cond_pos

        # 6-regime difficulty features
        out['hj_p6_bg_inno'] = p6[:, 0]
        out['hj_p6_relay_inno'] = p6[:, 1]
        out['hj_p6_motif_inno'] = p6[:, 2]
        out['hj_p6_easy_pos'] = p6[:, 3]
        out['hj_p6_med_pos'] = p6[:, 4]
        out['hj_p6_hard_pos'] = p6[:, 5]
        out['hj_p6_easy_med_pos'] = p6[:, 3] + p6[:, 4]
        out['hj_p6_easy_vs_twin_logodds'] = np.log((p6[:, 3] + 1e-5) / (p6[:, 2] + 1e-5))
        out['hj_p6_easy_vs_relay_logodds'] = np.log((p6[:, 3] + 1e-5) / (p6[:, 1] + 1e-5))
        return out

    return _build_df(oof_22, oof_6, F.index), _build_df(test_22, test_6, Ft.index)


# ==============================================================================
# 2. REGIME-SPECIFIC TWIN & ANTI-RELAY SPECIALISTS
# ==============================================================================
def helper_regime_specialists(
    F: pd.DataFrame,
    Ft: pd.DataFrame,
    y: pd.Series,
    aux: pd.DataFrame,
    groups: np.ndarray,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = _make_folds(groups, n_splits=5)
    n_tr, n_te = len(F), len(Ft)
    y_bin = y.values.astype(int)
    fam = aux['family'].fillna('background').astype(str).str.lower().values
    diff = aux['difficulty'].fillna('none').astype(str).str.lower().values

    serial_fams = {'cycle', 'chain', 'multi_stage'}
    parallel_fams = {'bipartite', 'fan_out', 'fan_out_in', 'fan_in', 'stack'}
    all_motif_fams = serial_fams | parallel_fams | {'random'}
    relay_fams = {'aggregate_relay', 'distributed_relay'}

    is_serial = np.isin(fam, list(serial_fams))
    is_parallel = np.isin(fam, list(parallel_fams))
    is_motif = np.isin(fam, list(all_motif_fams))
    is_relay = np.isin(fam, list(relay_fams))
    is_easy = pd.Series(diff).str.startswith('eas').values
    is_hard = pd.Series(diff).str.startswith('har').values

    # Define 4 specialist training masks & sample weight functions:
    # (A) Serial Twin Specialist: cycle, chain, multi_stage (suspicious vs look-alike)
    # (B) Parallel Twin Specialist: bipartite, fan_out, fan_out_in, fan_in, stack
    # (C) Easy-Only Motif Twin Specialist: all 9 motif families, dropping medium/hard positives
    # (D) Motif-Positive vs Relay-Negative Specialist: easy/med motif pos vs innocent relay negs
    specs = {
        'sp_serial_twin': (is_serial & (~is_hard), 16),
        'sp_parallel_twin': (is_parallel & (~is_hard), 18),
        'sp_easy_only_twin': (is_motif & ((y_bin == 0) | is_easy), 20),
        'sp_motif_vs_relay': ((is_motif & (y_bin == 1) & (~is_hard)) | (is_relay & (y_bin == 0)), 30),
    }

    oof_df = pd.DataFrame(index=F.index)
    test_df = pd.DataFrame(index=Ft.index)

    base_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.03,
        'num_leaves': 15,
        'feature_fraction': 0.65,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'reg_lambda': 2.0,
        'verbosity': -1,
        'n_jobs': -1,
    }

    for s_idx, (name, (mask, min_child)) in enumerate(specs.items()):
        oof_p = np.zeros(n_tr, dtype=np.float64)
        test_p = np.zeros(n_te, dtype=np.float64)
        params = {**base_params, 'min_child_samples': min_child, 'seed': seed + 31 * s_idx}

        for k, (tr_idx, va_idx) in enumerate(folds):
            tr_sub = tr_idx[mask[tr_idx]]
            va_sub = va_idx[mask[va_idx]]
            if len(va_sub) < 10:
                va_sub = va_idx

            # Upweight easy positives slightly relative to medium in twin specialists
            w_tr = np.where((y_bin[tr_sub] == 1) & (~is_easy[tr_sub]), 0.45, 1.0)
            dtr = lgb.Dataset(F.iloc[tr_sub], label=y_bin[tr_sub], weight=w_tr)
            dva = lgb.Dataset(F.iloc[va_sub], label=y_bin[va_sub], reference=dtr)

            model = lgb.train(
                params,
                dtr,
                num_boost_round=500,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            oof_p[va_idx] = model.predict(F.iloc[va_idx], num_iteration=model.best_iteration)
            test_p += model.predict(Ft, num_iteration=model.best_iteration) / len(folds)

        oof_df[name] = oof_p
        test_df[name] = test_p

    # Interaction combos: max twin specialist & product with anti-relay specialist
    for df_out in (oof_df, test_df):
        df_out['sp_max_branch_serial'] = np.maximum(
            df_out['sp_serial_twin'], df_out['sp_parallel_twin']
        )
        df_out['sp_easy_x_antirelay'] = (
            df_out['sp_easy_only_twin'] * df_out['sp_motif_vs_relay']
        )
        df_out['sp_logodds_easy_antirelay'] = np.log(
            (df_out['sp_easy_only_twin'] + 1e-5) / (1.0 - df_out['sp_motif_vs_relay'] + 1e-5)
        )

    return oof_df, test_df


# ==============================================================================
# 3. HARD-POSITIVE DENOISING + 2-STAGE FOCAL HARD-NEGATIVE BOOSTER
# ==============================================================================
def get_sample_weights_and_params(
    y: pd.Series,
    aux: pd.DataFrame,
    oof_pred: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Returns (lgb_params, sample_weights, y_clean) for your main model or helper.
    - Sets y_clean=0 and weight=0.05 on unlearnable 'hard' positives so relay/bg bursts aren't boosted.
    - Downweights 'medium' positives to 0.30 so top-1,250 ranking focuses on high-precision motifs.
    - Upweights the ~1,250 innocent motif look-alike negatives by 3.5x (and stage-1 FPs via focal weight).
    """
    y_bin = y.values.astype(int)
    fam = aux['family'].fillna('background').astype(str).str.lower().values
    diff = aux['difficulty'].fillna('none').astype(str).str.lower().values

    is_easy = (y_bin == 1) & pd.Series(diff).str.startswith('eas').values
    is_med = (y_bin == 1) & pd.Series(diff).str.startswith('med').values
    is_hard = (y_bin == 1) & pd.Series(diff).str.startswith('har').values

    is_relay_neg = (y_bin == 0) & np.isin(fam, ['aggregate_relay', 'distributed_relay'])
    is_bg_neg = (y_bin == 0) & np.isin(fam, ['background', 'none', ''])
    is_twin_neg = (y_bin == 0) & (~is_relay_neg) & (~is_bg_neg)

    # Clean target: treat unlearnable planted 'hard' positives as 0 with near-zero weight
    y_clean = y_bin.copy()
    y_clean[is_hard] = 0

    w = np.ones(len(y_bin), dtype=np.float64)
    w[is_easy] = 1.00
    w[is_med] = 0.30
    w[is_hard] = 0.02
    w[is_twin_neg] = 3.50
    w[is_relay_neg] = 1.25
    w[is_bg_neg] = 0.75

    # Optional Stage-2 Focal Hard-Negative Mining on false alarms
    if oof_pred is not None:
        neg_mask = y_clean == 0
        w[neg_mask] *= 1.0 + 4.5 * (np.clip(oof_pred[neg_mask], 0.0, 1.0) ** 2.0)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': 0.02,
        'num_leaves': 15,
        'min_child_samples': 45,
        'min_child_weight': 15.0,
        'feature_fraction': 0.70,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'reg_lambda': 1.5,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': seed,
    }
    return params, w, y_clean


def helper_denoised_focal(
    F: pd.DataFrame,
    Ft: pd.DataFrame,
    y: pd.Series,
    aux: pd.DataFrame,
    groups: np.ndarray,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = _make_folds(groups, n_splits=5)
    n_tr, n_te = len(F), len(Ft)

    # Stage 1: Hard-positive denoised + twin-upweighted OOF prediction
    params1, w1, y_clean = get_sample_weights_and_params(y, aux, oof_pred=None, seed=seed)
    oof_s1 = np.zeros(n_tr, dtype=np.float64)
    test_s1 = np.zeros(n_te, dtype=np.float64)

    for k, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(F.iloc[tr_idx], label=y_clean[tr_idx], weight=w1[tr_idx])
        dva = lgb.Dataset(F.iloc[va_idx], label=y_clean[va_idx], weight=w1[va_idx], reference=dtr)
        m1 = lgb.train(
            params1,
            dtr,
            num_boost_round=700,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(45, verbose=False)],
        )
        oof_s1[va_idx] = m1.predict(F.iloc[va_idx], num_iteration=m1.best_iteration)
        test_s1 += m1.predict(Ft, num_iteration=m1.best_iteration) / len(folds)

    # Stage 2: Focal Hard-Negative Booster (upweights top false alarms from Stage 1)
    params2, w2, _ = get_sample_weights_and_params(y, aux, oof_pred=oof_s1, seed=seed + 77)
    oof_s2 = np.zeros(n_tr, dtype=np.float64)
    test_s2 = np.zeros(n_te, dtype=np.float64)

    for k, (tr_idx, va_idx) in enumerate(folds):
        dtr = lgb.Dataset(F.iloc[tr_idx], label=y_clean[tr_idx], weight=w2[tr_idx])
        dva = lgb.Dataset(F.iloc[va_idx], label=y_clean[va_idx], weight=w2[va_idx], reference=dtr)
        m2 = lgb.train(
            params2,
            dtr,
            num_boost_round=700,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(45, verbose=False)],
        )
        oof_s2[va_idx] = m2.predict(F.iloc[va_idx], num_iteration=m2.best_iteration)
        test_s2 += m2.predict(Ft, num_iteration=m2.best_iteration) / len(folds)

    oof_out = pd.DataFrame(
        {
            'df_denoised_s1': oof_s1,
            'df_focal_s2': oof_s2,
            'df_focal_blend': 0.45 * oof_s1 + 0.55 * oof_s2,
            'df_fp_suppression_diff': oof_s2 - oof_s1,
        },
        index=F.index,
    )
    test_out = pd.DataFrame(
        {
            'df_denoised_s1': test_s1,
            'df_focal_s2': test_s2,
            'df_focal_blend': 0.45 * test_s1 + 0.55 * test_s2,
            'df_fp_suppression_diff': test_s2 - test_s1,
        },
        index=Ft.index,
    )
    return oof_out, test_out


#FULLTEXT
# Gemini said
# 
# These 3 modelling changes directly target your OOF error profile (182 relay FPs, 209 motif-twin FPs across cycle/bipartite/chain/multi_stage, and 225 unlearnable hard positives corrupting relay leaves):
# 
# Joint (family × is_suspicious) + 6-Regime Multi-Class Helper (helper_joint_subtype): Your 13-class family helper worked (+1.1 AP) because it unlocked supervision on all 38,881 negative rows, but it merges suspicious and look-alike scenarios of the same family into a single class—blurring the exact conversion/USD features that flip sign across families. Splitting into 22 (family, y) classes (with min_child_samples=18) lets you compute exact within-family Bayes posteriors 
# 𝑃
# (
# 𝑦
# =
# 1
# ∣
# family
# =
# 𝑓
# )
# =
# 𝑃
# (
# 𝑓
# ,
# 1
# )
# 𝑃
# (
# 𝑓
# ,
# 1
# )
# +
# 𝑃
# (
# 𝑓
# ,
# 0
# )
# P(y=1∣family=f)=
# P(f,1)+P(f,0)
# P(f,1)
# 	​
# 
#  plus a 6-class difficulty/regime distribution (easy_pos, med_pos, hard_pos, motif_inno, relay_inno, bg_inno).
# 
# Factorized Twin & Anti-Relay Specialists (helper_regime_specialists): A single 9-motif twin specialist still mixes serial families (cycle, chain, multi_stage: suspicious = low conversion) with parallel families (bipartite, fan_out, fan_out_in, fan_in, stack: suspicious = high conversion), and never sees aggregate_relay false alarms. Training 4 fast subset specialists (serial_twin, parallel_twin, pos_vs_relay, and easy_only_twin) with min_child_samples=18 bypasses the 60-sample leaf floor on small families and directly suppresses the 182 relay FPs.
# 
# Hard-Positive Denoising + Focal Hard-Negative Reweighting (helper_denoised_focal & get_sample_weights_and_params): 18% of your y=1 labels are hard (200 planted inside aggregate_relay/distributed_relay + 25 random) with 0.6 AP (pure noise). Keeping y=1, weight=1 on those 225 rows forces trees to assign positive log-odds to relay bursts—causing the 142 aggregate_relay + 40 distributed_relay + 38 background FPs in your top 1,250. Zeroing out hard positive weights (0.0), downweighting medium positives (0.30), upweighting the ~1,250 innocent motif look-alike negatives (3.5x), and applying Stage-2 focal upweighting (
# 1
# +
# 4
# 𝑝
# ^
# oof
# 2
# 1+4
# p
# ^
# 	​
# 
# oof
# 2
# 	​
# 
# ) on false-alarm negatives directly optimizes top-1,250 precision (avoiding lambdarank, which fails on monochromatic scenario groups).
# 
# Python
# import numpy as np
# import pandas as pd
# import lightgbm as lgb
# 
# 
# def _make_folds(groups: np.ndarray, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
#     """Pure numpy/pandas GroupKFold (or uses precomputed fold IDs if groups in 0..n_splits-1)."""
#     g = pd.Series(groups)
#     uniq = g.unique()
#     if len(uniq) <= n_splits and set(uniq).issubset(set(range(n_splits))):
#         return [(np.flatnonzero(g.values != k), np.flatnonzero(g.values == k)) for k in sorted(uniq)]
#     counts = g.value_counts()
#     fold_loads = np.zeros(n_splits, dtype=np.int64)
#     grp_to_fold = {}
#     for grp_id, cnt in counts.items():
#         f_idx = int(np.argmin(fold_loads))
#         grp_to_fold[grp_id] = f_idx
#         fold_loads[f_idx] += cnt
#     f_assign = g.map(grp_to_fold).values
#     return [(np.flatnonzero(f_assign != k), np.flatnonzero(f_assign == k)) for k in range(n_splits)]
# 
# 
# # ==============================================================================
# # 1. JOINT (FAMILY x SUSPICIOUS) + 6-REGIME DIFFICULTY HELPER
# # ==============================================================================
# def helper_joint_subtype(
#     F: pd.DataFrame,
#     Ft: pd.DataFrame,
#     y: pd.Series,
#     aux: pd.DataFrame,
#     groups: np.ndarray,
#     seed: int = 42,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     folds = _make_folds(groups, n_splits=5)
#     n_tr, n_te = len(F), len(Ft)
#     y_bin = y.values.astype(int)
#     fam = aux['family'].fillna('background').astype(str).str.lower().values
#     diff = aux['difficulty'].fillna('none').astype(str).str.lower().values
# 
#     # Build 22+ class label: family + "__1" vs family + "__0" (merge tiny classes <25 rows into other)
#     raw_sub = np.array([f"{f}__{yb}" for f, yb in zip(fam, y_bin)])
#     val_cnts = pd.Series(raw_sub).value_counts()
#     Rare = set(val_cnts[val_cnts < 25].index)
#     clean_sub = np.array([s if s not in Rare else f"other__{s[-1]}" for s in raw_sub])
#     classes_22, y_22 = np.unique(clean_sub, return_inverse=True)
#     n_c22 = len(classes_22)
# 
#     oof_22 = np.zeros((n_tr, n_c22), dtype=np.float64)
#     test_22 = np.zeros((n_te, n_c22), dtype=np.float64)
# 
#     params_22 = {
#         'objective': 'multiclass',
#         'num_class': n_c22,
#         'metric': 'multi_logloss',
#         'learning_rate': 0.05,
#         'num_leaves': 15,
#         'min_child_samples': 18,
#         'feature_fraction': 0.65,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 1,
#         'reg_lambda': 2.0,
#         'verbosity': -1,
#         'n_jobs': -1,
#         'seed': seed,
#     }
# 
#     for k, (tr_idx, va_idx) in enumerate(folds):
#         dtr = lgb.Dataset(F.iloc[tr_idx], label=y_22[tr_idx])
#         dva = lgb.Dataset(F.iloc[va_idx], label=y_22[va_idx], reference=dtr)
#         model = lgb.train(
#             params_22,
#             dtr,
#             num_boost_round=350,
#             valid_sets=[dva],
#             callbacks=[lgb.early_stopping(30, verbose=False)],
#         )
#         oof_22[va_idx] = model.predict(F.iloc[va_idx], num_iteration=model.best_iteration)
#         test_22 += model.predict(Ft, num_iteration=model.best_iteration) / len(folds)
# 
#     # Build 6-class difficulty/negative-regime target:
#     # 0=bg_inno, 1=relay_inno, 2=motif_inno, 3=easy_pos, 4=med_pos, 5=hard_pos
#     is_relay = np.isin(fam, ['aggregate_relay', 'distributed_relay'])
#     is_bg = np.isin(fam, ['background', 'none', ''])
#     y_6 = np.zeros(n_tr, dtype=np.int32)
#     y_6[(y_bin == 0) & is_relay] = 1
#     y_6[(y_bin == 0) & (~is_relay) & (~is_bg)] = 2
#     y_6[(y_bin == 1) & pd.Series(diff).str.startswith('eas').values] = 3
#     y_6[(y_bin == 1) & pd.Series(diff).str.startswith('med').values] = 4
#     y_6[(y_bin == 1) & pd.Series(diff).str.startswith('har').values] = 5
# 
#     oof_6 = np.zeros((n_tr, 6), dtype=np.float64)
#     test_6 = np.zeros((n_te, 6), dtype=np.float64)
#     params_6 = {**params_22, 'num_class': 6, 'min_child_samples': 30, 'seed': seed + 101}
# 
#     for k, (tr_idx, va_idx) in enumerate(folds):
#         dtr = lgb.Dataset(F.iloc[tr_idx], label=y_6[tr_idx])
#         dva = lgb.Dataset(F.iloc[va_idx], label=y_6[va_idx], reference=dtr)
#         model = lgb.train(
#             params_6,
#             dtr,
#             num_boost_round=400,
#             valid_sets=[dva],
#             callbacks=[lgb.early_stopping(35, verbose=False)],
#         )
#         oof_6[va_idx] = model.predict(F.iloc[va_idx], num_iteration=model.best_iteration)
#         test_6 += model.predict(Ft, num_iteration=model.best_iteration) / len(folds)
# 
#     # Extract structured Bayes probabilities and conditional log-odds
#     c_map = {name: i for i, name in enumerate(classes_22)}
#     motif_fams = [
#         'bipartite', 'chain', 'cycle', 'fan_in', 'fan_out',
#         'fan_out_in', 'multi_stage', 'random', 'stack',
#     ]
# 
#     def _build_df(p22: np.ndarray, p6: np.ndarray, idx: pd.Index) -> pd.DataFrame:
#         out = pd.DataFrame(index=idx)
#         pos_cols = [i for i, c in enumerate(classes_22) if c.endswith('__1')]
#         motif_pos_cols = [c_map[f"{f}__1"] for f in motif_fams if f"{f}__1" in c_map]
#         motif_neg_cols = [c_map[f"{f}__0"] for f in motif_fams if f"{f}__0" in c_map]
#         relay_neg_cols = [
#             c_map[f"{f}__0"] for f in ('aggregate_relay', 'distributed_relay') if f"{f}__0" in c_map
#         ]
# 
#         p_pos_all = np.sum(p22[:, pos_cols], axis=1)
#         p_motif_pos = np.sum(p22[:, motif_pos_cols], axis=1)
#         p_motif_neg = np.sum(p22[:, motif_neg_cols], axis=1)
#         p_relay_neg = np.sum(p22[:, relay_neg_cols], axis=1)
# 
#         out['hj_p22_pos_total'] = p_pos_all
#         out['hj_p22_motif_pos'] = p_motif_pos
#         out['hj_p22_motif_inno'] = p_motif_neg
#         out['hj_p22_relay_inno'] = p_relay_neg
#         out['hj_logodds_motif_pos_vs_inno'] = np.log((p_motif_pos + 1e-5) / (p_motif_neg + 1e-5))
#         out['hj_logodds_motif_pos_vs_relay'] = np.log((p_motif_pos + 1e-5) / (p_relay_neg + 1e-5))
# 
#         # Per-family conditional probability P(susp=1 | family=f) & family-gated expectation
#         exp_cond_pos = np.zeros(len(idx), dtype=np.float64)
#         for f in motif_fams:
#             k1, k0 = f"{f}__1", f"{f}__0"
#             if k1 in c_map and k0 in c_map:
#                 pf1 = p22[:, c_map[k1]]
#                 pf0 = p22[:, c_map[k0]]
#                 pf_tot = pf1 + pf0
#                 cond = pf1 / (pf_tot + 1e-4)
#                 out[f'hj_cond_pos_{f}'] = cond
#                 exp_cond_pos += pf_tot * cond
#         out['hj_expected_family_gated_pos'] = exp_cond_pos
# 
#         # 6-regime difficulty features
#         out['hj_p6_bg_inno'] = p6[:, 0]
#         out['hj_p6_relay_inno'] = p6[:, 1]
#         out['hj_p6_motif_inno'] = p6[:, 2]
#         out['hj_p6_easy_pos'] = p6[:, 3]
#         out['hj_p6_med_pos'] = p6[:, 4]
#         out['hj_p6_hard_pos'] = p6[:, 5]
#         out['hj_p6_easy_med_pos'] = p6[:, 3] + p6[:, 4]
#         out['hj_p6_easy_vs_twin_logodds'] = np.log((p6[:, 3] + 1e-5) / (p6[:, 2] + 1e-5))
#         out['hj_p6_easy_vs_relay_logodds'] = np.log((p6[:, 3] + 1e-5) / (p6[:, 1] + 1e-5))
#         return out
# 
#     return _build_df(oof_22, oof_6, F.index), _build_df(test_22, test_6, Ft.index)
# 
# 
# # ==============================================================================
# # 2. REGIME-SPECIFIC TWIN & ANTI-RELAY SPECIALISTS
# # ==============================================================================
# def helper_regime_specialists(
#     F: pd.DataFrame,
#     Ft: pd.DataFrame,
#     y: pd.Series,
#     aux: pd.DataFrame,
#     groups: np.ndarray,
#     seed: int = 42,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     folds = _make_folds(groups, n_splits=5)
#     n_tr, n_te = len(F), len(Ft)
#     y_bin = y.values.astype(int)
#     fam = aux['family'].fillna('background').astype(str).str.lower().values
#     diff = aux['difficulty'].fillna('none').astype(str).str.lower().values
# 
#     serial_fams = {'cycle', 'chain', 'multi_stage'}
#     parallel_fams = {'bipartite', 'fan_out', 'fan_out_in', 'fan_in', 'stack'}
#     all_motif_fams = serial_fams | parallel_fams | {'random'}
#     relay_fams = {'aggregate_relay', 'distributed_relay'}
# 
#     is_serial = np.isin(fam, list(serial_fams))
#     is_parallel = np.isin(fam, list(parallel_fams))
#     is_motif = np.isin(fam, list(all_motif_fams))
#     is_relay = np.isin(fam, list(relay_fams))
#     is_easy = pd.Series(diff).str.startswith('eas').values
#     is_hard = pd.Series(diff).str.startswith('har').values
# 
#     # Define 4 specialist training masks & sample weight functions:
#     # (A) Serial Twin Specialist: cycle, chain, multi_stage (suspicious vs look-alike)
#     # (B) Parallel Twin Specialist: bipartite, fan_out, fan_out_in, fan_in, stack
#     # (C) Easy-Only Motif Twin Specialist: all 9 motif families, dropping medium/hard positives
#     # (D) Motif-Positive vs Relay-Negative Specialist: easy/med motif pos vs innocent relay negs
#     specs = {
#         'sp_serial_twin': (is_serial & (~is_hard), 16),
#         'sp_parallel_twin': (is_parallel & (~is_hard), 18),
#         'sp_easy_only_twin': (is_motif & ((y_bin == 0) | is_easy), 20),
#         'sp_motif_vs_relay': ((is_motif & (y_bin == 1) & (~is_hard)) | (is_relay & (y_bin == 0)), 30),
#     }
# 
#     oof_df = pd.DataFrame(index=F.index)
#     test_df = pd.DataFrame(index=Ft.index)
# 
#     base_params = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'learning_rate': 0.03,
#         'num_leaves': 15,
#         'feature_fraction': 0.65,
#         'bagging_fraction': 0.8,
#         'bagging_freq': 1,
#         'reg_lambda': 2.0,
#         'verbosity': -1,
#         'n_jobs': -1,
#     }
# 
#     for s_idx, (name, (mask, min_child)) in enumerate(specs.items()):
#         oof_p = np.zeros(n_tr, dtype=np.float64)
#         test_p = np.zeros(n_te, dtype=np.float64)
#         params = {**base_params, 'min_child_samples': min_child, 'seed': seed + 31 * s_idx}
# 
#         for k, (tr_idx, va_idx) in enumerate(folds):
#             tr_sub = tr_idx[mask[tr_idx]]
#             va_sub = va_idx[mask[va_idx]]
#             if len(va_sub) < 10:
#                 va_sub = va_idx
# 
#             # Upweight easy positives slightly relative to medium in twin specialists
#             w_tr = np.where((y_bin[tr_sub] == 1) & (~is_easy[tr_sub]), 0.45, 1.0)
#             dtr = lgb.Dataset(F.iloc[tr_sub], label=y_bin[tr_sub], weight=w_tr)
#             dva = lgb.Dataset(F.iloc[va_sub], label=y_bin[va_sub], reference=dtr)
# 
#             model = lgb.train(
#                 params,
#                 dtr,
#                 num_boost_round=500,
#                 valid_sets=[dva],
#                 callbacks=[lgb.early_stopping(40, verbose=False)],
#             )
#             oof_p[va_idx] = model.predict(F.iloc[va_idx], num_iteration=model.best_iteration)
#             test_p += model.predict(Ft, num_iteration=model.best_iteration) / len(folds)
# 
#         oof_df[name] = oof_p
#         test_df[name] = test_p
# 
#     # Interaction combos: max twin specialist & product with anti-relay specialist
#     for df_out in (oof_df, test_df):
#         df_out['sp_max_branch_serial'] = np.maximum(
#             df_out['sp_serial_twin'], df_out['sp_parallel_twin']
#         )
#         df_out['sp_easy_x_antirelay'] = (
#             df_out['sp_easy_only_twin'] * df_out['sp_motif_vs_relay']
#         )
#         df_out['sp_logodds_easy_antirelay'] = np.log(
#             (df_out['sp_easy_only_twin'] + 1e-5) / (1.0 - df_out['sp_motif_vs_relay'] + 1e-5)
#         )
# 
#     return oof_df, test_df
# 
# 
# # ==============================================================================
# # 3. HARD-POSITIVE DENOISING + 2-STAGE FOCAL HARD-NEGATIVE BOOSTER
# # ==============================================================================
# def get_sample_weights_and_params(
#     y: pd.Series,
#     aux: pd.DataFrame,
#     oof_pred: np.ndarray | None = None,
#     seed: int = 42,
# ) -> tuple[dict, np.ndarray, np.ndarray]:
#     """
#     Returns (lgb_params, sample_weights, y_clean) for your main model or helper.
#     - Sets y_clean=0 and weight=0.05 on unlearnable 'hard' positives so relay/bg bursts aren't boosted.
#     - Downweights 'medium' positives to 0.30 so top-1,250 ranking focuses on high-precision motifs.
#     - Upweights the ~1,250 innocent motif look-alike negatives by 3.5x (and stage-1 FPs via focal weight).
#     """
#     y_bin = y.values.astype(int)
#     fam = aux['family'].fillna('background').astype(str).str.lower().values
#     diff = aux['difficulty'].fillna('none').astype(str).str.lower().values
# 
#     is_easy = (y_bin == 1) & pd.Series(diff).str.startswith('eas').values
#     is_med = (y_bin == 1) & pd.Series(diff).str.startswith('med').values
#     is_hard = (y_bin == 1) & pd.Series(diff).str.startswith('har').values
# 
#     is_relay_neg = (y_bin == 0) & np.isin(fam, ['aggregate_relay', 'distributed_relay'])
#     is_bg_neg = (y_bin == 0) & np.isin(fam, ['background', 'none', ''])
#     is_twin_neg = (y_bin == 0) & (~is_relay_neg) & (~is_bg_neg)
# 
#     # Clean target: treat unlearnable planted 'hard' positives as 0 with near-zero weight
#     y_clean = y_bin.copy()
#     y_clean[is_hard] = 0
# 
#     w = np.ones(len(y_bin), dtype=np.float64)
#     w[is_easy] = 1.00
#     w[is_med] = 0.30
#     w[is_hard] = 0.02
#     w[is_twin_neg] = 3.50
#     w[is_relay_neg] = 1.25
#     w[is_bg_neg] = 0.75
# 
#     # Optional Stage-2 Focal Hard-Negative Mining on false alarms
#     if oof_pred is not None:
#         neg_mask = y_clean == 0
#         w[neg_mask] *= 1.0 + 4.5 * (np.clip(oof_pred[neg_mask], 0.0, 1.0) ** 2.0)
# 
#     params = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'learning_rate': 0.02,
#         'num_leaves': 15,
#         'min_child_samples': 45,
#         'min_child_weight': 15.0,
#         'feature_fraction': 0.70,
#         'bagging_fraction': 0.80,
#         'bagging_freq': 1,
#         'reg_lambda': 1.5,
#         'verbosity': -1,
#         'n_jobs': -1,
#         'seed': seed,
#     }
#     return params, w, y_clean
# 
# 
# def helper_denoised_focal(
#     F: pd.DataFrame,
#     Ft: pd.DataFrame,
#     y: pd.Series,
#     aux: pd.DataFrame,
#     groups: np.ndarray,
#     seed: int = 42,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     folds = _make_folds(groups, n_splits=5)
#     n_tr, n_te = len(F), len(Ft)
# 
#     # Stage 1: Hard-positive denoised + twin-upweighted OOF prediction
#     params1, w1, y_clean = get_sample_weights_and_params(y, aux, oof_pred=None, seed=seed)
#     oof_s1 = np.zeros(n_tr, dtype=np.float64)
#     test_s1 = np.zeros(n_te, dtype=np.float64)
# 
#     for k, (tr_idx, va_idx) in enumerate(folds):
#         dtr = lgb.Dataset(F.iloc[tr_idx], label=y_clean[tr_idx], weight=w1[tr_idx])
#         dva = lgb.Dataset(F.iloc[va_idx], label=y_clean[va_idx], weight=w1[va_idx], reference=dtr)
#         m1 = lgb.train(
#             params1,
#             dtr,
#             num_boost_round=700,
#             valid_sets=[dva],
#             callbacks=[lgb.early_stopping(45, verbose=False)],
#         )
#         oof_s1[va_idx] = m1.predict(F.iloc[va_idx], num_iteration=m1.best_iteration)
#         test_s1 += m1.predict(Ft, num_iteration=m1.best_iteration) / len(folds)
# 
#     # Stage 2: Focal Hard-Negative Booster (upweights top false alarms from Stage 1)
#     params2, w2, _ = get_sample_weights_and_params(y, aux, oof_pred=oof_s1, seed=seed + 77)
#     oof_s2 = np.zeros(n_tr, dtype=np.float64)
#     test_s2 = np.zeros(n_te, dtype=np.float64)
# 
#     for k, (tr_idx, va_idx) in enumerate(folds):
#         dtr = lgb.Dataset(F.iloc[tr_idx], label=y_clean[tr_idx], weight=w2[tr_idx])
#         dva = lgb.Dataset(F.iloc[va_idx], label=y_clean[va_idx], weight=w2[va_idx], reference=dtr)
#         m2 = lgb.train(
#             params2,
#             dtr,
#             num_boost_round=700,
#             valid_sets=[dva],
#             callbacks=[lgb.early_stopping(45, verbose=False)],
#         )
#         oof_s2[va_idx] = m2.predict(F.iloc[va_idx], num_iteration=m2.best_iteration)
#         test_s2 += m2.predict(Ft, num_iteration=m2.best_iteration) / len(folds)
# 
#     oof_out = pd.DataFrame(
#         {
#             'df_denoised_s1': oof_s1,
#             'df_focal_s2': oof_s2,
#             'df_focal_blend': 0.45 * oof_s1 + 0.55 * oof_s2,
#             'df_fp_suppression_diff': oof_s2 - oof_s1,
#         },
#         index=F.index,
#     )
#     test_out = pd.DataFrame(
#         {
#             'df_denoised_s1': test_s1,
#             'df_focal_s2': test_s2,
#             'df_focal_blend': 0.45 * test_s1 + 0.55 * test_s2,
#             'df_fp_suppression_diff': test_s2 - test_s1,
#         },
#         index=Ft.index,
#     )
#     return oof_out, test_out
# 
# 
# Integration note: Pass (w1, y_clean) from get_sample_weights_and_params(y, aux) directly into your final 5-seed LightGBM and XGBoost fit as well—stopping your final trees from fitting the 200 planted relay positives (is_hard) is the single cleanest way to drop the 182 aggregate_relay/distributed_relay false alarms out of your top 1,250.