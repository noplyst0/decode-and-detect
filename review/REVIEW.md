# Independent review: ~/xrpl-hack (24 Oct, 12:35)

**TL;DR for leaderboard testing:**
1. Upload a smoothed copy of your best CSV:
   `.venv/bin/python review/smooth_sub.py --test data/test --sub <best>.csv --out <best>_s.csv`
   For code, `review/burst_smooth.patch` applies cleanly to 0a9badb.
2. Leave `NO_CONV` off. It lost on the time-split check (-1.95 / +0.24), which is the one check that sees train/test
   feature shift. Its CV gain does not show that shift.
3. No further correctness bugs found beyond the USD one; see section 6 for what was checked.

Setup: every number comes from `--fast --stack 0 --drop memo,probe`. **base** = a snapshot of `src/pipeline.py` taken
at 11:53 (`review/pipeline_base.py`); **exp** = the same file plus switchable changes (`review/pipeline_exp.py`).
Both log to `review/runs/`, not `runs/log.csv`. There are two kinds of check:
- **CV**: the usual grouped 5-fold CV on train.
- **TS** (time split): train is cut at mid-week into half A and half B, and features are computed separately on each
  half, just as train and test are separate weeks. Train on A, score B, then the reverse (`review/ts_eval.sh`).
  This is the only check here that can see train/test *feature shift*, which grouped CV can't.

## Ranked findings

### 1. BUG, already fixed in 58fe1a3: every asset was priced as XRP, in micro-USD
`src/pipeline.py:121` `usd_rates`. `market_rates.csv` has `TIMESTAMP, ASSET, PRICE_MICRO_USD` and no
`TOKEN_ADDRESS`/`SYMBOL`. So `key=[]`, the merge ran on the hour alone, and `drop_duplicates(["H"])` kept the
first row of each hour, which is XRP. Before the fix, conversions read `USD_RECV/USD_SENT` = 0.49-2.05; after it,
0.988-0.9999. This affected value matching across conversions, amount echo, the round-number features, and the
`gm_*` amount features.

| check | base | fixed | diff |
|---|---|---|---|
| CV seed 1 / 2 | 59.15 / 58.84 | 58.77 / 59.78 | -0.38 / +0.94 |
| TS A→B / B→A | 55.17 / 56.02 | 58.53 / 55.50 | +3.36 / -0.52 |

Mean gain is about +0.85, noisy. Keep it: it is a correctness fix.

### 2. RECOMMENDED: burst smoothing of the final scores (expected LB about +0.4 to +1)
Patch: `review/burst_smooth.patch` (against HEAD 58fe1a3; on by default, `SMOOTH=0` turns it off).
To apply it to a CSV you already have, with no retraining (~20 s):
`.venv/bin/python review/smooth_sub.py --test data/test --sub sub_vN.csv --out sub_vNs.csv`

How it works: payments that touch the same wallet within 5 minutes are linked into one group. Each payment's final
score becomes `0.5·rank(score) + 0.5·rank(highest score in its group)`. Easy flows finish in minutes, so their weakest
payments are lifted by the strongest one. That is where the remaining easy misses sit: 10% of easy positives rank
below 1,100, and they are mostly single payments with `bu2_n=2`.

| check | effect of smoothing (300 s, a=0.5) |
|---|---|
| OOF, 3 dumps (fixed s1, base s2, fixed s2) | +1.26, +1.04, +1.07 |
| TS out-of-sample (base A→B, B→A; fixed A→B, B→A) | +0.58, +0.46, +0.37, -0.02 |
| **Re-based on current HEAD 0a9badb** (`WEIGHTS=hard0`), OOF seed 1 / 42 | 59.49→60.82 (+1.33), 60.16→61.07 (+0.91) |

The 450 s window scored better on all 7 checks (TS +0.89/+0.85/+0.55/+0.27). But the test week's largest
450 s group already has 127 payments (train: 83), and at 600 s groups merge into one giant group (train 899, test
610), after which OOF AP drops sharply. So 300 s is the safe choice: the largest group is 38 in train and 30 in test.
Taking the leave-one-out max instead of the max was worse (-1 to -5). Blending with the group mean was flat at 1-2 min and worse at 5 min.

### 3. Why medium = 0% in the top 1%: the model ranks medium BELOW its look-alikes (diagnosis only, no fix)
On out-of-fold scores, among motif-family payments that are not easy, AUC(medium vs look-alike) = **0.40**.
Scenario stats from the train labels (`review/scen_stats.py`):

| | median gap between payments | median duration |
|---|---|---|
| easy | 1.0 min | 5 min |
| look-alike | 9.5 min | 62 min |
| medium | 17.8 min | 110 min |

Medium flows are *slower* than look-alikes. The model learnt "faster = more suspicious" from the 800 easy
positives, which pushes medium down. I tried a second model trained without the easy positives
(`review/medium_model.py`): AUC was still 0.46, and blending it in gave +0.06. Medium has only **27 scenarios**, and
even oracle scenario-level gap separates it from look-alikes with AUC 0.70 only. This is not learnable in time.
It is worth a line in METHOD.md.

### 4. Upper limit on easy (sets expectations)
21 of 140 look-alike scenarios (15%) are as fast as easy ones (median gap under 2.5 min). Their gap, duration, size,
wallet count, value kept, USD amount and hour are all the same as easy scenarios. Easy AP (easy positives vs all
negatives) is 79 now; the estimated ceiling is about 85. The 181 negatives that outscore the bottom quarter of easy
positives are 157 look-alikes, 18 relay and 6 background. Big new gains on easy are therefore unlikely. Variance
reduction (more seeds) and smoothing are the remaining levers.

### 5. Train/test shift in the burst features: real, but both fixes I tried made things worse (don't apply)
- `burst_features` (`src/pipeline.py:693`): at 15 and 45 minutes, single linkage merges almost everything into one
  giant group (train 4,536 / 14,220 tx, test 3,782 / 14,019). That group's `bu15_*`/`bu45_*` statistics are one
  constant per week, and the constant differs between weeks: KS 0.37 between train and test, the largest of any feature.
  Masking groups over 200 payments to NaN: TS -2.39 / -0.26. **Do not apply.**
- Conversion share: 41% of train payments are conversions vs 50% of test (KS 0.32 on `s_conv_share`). The README
  says conversion choices ignore labels. Dropping every conversion feature: TS -1.95 / +0.24. **Do not apply.**

### 6. Leakage and consistency checks that came up clean
- Family helper (`family_aux`, :831): its probabilities are out-of-fold, using the same deterministic GroupKFold folds
  as the main model. The only effect is the usual second-order stacking one. CV (≈59) < LB (63.9), so CV is not
  optimistic.
- No feature's NaN rate differs between train and test by more than 2 points.
- Scenario groups (`scenario_groups`) are used only for folds.
- Times: `DATETIME` equals the numeric `TIMESTAMP` exactly (UTC) in both weeks, every transfer matches an hourly
  price (no median fallback), and transfer times equal transaction times.
- Ties: a single-seed `--fast` run on HEAD gives only ~25-27k distinct scores out of 40,101. The largest tied block
  (5-8k rows) sits at the bottom score; only 8-14 tied rows reach the top 1,000. Breaking the ties moves OOF AP by
  +0.03, so this is harmless. The 5-seed run plus the XGB rank blend in `run.sh` breaks most ties anyway.
- `gemini_feats.py:254-274` numbers issuers by `np.unique` of their addresses. The test issuers are different addresses
  from the train ones, but they sort in the same role order (the GBP/USD issuer before the EUR/USD issuer in both
  weeks), so `tx_iss_mask` happens to be consistent. Fine for this data, but fragile.

## Files
- `review/burst_smooth.patch`: the one recommended patch (`git apply review/burst_smooth.patch`, checked against HEAD).
- `review/smooth_sub.py`: smooths an existing submission CSV.
- `review/pipeline_exp.py`: the switches `USD_FIX`, `SMOOTH=300`, `BURST_CAP=200`, `NO_CONV=1`.
- `review/ts_eval.sh`: the time-split check (half-week data lives in the session scratchpad, not the repo).
- Logs are in `review/runs/*.log`; `ts_all.txt` has the time-split table.
- `review/` is untracked. `make_zip.sh` zips only tracked files, so none of this ships unless someone runs `git add`.
- OOF dumps, scenario table and prediction CSVs were moved to the session scratchpad (`.../scratchpad/review_outputs/`) to keep `review/` small (208 KB).
