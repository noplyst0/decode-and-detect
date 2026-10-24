You are helping a student team in a 6-hour hackathon (ends 16:30 London). I can only share aggregate findings, not the data. Please think hard, then answer in the exact format at the end.

TASK: synthetic XRPL-style payment ledger, one week, 40,131 labelled train tx (1,250 suspicious), 40,101 unlabelled test tx (a different week, disjoint wallets and hashes). Score every test tx 0-1; metric = average precision. Only Payment tx, all succeed, no memos, fees/flags/tags identical for everyone (no metadata leak). 700 wallets in a split, ~57 tx per wallet, and 535 of the 700 wallets touch some suspicious scenario, so wallet identity or wallet reputation carries no signal.

GENERATOR FACTS (from train labels):
- Suspicious = 64% easy, 18% medium, 18% hard (hard = 200 planted + 25 deliberately random).
- Families: bipartite, chain, cycle, fan_in, fan_out, fan_out_in, multi_stage, random, stack. EVERY family has innocent "look-alike" scenarios of the same shape, roughly 1:1 with the suspicious ones. Scenarios are 4-16 tx (median 7).
- Easy suspicious: whole scenario finishes in minutes (median gap between consecutive scenario tx ~1 min, duration ~5 min); intermediaries pass on ~97% of USD received.
- Innocent look-alikes: median gap ~10 min, duration ~1 h (10th pct 11 min), intermediaries keep ~89%.
- Medium suspicious: median gap ~18 min, duration ~1.8 h, intermediaries keep ~90%, gaps slightly MORE regular (cv of gaps 0.41 vs 0.52). Otherwise very close to look-alikes.
- Per-family medians, suspicious vs look-alike: cycle conversion share 0.00 vs 0.62 and USD median 241 vs 94; chain conv 0.12 vs 0.38; bipartite conv 0.75 vs 0.00; fan_out conv 0.50 vs 0.00 and USD 99 vs 42; fan_out_in conv 0.62 vs 0.00; fan_in USD 214 vs 95; stack scenario size 12 vs 4. Same-family scenarios otherwise match in size and wallet counts.
- Hard: ~100 suspicious tx hidden in each of two "relay" families of ~10,000 innocent tx (collector wallets gather over days, then pay out in bursts). We treat hard as unlearnable for now.
- Conversions: USD/GBP/EUR issued by two issuers (USD of issuer A != USD of issuer B); conversion spread ~0.1-0.5%.

OUR PIPELINE: per-tx edge table `e` (index TRANSACTION_HASH), ~170 features, LightGBM (lr 0.02, 15 leaves, min_child 60), 5-fold CV grouped by scenario. Features: amounts/USD, conversions, account history (degree, timing, pass-through), next/prev hop timing, value matching (money followed by amount 80-102% within 7 days), chains, cycles, scatter-gather, graph components, isolation forest, and a "burst" block: single-linkage groups of tx that share a wallet within 2/5/15/45 minutes, with group size, duration, rate, wallets per tx, USD cv, position, and (5/15 min) conversion share, distinct assets, median USD, distinct senders/receivers, max in/out degree, wallets that both send and receive. Stacking on neighbour scores HURTS here (wallet reuse).
Scores: CV AP 56.7, leaderboard 59.9 (v2). By difficulty in CV: easy 75, medium 6.6, hard 0.6. Run-to-run seed noise in CV is about +-2.5 AP.
Top false alarms are fast innocent look-alikes (bursts of 7-10 tx passing on ~98%), indistinguishable from easy positives by timing.

THE EDGE TABLE `e` (one row per tx, pandas, index = TRANSACTION_HASH):
SRC str (sender), DST str (receiver), TS int64 (unix seconds), USD_SENT float (USD value of first leg), USD_RECV float (USD of last leg), ASSET_SENT str (e.g. "XRP" or "rIssuer...:USD"), ASSET_RECV str, IS_CONV bool (ASSET_SENT != ASSET_RECV), N_LEGS int (1-3 transfer legs; 3 = bridge via XRP), V_SENT float (native units), HAS_SENDMAX int, TR int (seconds since split start). Ignore FEE, DEST_TAG, MEMO_TEXT, HAS_MEMO, SUCCESS, TRANSACTION_TYPE (constant).

WHAT I WANT: new per-transaction features aimed at (1) separating FAST innocent look-alikes from easy suspicious motifs of the same shape, and (2) lifting medium cases. Think about what a generator that equalises size/shape/timing between twins might still leave different: e.g. how amounts relate along the motif (splits summing to the input, decay per hop, fixed fees, round USD targets), the order/sequence of conversions along a path, whether the same issuer is kept or swapped, intra-motif amount similarity, time-of-day, whether wallets in the motif were active just before/after, "reconstructed motif" detection by amount-matching rather than wallet-sharing, etc. Features must be unsupervised (no labels), deterministic, and computable on the test week alone.

FORMAT OF YOUR ANSWER:
1. One Python code block defining `def gemini_features(e: pd.DataFrame) -> pd.DataFrame:` returning a DataFrame with the SAME index as e and 15-40 float columns, prefixed `gm_`. numpy/pandas/scipy only, vectorised or near-vectorised, must run in under 30 s on 40k rows. Comment each feature in one line.
2. Then at most 250 words: the 3 ideas you think are most likely to work and why, plus anything in our approach you think is wrong.
