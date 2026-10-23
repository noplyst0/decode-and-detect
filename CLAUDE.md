# Decode & Detect: XRPL AI Hack (Sat 24 Oct 2026, IDEALondon)

Faizan's team entry. This repo was pre-built the night before, against MOCK data in the real schema.
**Hard deadline: 16:30 London.** Upload at hackathon.ukcbt.org with the team's organiser access code.

## The task
Score every test transaction for laundering-style behaviour. Supervised: train labels given.
- Data (synthetic, AWS/SonarX XRPL schema), per split: `transactions.parquet`, `transfers.parquet`
  (1-3 money legs per tx), `blocks.parquet`, `market_rates.csv` (hourly USD per asset), `labels.csv`
  (train only). Train is 40,131 tx, test is 40,101 tx, each a separate 7-day window. Organisers ship
  `README.md`, `payments.py` (one edge per payment) and `baseline.py`. **Read README.md first.**
- Metric: **100 × Average Precision** over all 40,101 test tx. Only the ORDER matters. Ties are scored as
  one block, so keep scores continuous. Also shown (not ranked): precision/recall in the top 402, ROC-AUC.
- The leaderboard scores the full test set. It is the real score, so submit often. **50 uploads per team per
  UTC day**, and the best completed run shows.
- Suspicious mix: easy 64%, medium 18%, hard 18%. "Hard means history": multi-payment flows with changing
  counterparties. The test set has variations not seen in training, plus a few deliberately ambiguous cases.

## Submission (two files)
1. Predictions CSV: `TRANSACTION_HASH,score`, every test tx exactly once, score in [0,1], ≤12 MiB.
2. Code ZIP (≤8 MiB, `./make_zip.sh`): code, exact run command (`run.sh`), seeds, `requirements.txt`, team
   list, commit hash, results, and the method (<10 slides or 4 pages). No data, no venv, no secrets.
- Rules: organiser data only (public docs and papers are fine). AI allowed if disclosed in
  `AI_DISCLOSURE.md`. Must be reproducible. Open source (MIT `LICENSE` is already in place) and public by the
  deadline. Language: say "flagged for review", never "criminal".

## Repo state
- `src/pipeline.py`: load → `build_edges` (one row per tx, USD via hourly rates, first leg = paid, last
  leg = delivered) → features (tx, sender/receiver account, timing/forwarding, graph: pair, reciprocity,
  same-SCC cycles, scatter-gather, hub flags) → LightGBM 5-fold CV over day blocks (AP early stopping) →
  stage 2 stacking (account-level leave-one-out OOF scores) → 3-seed average → CSV. Each run appends to
  `runs/log.csv` (time, commit, seed, n_features, CV AP, file).
  Run it with: `.venv/bin/python src/pipeline.py --train <dir> --test <dir> --out sub_vN.csv`
- `src/ensemble.py`: rank-average CSVs (teammates' models). `src/mock_data.py`: a fake dataset for tests.
- Column names follow the real SonarX parquet files (checked 2026-10-23): transfers have TRANSACTION_HASH,
  TRANSFER_INDEX (-999 = direct), FROM_ADDRESS, TO_ADDRESS, TOKEN_ADDRESS, SYMBOL, SOURCE_VALUE (drops for
  XRP), VALUE (human units), DATETIME. Transactions have ACCOUNT, TRANSACTION_TYPE, FEE, TRANSACTION_RESULT,
  DELIVERED_AMOUNT (JSON text), EXTRA_TRANSACTION_FIELDS (JSON text: Destination, Amount, SendMax,
  DestinationTag), MEMOS. **The market_rates and labels column names are guessed** (`find_col` matches on
  substrings), so check them against the README.

## First 30 minutes tomorrow (in this order)
1. Put the data in `data/train`, `data/test`. Read README.md. Fix load-layer names if needed (top of pipeline.py).
2. Run `baseline.py` and submit it. That proves the upload works and gives the number to beat.
3. Run `pipeline.py` and submit it. Compare CV AP with the leaderboard AP.
4. Study the labelled suspicious flows in train (what does a hard case look like?) and write features for them.

## Feature backlog (highest value first)
- Multi-hop forwarding chains (depth of a time-ordered path where ≥90% of the value carries on).
- Time-respecting cycles (return to origin within N hours, value retained, assets changed on the way).
- Fan-in/out amount similarity within windows. Fresh-account batches (same funder, used once).
- Asset hopping (count of conversions along a flow). USD-based structuring (just under round thresholds).
- Neighbour aggregation: counterparties' pass_through / stage-1 score (2-hop stacking).
- Then try XGBoost or a second LightGBM config, and rank-average.

## How to work (Pro plan: tokens are the scarce resource)
- Be terse. Run code yourself; print summaries (shapes, AP, top features), never whole frames.
- After every improvement: commit, then submit with `sub_vN.csv` named after the commit. Log it in NOTES.md.
- Long runs go in the background. Never send raw data files to an external service.
- If context passes ~70k tokens, update NOTES.md and start a fresh session that reads this file and NOTES.md.
