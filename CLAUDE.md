# Decode & Detect: XRPL AI Hack (Sat 24 Oct 2026, IDEALondon)

Faizan's team entry. This repo was pre-built the night before against a MOCK simulator in the real schema.
**Hard deadline: 16:30 London.** Upload at hackathon.ukcbt.org with the team's organiser access code.
Faizan is a first-year CS student who is new to blockchain and ML: explain terms plainly when it matters.

## The task
Score every test transaction for laundering-style behaviour. Supervised: training labels are given.
- Data (synthetic, AWS/SonarX XRPL schema), per split: `transactions.parquet`, `transfers.parquet`
  (1-3 money legs per tx), `blocks.parquet`, `market_rates.csv` (hourly USD per asset), `labels.csv`
  (train only). Train is 40,131 tx, test is 40,101 tx, each a separate 7-day window. The organisers ship
  `README.md`, `payments.py` (one edge per payment) and `baseline.py`. **Read README.md first.**
- Metric: **100 × Average Precision** over all 40,101 test tx. Only the ORDER matters. Ties are scored as
  one block, so keep scores continuous. Also shown (not ranked): precision/recall in the top 402, ROC-AUC.
- The leaderboard scores the full test set. It is the real score, so submit after every real improvement.
  **50 uploads per team per UTC day**, and the best completed run shows.
- Suspicious mix: easy 64%, medium 18%, hard 18%. "Hard means history": multi-payment flows with changing
  counterparties. The test set has variations not seen in training, plus a few deliberately ambiguous cases.

## Submission (two files)
1. Predictions CSV: `TRANSACTION_HASH,score`, every test tx exactly once, score in [0,1], ≤12 MiB.
2. Code ZIP (≤8 MiB, `./make_zip.sh`): code, exact run command (`run.sh`), seeds, `requirements.txt`, team
   list, commit hash, results, and the method (<10 slides or 4 pages). No data, no venv, no secrets.
- Rules: organiser data only (public docs and papers are fine). AI allowed if disclosed in
  `AI_DISCLOSURE.md`. Must be reproducible. Open source (MIT `LICENSE` is in place) and public by the
  deadline. Language: say "flagged for review", never "criminal".

## Repo
- `src/pipeline.py`: the whole job. Flags: `--fast` (1 seed, 1 stacking round), `--drop block1,block2`
  (ablation; blocks: tx account history timing chains match cycles sg graph iforest conv memo probe), `--stack N`,
  `--seeds N`, `--xgb` (XGBoost blended in if OOF AP improves), `--dump` (writes runs/oof.parquet).
  Opt-in (did not help on either practice set, may on real data): `--add subset,dormancy`, `--vtags` (split
  busy receivers per destination tag), `--local-stack` (+-24h neighbour scores), `--stack-dropout 0.3`.
  Every run appends to `runs/log.csv` (time, commit, seed, n_features, CV AP per round, final CV AP, drop, file).
- `src/inspect_errors.py` (after `--dump`): worst misses and false alarms, plus the money flow around them.
- `src/mock_gemini.py`: an INDEPENDENT practice generator written by Gemini (Argon 4) from the challenge brief
  only. It's much harder than ours: warm mules, amount-matched, 1.5-5 day hard cases, issuer swaps, probes,
  memo templates. Test every change on BOTH generators; keep it only if neither gets worse.
- `src/ensemble.py`: rank-average CSVs (ours plus teammates'). `src/mock_data.py`: our practice-week simulator
  (innocent twins for every shape, recruited mules, slow hard cases, test-only shapes). `src/eval_mock.py`:
  AP by difficulty and shape on the mock.
- Column names follow the real SonarX parquet files (checked 2026-10-23): transfers have TRANSACTION_HASH,
  TRANSFER_INDEX (-999 = direct), FROM_ADDRESS, TO_ADDRESS, TOKEN_ADDRESS, SYMBOL, SOURCE_VALUE (drops for
  XRP), VALUE (human units), DATETIME. Transactions have ACCOUNT, TRANSACTION_TYPE, FEE, TRANSACTION_RESULT,
  DELIVERED_AMOUNT (JSON text), EXTRA_TRANSACTION_FIELDS (JSON text: Destination, Amount, SendMax,
  DestinationTag), MEMOS. **The market_rates and labels column names are guessed** (`find_col` matches on
  substrings), so check them against the README.

## Design decisions (and why)
- CV groups keep each laundering case (a connected component of positive edges, hubs excluded) in one
  fold. Random or day folds leak half a case into training. Groups larger than 1/40 of the data are split by
  day, because cases sharing mules chain into one giant group that starves a fold. (On the mock that giant
  group made CV read 83-86 while the hidden test read 99.)
- Value matching (`match` block) follows money by amount, 80-102% of what arrived, up to 7 days later. It
  survives slow hops and mules who make ordinary payments in between. "Next payment" links do not.
- Stacking uses leave-one-out neighbour scores: same wallet, funding tx, next tx, matched prev/next, same pair.
- Hubs (degree above the 99.5th percentile and at least 50) are excluded from the graph searches.

## What the practice runs taught us (2026-10-24, see NOTES.md)
- On Gemini's data: 83.3 -> 93.7 after adding conv (issuer swaps, 3-leg, conversion habits), memo
  (hex-decoded text pattern + out-of-fold laundering rate per pattern) and probe (failed attempts) blocks.
  Almost all of the hard-case gain (9.7 -> 59) came from memo patterns. Without memos its hard cases stay
  ~9 AP whatever we add, so check early whether the real MEMOS column carries signal.
- Our CV overstates on shifted data (Gemini: CV 97 vs test 94). Trust the leaderboard over CV for close calls.

## Gemini (second model)
Gemini CLI sign-in is refused for Faizan's account. Use gemini.google.com in Chrome: model "4 Argon",
Thinking level High (recipe in memory `gemini-access-route`). Use it for independent ideas and reviews,
never for the organisers' raw data. Test every suggestion before keeping it.

## First 30 minutes tomorrow (in this order)
1. Put the data in `data/train`, `data/test`. Read README.md. Fix load-layer names if needed (top of pipeline.py).
2. Run `baseline.py` and submit it. That proves the upload works and gives the number to beat.
3. `pipeline.py --fast`, then the full run, then submit. Compare CV AP with the leaderboard AP.
4. Check the MEMOS and DestinationTag columns: how often filled, and the label rate by memo pattern
   (memo_prefix_te in the importance list). If tags are heavily used on exchange deposits, try `--vtags`.
5. `--dump`, then `inspect_errors.py`. Study what the misses look like and add features for them.
6. Ablate blocks that might not transfer (`--drop`), try the opt-in flags, keep what the leaderboard confirms.

## How to work (Pro plan: tokens are the scarce resource)
- Be terse. Run code yourself; print summaries (shapes, AP, top features), never whole frames.
- After every improvement: commit, then submit `sub_vN.csv` named after the commit. Log it in NOTES.md.
- Long runs go in the background. Never send raw data files to an external service.
- If context passes ~70k tokens, update NOTES.md and start a fresh session that reads this file and NOTES.md.
