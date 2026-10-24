# Method: Decode & Detect (XRPL AI Hack, 24 Oct 2026)

**Result:** leaderboard AP **[FILL]** (commit `[FILL]`, seed 42). Baseline AP **[FILL]**.
Reproduce: `./run.sh` with the organiser data in `data/train` and `data/test`.

## 1. Problem
Score every test transaction so that activity worth flagging for review ranks first. The metric is
average precision, so only the order matters. We keep scores continuous to avoid tied blocks.

## 2. Features (per transaction, ~150)
- **Transaction:** type, fee, result, legs, USD value from hourly market rates, round or just-under-threshold
  amounts, destination tag, decoded memo text pattern.
- **Account history:** the sending and receiving wallets' age within the window, activity bursts, fan-in and
  fan-out, how fast money leaves after it arrives, dormancy.
- **Flow following:** *value matching* follows money by amount (80-102% of what arrived, up to 7 days
  later), so slow hops and mules who make ordinary payments in between are still linked. On top of that:
  chains, cycles (money returning to its origin), and timed subgraph features.
- **Currency behaviour:** conversions, swaps between two issuers of the same currency, three-leg payments.
- **Probes:** failed payment attempts shortly before a successful transfer on the same route.
- **Graph:** degree, PageRank-style centrality and an Isolation Forest anomaly score. Exchange-like hubs are
  excluded from the graph searches so they do not link unrelated users.

## 3. Model
LightGBM, averaged over seeds, then **stacking**: a second round adds each transaction's neighbours'
out-of-fold scores (same wallet, funding transaction, next and matched transactions, same pair), so a weak
signal on one hop is supported by its neighbours.

## 4. Validation without leakage
Random folds leak half of a case into training. We group each case (a connected component of flagged
payments, hubs excluded) into one fold. Groups bigger than 1/40 of the data are split by day, because cases
that share mules otherwise chain into one giant group.

## 5. Testing against data we did not write
Before the event we built two practice generators in the real schema: our own, and an independent one that
Gemini wrote from the challenge brief alone, without seeing our code. Our pipeline fell from 99 on our data to
83 on Gemini's, which exposed blind spots (issuer swaps, memos, probe payments). Fixing those raised it to
93.7. Noise floor across generator seeds: **[FILL from NOTES.md]**.

## 6. Results on the real data
| version | change | CV AP | leaderboard AP |
|---|---|---|---|
| baseline | organiser `baseline.py` | - | [FILL] |
| [FILL] | | | |

What mattered most: **[FILL: top features / ablations]**.

## 7. Limits
Synthetic data; a score means "worth a human look", not proof of wrongdoing. The deliberately ambiguous
cases will produce false alarms. AI use is listed in `AI_DISCLOSURE.md`.
