# Method: Decode & Detect (XRPL AI Hack, 24 Oct 2026)

**Result:** leaderboard AP **[FINAL]** (commit `[FINAL]`, seed 42, 5-seed average). Progress: 58.1 → 59.9 → 62.0 →
62.3 → 63.9 → [FINAL]. Reproduce with `./run.sh` (organiser data in `data/train`, `data/test`; ~3 min, ~1 GB RAM, CPU).
Scores mean "worth a human look" on synthetic data, never evidence of wrongdoing.

## 1. What the data taught us (before any modelling)
- 700 wallets, ~57 payments each; 535 of the 700 touch some planted scenario. Wallet identity carries no signal, and
  propagating scores along wallets (stacking) made things worse.
- Fees, flags, tags, memos and results are identical for everyone: only **behaviour** separates the classes.
- Every planted shape (cycle, chain, fan-in/out, bipartite, stack, multi-stage, random) has innocent look-alikes of
  the same shape. The difference is mostly **speed and exactness**: easy planted flows finish in ~5 minutes (~1 min
  between hops) and forward ~99% of what arrives; look-alikes take ~1 hour and keep ~8% per hop.
- Medium cases are the same motifs slowed to ~18-minute hops inside normal traffic; hard cases hide among ~10,000
  innocent relay payments per family. Both are close to unlearnable from the 27 and ~12 training scenarios they have.

## 2. Features (~250 per transaction)
- **Payment:** USD value from the hourly reference rates (asset = issuer + currency), legs, conversions, SendMax.
- **Account history and timing:** activity, fan-in/out, time since funded, time to the next and previous hop.
- **Flow following:** value matching (money followed by amount), chains, cycles, scatter-gather, graph components.
- **Bursts:** payments that share a wallet within 2/5/15/45 minutes are linked into groups (single linkage); each
  payment gets its group's size, duration, rate, wallets per payment, shape (distinct senders and receivers, max in-
  and out-degree, wallets that both send and receive), conversion share, asset count and median USD.
- **Motif features designed by Gemini** (`src/gemini_feats.py`) from our aggregate findings only: amount-filtered
  motif reconstruction, per-shape conversion interactions, split- and merge-aware pass-through.

## 3. Model
LightGBM (learning rate 0.02, 15 leaves, at least 60 rows per leaf), averaged over 5 seeds, blended by rank with
XGBoost (weight chosen out of fold). Two modelling changes mattered more than any feature:
1. **Early stopping on log-loss instead of AP.** AP on ~250 validation positives is jumpy, so every fold stopped at
   a lucky round. Switching cut seed-to-seed CV noise from about ±2.5 to ±0.5 AP and raised CV by ~2.
2. **A family helper model.** The training labels name the scenario family of *every* payment (innocent look-alikes,
   background and relays included). A 13-class model predicts the family out of fold; its probabilities become
   features, so the main model knows which shape a payment sits in. That matters because the suspicious-vs-look-alike
   rules flip by shape (planted cycles avoid conversions; planted fan-outs use them). +1.1 CV on all three seeds.

## 4. Validation
5-fold GroupKFold with the organisers' `scenario_id` as the group, so no scenario is split between training and
validation (our first grouping by wallet components merged everything into 8 groups and gave meaningless CV).
Every decision was compared over 2-3 seeds, because single-seed differences under ±1 AP were noise.

## 5. Results
| version | change | CV AP | leaderboard AP |
|---|---|---|---|
| v1 | base pipeline, scenario-grouped CV | 50.2 | 58.1 |
| v2 | + burst features | 54.5 | 59.9 |
| v3 | + burst shape/conversion, simpler trees, 5-seed average | 56.8 | 62.0 |
| v4 | + Gemini motif features, XGBoost blend | 56.6 | 62.3 |
| v5 | log-loss early stopping + family helper | 58.6 | 63.9 |
| [FINAL] | | | |

Tried and rejected after testing (2-3 seeds each): stacking on neighbour scores, memo and failed-payment features
(the real data has none), Gemini's relay, repeat-burst and twin-specialist ideas, looser trees.

## 6. Interesting findings
- Seed noise, not features, was the main obstacle: half of our early "improvements" were coin flips.
- Several features separated suspicious from innocent payments strongly on their own (AUC 0.8-0.9) yet hurt the full
  model, because it already held the same information. Auxiliary targets, not more features, unlocked the gain.
- Planted collectors in the relay families pay out after an almost constant hold (~4.3 h, variation < 15% vs ~70% for
  innocent relays), but rebuilding those episodes blind on the test week was too noisy to help.

## 7. Limits
Synthetic data; medium and hard cases are mostly missed (0% of them in the top 1%); the 25 ambiguous cases are random
by design. AI use (Claude Code, Gemini) is listed in `AI_DISCLOSURE.md` with the prompts in `docs/` and `gemini_in/`.
