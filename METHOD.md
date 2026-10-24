# Method: Decode & Detect (XRPL AI Hack, 24 Oct 2026)

**Result:** leaderboard AP **67.2** (precision 94% / recall 30% in the top 1%). Progress: 58.1 → 59.9 →
62.0 → 62.3 → 63.9 → 65.0 → 65.4 → 65.7 → 66.1 → 66.9 → 67.0 → **67.2**. Reproduce with `./run.sh` (organiser data in `data/train`,
`data/test`; ~50 min and ~1.5 GB RAM on a laptop CPU); it rebuilds the final uploaded file (scored 66.1) byte for byte on our machine; on another CPU, XGBoost may differ in the last digits.
Scores mean "worth a human look" on synthetic data, never evidence of wrongdoing.

## 1. What the data taught us (before any modelling)
- 700 wallets, ~57 payments each; 535 of the 700 touch some planted scenario. Wallet identity carries no signal, and
  propagating scores along wallets (stacking) made things worse.
- Fees, flags, tags, memos and results are identical for everyone: only **behaviour** separates the classes.
- Every planted shape (cycle, chain, fan-in/out, bipartite, stack, multi-stage, random) has innocent look-alikes of
  the same shape. The difference is mostly **speed and exactness**: easy planted flows finish in ~5 minutes (~1 min
  between hops) and forward ~99% of what arrives; look-alikes take ~1 hour and lose ~8-11% per hop.
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
LightGBM (learning rate 0.02, 15 leaves, at least 60 rows per leaf), averaged over 5 seeds, blended by rank with one
single-seed XGBoost (weight chosen out of fold). Five modelling changes mattered more than any feature:
1. **Early stopping on log-loss instead of AP.** AP on ~250 validation positives is jumpy, so every fold stopped at
   a lucky round. Switching cut seed-to-seed CV noise from about ±2.5 to ±0.5 AP and raised CV by ~2.
2. **A family helper model.** The training labels name the scenario family of *every* payment (innocent look-alikes,
   background and relays included). A 13-class model predicts the family out of fold; its probabilities become
   features, so the main model knows which shape a payment sits in. That matters because the suspicious-vs-look-alike
   rules flip by shape (planted cycles avoid conversions; planted fan-outs use them). +1.1 CV on all three seeds.
3. **A joint helper** (`src/gemini_modelling.py`, written by Gemini, made deterministic by us). Two more out-of-fold
   models predict family × suspicious (~22 classes) and a 6-way difficulty regime; their 25 probabilities become
   features, so the model can tell a planted fan-out from an innocent one directly. +0.8 on the leaderboard.
4. **A pattern × difficulty helper** (`src/famdiff.py`, ours). The joint helper merges easy and medium positives of a
   pattern into one class; this out-of-fold model predicts e.g. stack-medium vs stack-easy vs stack-innocent (~30
   classes), giving the main model an explicit per-pattern medium signal. +0.1 on the leaderboard (66.9 → 67.0).
5. **Self-training on the test week** (`src/selftrain.py`, ours; no labels). The ensemble's top 600 test payments
   are pseudo-labelled suspicious and those ranked below 6,000 innocent; a small LightGBM is cross-fitted on the
   test week's own features and its rank is blended in at weight 0.25. It adapts the ranking to the week shift and
   lifts other payments of confidently flagged flows. +0.2 on the leaderboard (67.0 → 67.2).

**Final submission** (`run.sh`): six models, each a 5-seed LightGBM blended with a single-seed XGBoost and using all three helpers, rank-averaged, then test-week self-training: three copies
(seeds 42, 7, 11) valued in *nominal* units and three in true per-asset USD (see section 6 for why both), then
**burst smoothing**: payments sharing a wallet within 7.5 minutes form a group, and each score becomes
0.5 × its own rank + 0.5 × the rank of its group's best score, so one confident payment in a fast flow lifts the rest.

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
| v6 | + true per-asset USD (bug fix) | 61.0 | 62.8 (alone) |
| ens | rank-average of v5 and v6 | - | 64.1 |
| ens + smooth | + burst smoothing (5 min) | - | 65.0 |
| 4 models | v5 and v6, two seeds each | - | 65.4 |
| 4 models, 7.5 min | wider smoothing window | - | 65.7 |
| 6 models | six models (three seeds each), 7.5-minute smoothing | - | 66.1 |
| 6 models + joint helper | + joint family × suspicious and difficulty helper | - | 66.9 |
| 6 models, 3 helpers | + pattern × difficulty helper | - | 67.0 |
| **final** | + test-week self-training (top 600, weight 0.25) | - | **67.2** |

Late tuning that did not beat the six-model ensemble (66.1) on the leaderboard: eight models (66.0), smoothing strength 0.35 / 0.65 (66.0 /
65.8), a 9-minute window (65.6), multi-scale smoothing (65.9), weighting one model double (65.3), per-week
feature percentiles (65.8 vs 65.4 with two models, but 65.7 with six), the six models plus their six week-normalised
twins (66.1, a tie, so the simpler six-model ensemble stays), covariate-shift weights from a train-vs-test classifier
(65.4, no change with two models). With the joint helper, adding the six models without it gave 66.8 (vs 66.9). On top of the joint-helper
ensemble: 5- or 10-minute smoothing (66.6 / 66.5), smoothing strength 0.6 (66.7), and re-ranking by the helper's
medium-case probabilities (66.6-66.8, although +1.2 out of fold).

Tried and rejected after testing (2-3 seeds each): stacking on neighbour scores, memo and failed-payment features
(the real data has none), Gemini's relay, repeat-burst and twin-specialist ideas, looser trees.

## 6. Interesting findings
- Seed noise, not features, was the main obstacle: half of our early "improvements" were coin flips.
- Several features separated suspicious from innocent payments strongly on their own (AUC 0.8-0.9) yet hurt the full
  model, because it already held the same information. Auxiliary targets, not more features, unlocked the gain.
- Planted collectors in the relay families pay out after an almost constant hold (~4.3 h, variation < 15% vs ~70% for
  innocent relays), but rebuilding those episodes blind on the test week was too noisy to help.

- An independent Claude Code review session found that every asset had been priced as XRP (the rates file keys assets as
  `issuer:currency`). Fixing it raised CV by ~2.4 but *lowered* the leaderboard by ~1, so prices seem to shift between
  weeks; the final ensemble averages a model on nominal values with one on true USD, which beat both (64.1 vs 63.9).
- **Burst smoothing** (from the review): each payment's score is averaged with the best score among payments sharing a
  wallet within 5 minutes (7.5 in the final), so one confident payment in a fast flow lifts the rest. +0.7 to +1.2 on the leaderboard.
- Medium cases rank *below* their innocent look-alikes (AUC 0.40 inside the motif families): they are slower than the
  look-alikes, and the model learned "fast means suspicious" from the easy cases. 27 training scenarios cannot fix that.

## 7. Limits
Synthetic data; medium and hard cases are mostly missed (0% of them in the top 1%); the 25 ambiguous cases are random
by design. CV for the three helpers is slightly optimistic: their out-of-fold predictions reuse the main model's folds
instead of nested cross-fitting, so a validation fold's labels reach the helper features of the other folds (the
leaderboard, which never sees labels, is the number we trust). A train-vs-test classifier separates the two weeks
perfectly (AUC 1.0: amounts, conversion mix and burst sizes shift between weeks). That is week shift, not a label leak;
averaging nominal- and true-USD models by rank is our hedge, and re-weighting training rows towards the test week did not help. AI use (Claude Code, Gemini) is listed in `AI_DISCLOSURE.md` with the prompts in `docs/` and `gemini_in/`.
