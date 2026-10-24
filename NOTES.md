# Experiment log

Mock = my simulator (src/mock_data.py), NOT organiser data. mockB = 230 schemes, seed 5. Test AP is on the
mock's hidden test week, which has two shapes never seen in training (irregular, peel_chain) and 30% slower flows.

| when | change | CV AP | mock test AP | hard | medium | notes |
|---|---|---|---|---|---|---|
| 10-23 22:15 | first pipeline, easy mock | 99.7 | 100 | - | - | mock too easy to tell anything |
| 10-23 23:30 | v2: chains, cycles, timed SG, history, iforest, scenario-grouped CV, 2 stack rounds | 96.2 | 98.5 | 74.3 | 96.3 | hard mock (90 schemes) |
| 10-23 23:45 | + value matching (match block) | 97.0 | 98.4 | 70.7 | 96.3 | 161 hard positives: too noisy to judge |
| 10-24 00:10 | mockB, CV groups un-capped | 83.4 | 99.06 | 90.2 | 96.2 | CV broken: cases sharing mules chain into one giant group |
| 10-24 00:20 | mockB, groups >1/40 split by day | 98.26 | 99.14 | 90.1 | 96.4 | CV now honest |
| 10-24 00:20 | mockB, --drop match | 98.13 | 99.10 | 90.1 | 95.7 | match helps medium and peel chains (82.3 -> 84.5) |
| 10-24 00:20 | mockB, --stack 1 | 98.06 | 99.12 | 90.2 | 96.1 | 2nd stacking round adds ~nothing: --stack 1 is a safe speed-up |
| 10-24 00:30 | scrambled schema copy (mockV) | 98.06 | 99.11 | - | - | loader handles renamed files/columns, numeric times |

| 10-24 00:05 | Gemini's independent generator (mockG, seed 7), pipeline as above | 94.1 | 83.3 | 9.7 | 36.5 | our blind spot: warm mules, issuer swaps, memos, probes |
| 10-24 00:25 | + conv, memo (+ target encoding), probe blocks, 5-min windows | 97.3 | **93.6** | 58.0 | 60.9 | mockB 99.18 (no harm) |
| 10-24 00:40 | + Gemini's review ideas: vtags, subset, dormancy, local24 stacking | 97.5 | 92.6 | 52.1 | 54.9 | worse; mockB 99.19 |
| 10-24 00:40 | same, --drop memo | 94.4 | 81.0 | 9.2 | 28.1 | memo carries the hard-case gain on Gemini's data |
| 10-24 00:50 | ablation: all four review ideas off (+ failed-payment USD kept) | 97.3 | **93.7** | 58.9 | 61.7 | default now; the four ideas are opt-in flags |
| 10-24 08:55 | NOISE FLOOR: Gemini data seeds 7/11/23 + model seed 1 on seed 7 (full run ~50s, --fast 38s) | 97.2-97.3 | 93.66 / 93.55 / 95.75; seed-1 model 94.23 | 58.9/55.0/67.9; 61.2 | 61.7/64.0/71.5; 63.6 | model seed alone moves overall ~0.6, hard/medium ~2-3: smaller gains are noise, confirm with a 2nd seed |

## Leaderboard uploads (fill in tomorrow)
| time | commit | file | CV AP | LB AP | change |
|---|---|---|---|---|---|
| 10:52 | (scenario CV) | sub v1 | 50.2 | 58.1 | --fast --stack 0, CV folds by organiser scenario_id |
| 10:58 | 428daff | sub v2 | 54.5 | 59.9 | + burst block (wallet-sharing groups within 2/5/15/45 min); --drop memo,probe |
| 11:58 | d3d5e6a | sub v5 | 58.59 | **63.9** | log-loss early stopping + family helper (3-seed CV 58.9); easy 50% in top 1% |
| 11:35 | b028bb6 | sub v4 | 56.64 | 62.3 | + Gemini motif features (src/gemini_feats.py, +0.9 CV over 2 seeds) + XGB blend 0.5 |
| 11:08 | 0ed7435 | sub v3 | 56.75 | **62.0** | + burst shape/conversion, LGB lr .02/15 leaves/min_child 60, 5-seed avg (AUC .90; medium 0% in top 1%) |

## Real data facts (10-24 morning)
- 700 wallets, ~57 tx each; 535 wallets touch a suspicious scenario: no wallet-level signal, stacking hurts.
- No memos, no failures, fees/flags/tags identical: only behaviour separates. Every family has innocent twins.
- Easy flows finish in minutes (gap ~1 min, pass on ~97%); look-alikes take ~1 h (~89% kept). Medium ~ look-alike.
- Hard = 100 positives hidden in each 10k relay family, ~7 scenarios each in train: not learnable, deprioritised.
- LGB lr 0.02, 15 leaves, min_child 60: CV 56.4 vs 54.5 default.
- Ablation (2 seeds): graph+history -3 AP (keep); iforest/sg, cycles/chains neutral; XGB blend +0.6; min_child 25 worse.
- Gemini relay_report (train, labels used only for analysis): suspicious collectors pay out after an almost constant hold
  (agg ~4.3 h, dist ~2.1 h, cv<0.15 vs ~0.7, AUC ~0.8) with steady per-episode pass-through. Only 6 scenarios/family.
