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

## Leaderboard uploads (fill in tomorrow)
| time | commit | file | CV AP | LB AP | change |
|---|---|---|---|---|---|
