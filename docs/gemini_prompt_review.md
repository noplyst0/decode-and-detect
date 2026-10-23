Thanks. Second task: review our model as a sceptical competitor would. Same hackathon, same data spec.

## Our pipeline (one row per transaction; features are computed inside each 7-day split)
1. Edges: one row per transaction. Sender = ACCOUNT, receiver = Destination. USD sent = first leg x hourly
   rate, USD received = last leg. Conversion flag, leg count, destination tag, success flag.
2. Feature blocks (~125 columns):
   - tx: log USD, USD percentile, received/sent ratio, legs, conversion, XRP or token, fee, hour,
     "just under a power of ten", round amounts, self-payment, dest tag, SendMax, memo, success.
   - account (for sender AND receiver): week totals, distinct counterparties in/out, pass-through =
     min(in,out)/max(in,out), net flow, amount CV, active span, first-in to first-out delay, conversion share,
     distinct dest tags received, failure share, age at this payment.
   - history: this amount vs the sender's median, position in the sender's sequence, first/last use flags.
   - timing: hours since the sender was last funded and the ratio to that funding; hours until the
     receiver's next payment and its ratio; counts, amount-CV and share of sender out / receiver in within
     +-1h/6h/24h; collector exit (receiver's next payment vs everything it received in the 24h before).
   - chains: forwarding depth backwards and forwards via "next payment within 48h carrying 5-105%".
   - match: value matching. Receiver's first later payment carrying 80-102% of what it received, up to
     7 days later. Depth of matched chains, matched loop back to the sender (up to 8 hops), and "amount
     echo": how many payments this week carry the same USD amount within 0.2% / 1%.
   - cycles: BFS for the shortest time-respecting path receiver -> ... -> sender within 72h, max 6 hops,
     hubs (degree above the 99.5th percentile and at least 50) excluded. Length and value returned.
   - sg: timed scatter-gather. For s->m then m->d within 48h, distinct middlemen per (s,d), value kept.
   - graph: pair count, reciprocity, hub flags, same strongly connected component, SCC/WCC sizes.
   - iforest: isolation-forest anomaly score of the sender's and receiver's account vectors.
3. Model: LightGBM (lr 0.05, 31 leaves, feature_fraction 0.7, bagging 0.8), early stopping on AP.
   5-fold GroupKFold. Groups = connected components of positive edges (hubs excluded); negatives grouped
   by sender; any group bigger than 1/40 of the rows is split by day.
4. Stacking (2 rounds): add leave-one-out neighbour scores from the previous round's out-of-fold
   predictions: mean/max over the sender's other txs, over the receiver's other txs, the funding tx, the
   receiver's next tx, value-matched previous/next tx, mean over the same pair. Test uses the full-train
   model's test predictions.
5. 3-seed average, optional XGBoost blend chosen on OOF AP. Continuous scores in [0,1].

## Results on OUR OWN simulator (hidden test week, which has 2 shapes never seen in training)
Overall AP 99.1 (CV 98.3). Easy 99.8, medium 96.4, hard 90.1. By shape: fan-in 99+, scatter-gather 99+,
cycle 99.3, stack 97.3, bipartite ~98, gather-scatter ~97, fan-out ~95, irregular (unseen) 88.9,
peel chain (unseen) 84.5. Our worst misses are single laundering hops between two busy wallets
with ordinary histories, where nothing before or after the hop looks unusual.

## Results on YOUR generator
{GEMINI_RESULTS}

## What we want from you
1. The 7 most valuable concrete improvements, ranked by expected AP gain on the organisers' real data
   (not just our simulators). For each: the exact definition (columns, windows, thresholds) so it can be
   built in under 30 minutes with pandas/numpy/networkx/LightGBM, and why it should transfer.
2. Where are we most likely wrong about how the organisers generated their data, and what would you check
   in the first 30 minutes with the real files?
3. Any risk in our CV or stacking that could make our local score lie, or hurt us on a test week with
   variations absent from training.
Be specific and terse. No generic advice.
