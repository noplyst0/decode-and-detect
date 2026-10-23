# Brief: write an adversarial synthetic dataset generator for an AML hackathon

You are helping a team prepare for "Decode & Detect: The XRPL AI Hack" (London, tomorrow). The organisers
will release synthetic XRP Ledger data and ask teams to score every test transaction for money-laundering
behaviour, ranked by Average Precision. The team has a model already. Your job is to build an
INDEPENDENT test bed: a generator, designed by you from the spec below, that is realistic and deliberately
hard, so the team can find out where their model breaks. Do not try to make it easy. Think like the
organisers, who want to separate strong teams from average ones.

## What the organisers told participants (this is all we know)
- "A fictional ledger in the real XRPL format." Synthetic data in the AWS/SonarX XRP Ledger schema.
- Train: 40,131 labelled transactions over 7 days. Test: 40,101 transactions over a different 7 days.
- Suspicious transactions by difficulty: easy 64%, medium 18%, hard 18%.
- "Lots of normal life: shopping, peer payments, exchange and payroll hubs, legitimate treasury flows.
  Both sides use conversions."
- "Hard means history: hard cases span many payments and changing counterparties. Look at whole flows,
  not single amounts."
- "Test is not train: the test data has variations absent from training. A few cases are deliberately ambiguous."
- "A suspicious label describes the scenario's intent." Labels are per transaction.
- Laundering pattern families (from IBM's AMLworld datasets, Altman et al., NeurIPS 2023): fan-in, fan-out,
  scatter-gather, gather-scatter, cycle, layered stack (rows of intermediaries), bipartite (group pays
  group), irregular network (splits, merges, returns). Every one of these also happens in normal business
  (shops, payroll, exchanges, payment processors, market making, company treasury, friends).
- Mostly the layering stage: money moving between wallets. Placement can be an exchange withdrawal or a
  mule wallet; integration is often an exchange deposit (with a destination tag).
- XRPL facts that hold in the data: addresses start with "r". 1 XRP = 1,000,000 drops. Tokens are
  issuer + currency code (USD from two issuers = two assets). A cross-currency payment is ONE transaction
  with 2 or 3 transfer rows via one or two liquidity providers. Every transaction burns 10-15 drops of fee,
  even token payments. SendMax is a ceiling; the delivered amount is what arrived; the first leg is what was
  paid. Failed transactions (tec... results) stay on the ledger, pay the fee and move nothing.

## Required output files (exact schema)
For each split, write a folder `<out>/train/` and `<out>/test/` containing:
1. `transactions.parquet`, one row per transaction, columns:
   TRANSACTION_HASH (64 uppercase hex), LEDGER_INDEX (int), DATETIME (UTC timestamp), ACCOUNT (sender),
   TRANSACTION_TYPE ("Payment"), SOURCE_FEE (int drops), FEE (float XRP), TRANSACTION_RESULT
   ("tesSUCCESS" or e.g. "tecPATH_DRY"), SUCCESS (bool), DELIVERED_AMOUNT (JSON text: a drops string for
   XRP, or {"currency","issuer","value"} for tokens; "unavailable" if failed), EXTRA_TRANSACTION_FIELDS
   (JSON text with "Destination", "Amount", optional "SendMax", optional "DestinationTag"), MEMOS (null or text).
2. `transfers.parquet`, one row per money leg, columns: TRANSACTION_HASH, TRANSFER_INDEX (-999 for a
   single-leg payment, else 0,1,2), DATETIME, FROM_ADDRESS, TO_ADDRESS, TOKEN_ADDRESS (issuer address, or
   "0x1111222211112222111122221111222211112222" for XRP), SYMBOL ("XRP","USD","EUR","GBP"...), SOURCE_VALUE
   (drops for XRP, units for tokens), VALUE (human units), TRANSFER_TYPE. Failed transactions have no legs.
3. `blocks.parquet`: LEDGER_INDEX, CLOSE_TIME.
4. `market_rates.csv`: HOUR, SYMBOL, TOKEN_ADDRESS, USD_PRICE (hourly, with realistic drift).
5. Train only: `labels.csv` with TRANSACTION_HASH, LABEL (0/1).
Also write `<out>/test_labels_hidden.csv` with TRANSACTION_HASH, LABEL, SHAPE, DIFFICULTY (SHAPE and
DIFFICULTY empty for normal transactions) so the team can score per shape and difficulty.

## Design requirements
- About 40,000 transactions per week. Choose a plausible suspicious rate yourself (somewhere in 1-5%).
- Realistic normal life with legitimate twins of EVERY laundering shape, in meaningful volume.
- Laundering that hides well: overlapping amounts with normal activity, mules with genuine-looking
  histories, delays from minutes to days, conversions, changing counterparties, interleaving with normal
  payments. Easy/medium/hard should genuinely differ, in the stated 64/18/18 mix.
- The test week must contain variations absent from training: at least one shape or variant that never
  appears in train, and shifted parameters. Include a few deliberately ambiguous cases.
- Use your own ideas about how the organisers might generate this. Do not just mirror the list above;
  add realistic complications (e.g. dust/spam, failed transactions, exchange internal transfers,
  partial-payment-like behaviour, bots) where you think the organisers would.

## Format of your answer
A single self-contained Python 3 script using only numpy, pandas and pyarrow, runnable as
`python gen.py --out DIR --seed N`, that finishes in under 2 minutes. Put it in ONE ```python code block.
Before the code, give at most 10 bullet points on the design choices that make it hard.
