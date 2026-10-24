# Pitch notes: three flows our model flagged (test week, final model, LB 66.9)

Scores mean "flagged for review", never proof of wrongdoing. Ranks are out of 40,101 test payments.

## 1. Smurfing: split into small payments, regroup at a collector (Sun 02:15, 12 payments in 3.8 min)
Wallet `rNjNitR…` sends six USD payments of $12-26 to six wallets within 2 minutes; within the next 2 minutes each of
them pays a similar amount ($20-29) to one collector `rUYRMFC…`.
- First payment `77BEEAE7C370E5D07471C02D42E80AB07F346BC027500EFD2BADC77A3921BF05` (rank 2); last leg into the collector `4B1DC00A16B516F3A22DBFB2BB006114AAFEC19EC65A02180367ACCD26A9BD53` (rank 1).
- What caught it: fan-out then fan-in inside one 5-minute burst (burst group of 12 payments), money passed on within
  minutes, the joint helper's "planted fan-out/in" probability 0.98. All 12 payments rank in our top 20.

## 2. Currency hopping (Tue 06:28, 7 payments in 3.7 min, every one a conversion)
EUR is split three ways and converted on the way (EUR->XRP, EUR->USD, EUR->USD); the three receivers convert back to
EUR into one collector `r3vEgYQ…`, which immediately sends the total on as XRP.
- First payment `D938E3489B7F7B5612228577146D4528828385BA659D463F8B0CC2ED6FA34583` (rank 8); exit payment `0D91C969FFF94FF81C20400172A6215B93F6B7253271C09FE630DE7FD890D08F` (rank 50).
- What caught it: conversion habits (an asset swap on every hop, out and back), the split-then-collect shape, and
  the 4-minute burst.

## 3. Round trip (Fri 11:50, 7 hops in 6 min)
`rJE4JzP…` pays $191 into a chain of six wallets; each pays on within about a minute ($157-190 per hop) and
the last hop pays `rJE4JzP…` $157.
- First payment `3EBEB7B2CD8ED4B6DCCD0A623DB4F7BDD5A6CB98EE8DE8CFFA2E21BDC95FBD59` (rank 27); the payment that closes the loop `47ED315C742EEFA03E7EE419E7957A949602E26F3875E27D02550ED52CBD6467` (rank 20).
- What caught it: money returning to its origin (our cycle features search for time-ordered loops), fast
  pass-through, one burst.
- Precision detail: an unrelated small payment by a wallet in a similar flow (`3378EF40138D1346…`) sits at rank 10776: the
  model scores the flow, not the wallet.

## Honest limits (say these)
- We catch the fast, exact flows well: 94-95% of our top 1% are true alerts and half of all easy cases are there.
- Medium cases (same shapes, but slower and less exact than innocent look-alikes) and the planted relays mostly escape:
  the organisers' innocent twins copy their timing, and our tests of hold-time regularity found no separation.
- Synthetic data: no claim about real-world laundering detection. 25 ambiguous cases are random by design.
