# Gemini prompts used during the event (24 Oct 2026)

Model: Gemini app, "4 Argon", thinking level High, one chat. Gemini never received organiser data: only aggregate
statistics we computed and the column layout of our own feature table. Its raw answers are in `gemini_in/`.

1. **Feature design** (full text: `docs/gemini_prompt_features.md`). Our aggregate findings (family sizes, timing and
   pass-through medians for planted vs look-alike scenarios) and the edge-table schema; asked for an unsupervised
   `gemini_features(e)` function. Result: `src/gemini_feats.py`, kept (+0.9 CV over 2 seeds).
2. **Relay analysis.** Asked for `relay_report(e, lab)`, a script we ran locally on the training labels, returning
   only per-group medians and AUCs. Found the regular-hold signal in planted relays.
3. **Relay features.** Sent those aggregate results back; asked for unsupervised `relay_features(e)`.
   Result: `src/gemini_relay.py`, rejected (no hard-case gain).
4. **Repeat-burst features.** Sent our out-of-fold error profile (false alarms per family) and a hypothesis; asked for
   `repeat_features(e)`. Result: `src/gemini_repeat.py`, rejected (worse in the full model).
5. **Medium analysis.** Asked for `medium_report(e, lab)` (aggregate output only). Found no strong signal.
6. **Modelling review.** Sent what worked (log-loss early stopping, family helper) and asked for the three best
   modelling changes. Result: `src/gemini_modelling.py` (joint helper, sample weighting), tested as options.
