# AI and external-service disclosure (required by the T&Cs, cl. 3.3)

| Tool | Provider | Used for | Settings |
|---|---|---|---|
| Claude Code (Claude Opus 5.5 / Sonnet 5.5) | Anthropic | Writing and debugging pipeline code, feature ideas | default |
| Gemini app, model "Argon 4" | Google | Wrote an independent practice-data generator (`src/mock_gemini.py`) and reviewed our method before the event | thinking level High |
| ChatGPT | OpenAI | Two second-opinion reviews of our code and notes on the day (see below) | default app settings |
| DeepSeek | DeepSeek | One second-opinion review of the final write-up and run instructions | default app settings |

- Prompts: the Claude Code session transcript and the two Gemini prompts are exported and attached at the end of the day.
- No organiser data was sent to any service other than those listed. No external transaction data was used.
- Reproducibility limit: LLM outputs are not deterministic, but the submitted code is. `run.sh` reproduces the predictions exactly (fixed seeds).

## During the event (24 Oct)
- Gemini app, "4 Argon", thinking High: given only aggregate statistics (never the organisers' data), it wrote
  `src/gemini_feats.py` (motif features, kept after a 2-seed A/B), `src/gemini_modelling.py` (the joint
  family/difficulty helper used in the final: +0.8 on the leaderboard; we added deterministic LightGBM settings) and an analysis script whose aggregate output guided
  the relay features. Prompts are in `docs/` and `gemini_in/`. We tested every suggestion before keeping it.
- A second Claude Code session (same model family) did an independent code review (`review/REVIEW.md`). It found the
  USD valuation bug and proposed burst smoothing (`src/smooth_sub.py`, `review/burst_smooth.patch`); both were tested
  on the leaderboard before being kept. It never had access to anything beyond this repository and the organiser data.
- ChatGPT, given our code, notes and write-up (never the organisers' data): round 1 (`docs/second_opinion_prompt.md`)
  suggested testing per-week rank normalisation, the joint helper (`JOINT_AUX`) and covariate-shift weights, and flagged
  the family-helper CV caveat and the XRP-rate assumption in `usd_rates`. We tested all three on the leaderboard: only
  the joint helper helped (66.1 -> 66.9) and is in the final. Round 2 (`docs/second_opinion_prompt_2.md`) reviewed the
  write-up. DeepSeek got the same round-2 prompt; the documentation errors it found were checked against the code and
  fixed (commit 92c62e1). No code from either service was pasted into the repository.
- What the team decided and verified: every change was kept only after a 2-3 seed CV check or a leaderboard upload;
  the final ensemble and smoothing settings were chosen on the leaderboard (see `NOTES.md`).
