# AI and external-service disclosure (required by the T&Cs, cl. 3.3)

| Tool | Provider | Used for | Settings |
|---|---|---|---|
| Claude Code (Claude Opus 5.5 / Sonnet 5.5) | Anthropic | Writing and debugging pipeline code, feature ideas | default |
| Gemini app, model "Argon 4" | Google | Wrote an independent practice-data generator (`src/mock_gemini.py`) and reviewed our method before the event | thinking level High |

- Prompts: the Claude Code session transcript and the two Gemini prompts are exported and attached at the end of the day.
- No organiser data was sent to any service other than those listed. No external transaction data was used.
- Reproducibility limit: LLM outputs are not deterministic, but the submitted code is. `run.sh` reproduces the predictions exactly (fixed seeds).
