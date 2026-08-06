# Voice Collection Agent (Claude Code skill scaffold)

Real-time voice call agent for EMI/loan collection outreach:
**Indic Conformer (STT) → Gemini/Ollama Qwen (LLM) → Parler-TTS (TTS)**, audio routed
through a paired **Bluetooth** device, with a **persona + delinquency-risk
guardrail harness** so you red-team compliance before any real call.

See `SKILL.md` for the full skill description (this is written as a Claude
Code skill — drop the folder into `.claude/skills/voice-collection-agent/`
and Claude Code will pick it up).

## Quick start

```bash
pip install -r requirements.txt

# 1. Run a complete offline demo (no audio, GPU, or API key required)
python3 app.py

# Open the model-selection and call-simulation browser UI
python3 app.py ui
# Then visit http://127.0.0.1:8080

# 2. Run the guardrail harness (no models/audio required, stub-safe)
python3 app.py test

# 3. Talk to the safe offline agent interactively
python3 app.py chat --risk-tier medium_risk

# 4. Only once guardrails pass AND real models are wired in (see TODOs below),
#    route through a live Bluetooth call:
python3 scripts/orchestrator.py --risk-tier medium_risk --live --bt-device "your device name"
```

`app.py` is the supported local entry point. Its default demo is intentionally
offline and deterministic, making it suitable for a fresh checkout and CI.

Browser UI conversations are stored in one file per session at
`conversations/<session-id>/conversation.json`. The file contains session
metadata and an ordered `turns` array. API keys and raw microphone audio are
never written to it.
Each turn includes nested STT and TTS details: backend, full model ID,
language/transcript, synthesized text, and model sample rate.

## What's stubbed vs. real

Everything **runs today** with a deterministic offline safety simulator so the pipeline and guardrail
harness are testable with zero GPU/API dependencies. Each of these has a
single `TODO(integration)` block to fill in:

| File | What to wire up |
|---|---|
| `scripts/stt_indic_conformer.py` | Load `ai4bharat/indic-conformer-*` via `transformers` |
| `scripts/tts_parler.py` | Load `ai4bharat/indic-parler-tts` with selectable professional and warm voice profiles |
| `scripts/llm_gemini.py` | `google-generativeai` client + `GEMINI_API_KEY` |
| `scripts/bluetooth_audio.py` | Already functional against `sounddevice` — just needs a paired device |

## Guardrails are the point, not an afterthought

`guardrails/policy.yaml` is the single source of truth for hard rules — it
feeds **both** the LLM system prompt (`llm_gemini.py`) and the automated
grader (`test_harness.py`). Risk tiers (`personas/risk_delinquency.yaml`) can
only change *posture and permitted talking points*, never the hard rules —
`test_harness.py` is where you'd add a check that asserts this if you extend
the tier system.

`orchestrator.py --live` will not start unless the guardrail harness passes
in that run — this is intentional friction, not a bug.

## Extending the persona/risk test suite

Add new adversarial or vulnerable-caller scenarios in
`guardrails/test_cases.yaml`, referencing a persona from
`personas/persona_config.yaml`. Add any new keyword/pattern checks to the
`_check()` function in `guardrails/test_harness.py`, or route ambiguous ones
to an LLM-as-judge via the `judge_with_llm()` TODO hook.
