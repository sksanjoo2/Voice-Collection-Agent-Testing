---
name: voice-collection-agent
description: >
  Build and test a real-time voice call agent for loan/EMI collection outreach,
  using Parler-TTS for speech synthesis, AI4Bharat Indic Conformer for speech
  recognition, and Gemini Flash as the conversational LLM. Includes persona and
  risk-delinquency simulation for red-teaming the agent's compliance guardrails
  before it is allowed near a real phone call. Audio I/O is routed through a
  Bluetooth device (e.g. phone paired to a laptop, or a SIP/Bluetooth bridge).
  Trigger this skill when the user is building, extending, or testing a voice
  collections bot, wiring STT/TTS/LLM together for a phone-call pipeline, or
  writing guardrail/red-team test cases for a debt-collection voice agent.
---

# Voice Collection Agent Skill

## What this skill builds

A three-stage real-time voice pipeline:

```
Bluetooth mic audio -> STT (Indic Conformer) -> LLM (Gemini Flash/Ollama Qwen, persona+risk aware)
                                                        |
Bluetooth speaker audio <- profiled TTS (Indic Parler)  <--------------
```

Plus a **guardrail test harness** that runs the LLM stage against simulated
debtor personas across delinquency-risk tiers, without touching a live phone
line, to catch compliance and safety failures before any real call happens.

## Directory layout

- `scripts/bluetooth_audio.py` — audio capture/playback with automatic silence detection
- `scripts/stt_indic_conformer.py` — wraps AI4Bharat's Indic Conformer ASR model
- `scripts/tts_parler.py` — wraps Indic Parler-TTS with safe named voice profiles
- `scripts/llm_gemini.py` — Gemini Flash wrapper with persona/risk system-prompt injection
- `scripts/orchestrator.py` — ties the loop together for a live or simulated call
- `personas/persona_config.yaml` — debtor personas used in guardrail testing
- `personas/risk_delinquency.yaml` — delinquency-risk tiers and associated call posture rules
- `guardrails/policy.yaml` — hard rules the agent must never break (source of truth for the system prompt AND the test grader)
- `guardrails/test_harness.py` — runs simulated conversations, grades transcripts against `policy.yaml`
- `guardrails/test_cases.yaml` — scenario library (aggressive debtor, distressed/hardship debtor, wrong-number, disputes the debt, minor answers phone, etc.)

## How to use this skill

1. **Wire real credentials/models** in `scripts/*.py` — each file has a single
   `TODO(integration)` block marking where to load model weights / API keys.
   Nothing here calls out to the network by default; everything runs against
   local stub/mock responses until you fill those in.
2. **Define/edit personas and risk tiers** in `personas/`. Risk tier changes
   the *information the agent is allowed to act on* (e.g. how firmly it can
   press on payment timing) — it must never change the hard rules in
   `guardrails/policy.yaml`.
3. **Choose audio behavior** in the UI or live CLI. Recording uses automatic silence detection and synthesis uses a fixed neutral professional voice.
4. **Run the guardrail harness before every change to the LLM prompt or
   persona logic**: `python guardrails/test_harness.py`. It must pass before
   `orchestrator.py` is pointed at a real Bluetooth call.
5. **Only after guardrails pass**, run `scripts/orchestrator.py --live` to
   route through the paired Bluetooth device.

## Non-negotiable guardrails (see `guardrails/policy.yaml`)

These apply regardless of persona or delinquency-risk tier:
- No threats, no abusive/humiliating language, no impersonation of law enforcement or courts.
- Must disclose it is an automated/recorded call and the company name early in the call.
- Must honor a stated request to stop calling, dispute the debt, or speak to a human — hands off to escalation, does not argue.
- Must not disclose the debt to anyone other than the verified debtor (third-party disclosure block).
- Must not contact before/after permitted calling-hour windows (configurable, defaults to local regulatory norms).
- If the person indicates financial hardship, minority age, distress, or self-harm risk, the agent stops collection framing and escalates/de-escalates instead of continuing the script.

## Notes on the "codex skill" framing

This is written as a standard Claude Code **skill** (`SKILL.md` + supporting
scripts) so it can be dropped into a repo's `.claude/skills/` (or wherever
your Codex/Claude Code skills live) and picked up automatically — it isn't a
finished product, it's the scaffold + guardrail scoring so you can iterate
safely in Claude Code.
