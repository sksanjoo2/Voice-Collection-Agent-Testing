# Voice Collection Agent

> A safety-first, multilingual voice agent scaffold for EMI and loan repayment outreach.

The project combines speech recognition, a policy-aware conversational model, and speech synthesis into one testable pipeline:

```text
Microphone / audio input
        |
        v
AI4Bharat Indic Conformer (STT)
        |
        v
Gemini or local Ollama/Qwen (LLM) + policy guardrails
        |
        v
AI4Bharat Indic Parler-TTS
        |
        v
Speaker / audio output
```

It is designed to run safely without credentials or model downloads first. The default demo, terminal chat, browser UI, and guardrail suite all support deterministic offline development.

> [!IMPORTANT]
> This repository is a development scaffold, not a production-ready debt-collection system. Before real-world use, obtain legal and compliance review for every applicable jurisdiction, implement consent and calling-hour controls, protect customer data, and require human oversight.

## Highlights

- Offline-first demo with no API key, GPU, model download, or phone connection
- Browser interface for simulated calls and model selection
- Gemini and local Ollama/Qwen conversation backends
- AI4Bharat Indic Conformer speech recognition
- AI4Bharat Indic Parler-TTS with a calm, neutral voice profile
- Persona and delinquency-risk simulation for adversarial testing
- A non-bypassable guardrail check before the live CLI starts
- Per-session structured conversation records for local development

## Project status

| Component | Offline behavior | Optional live backend |
| --- | --- | --- |
| Conversation | Deterministic policy simulator | Gemini or Ollama/Qwen |
| Speech to text | Stub transcript | Indic Conformer |
| Text to speech | Silent test audio | Indic Parler-TTS |
| Audio I/O | Text/browser simulation | Paired `sounddevice` input/output device |
| Safety checks | Fully runnable | Required before live CLI startup |

## Quick start

### 1. Create an environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The full requirements include the ML and audio stack. For lightweight offline use, only PyYAML is required:

```bash
python -m pip install 'pyyaml>=6.0'
```

### 2. Run locally

```bash
# Deterministic end-to-end demonstration
python app.py

# Interactive terminal conversation
python app.py chat --risk-tier medium_risk

# Safety and compliance-oriented test suite
python app.py test

# Browser UI
python app.py ui
```

Open <http://127.0.0.1:8080> after starting the UI.

## Optional model and service setup

### Local speech models

The setup script creates `.venv`, installs the model dependencies, and downloads the configured Hugging Face repositories into `models/`:

```bash
./scripts/setup_huggingface.sh
```

Useful variants:

```bash
./scripts/setup_huggingface.sh --packages-only
./scripts/setup_huggingface.sh --models-only
```

Downloaded weights are intentionally excluded from Git because they occupy several gigabytes. Set `HF_TOKEN` if a selected model requires authentication.

### Gemini

Set the API key in the server environment, then select Gemini in the browser UI:

```bash
export GEMINI_API_KEY="your-api-key"
python app.py ui
```

Never commit API keys or place them in browser-facing source files.

### Ollama/Qwen

Start Ollama, make the desired Qwen model available, and select the Ollama backend in the UI. The default endpoint is `http://127.0.0.1:11434` and the default model name is `qwen2.5:7b`.

## Live audio

Only attempt live audio after the guardrail suite passes and the required models and audio device are configured:

```bash
python scripts/orchestrator.py \
  --risk-tier medium_risk \
  --live \
  --bt-device "your device name"
```

The `--bt-device` value is a substring matched against available audio devices. The live command runs the guardrail harness first and refuses to continue if it fails.

## Guardrail architecture

[`guardrails/policy.yaml`](guardrails/policy.yaml) is the source of truth for hard rules. It feeds both the LLM system prompt and the automated test grader.

Risk tiers in [`personas/risk_delinquency.yaml`](personas/risk_delinquency.yaml) may alter posture and permitted talking points, but they must never weaken the hard rules. Examples include:

- No threats, humiliation, harassment, or impersonation
- No disclosure of debt information to an unverified third party
- Immediate handling of opt-out, dispute, wrong-number, and human-agent requests
- De-escalation and human handoff for hardship, distress, or self-harm signals
- Calling-hour and identity-disclosure requirements

Add scenarios to [`guardrails/test_cases.yaml`](guardrails/test_cases.yaml), personas to [`personas/persona_config.yaml`](personas/persona_config.yaml), and deterministic checks to `guardrails/test_harness.py`.

## Data and privacy

Browser sessions are written locally to:

```text
conversations/<session-id>/conversation.json
```

Each record contains session metadata and ordered turns, including selected backend/model identifiers and transcripts. Raw microphone audio and API keys are not written by the current implementation. Conversation data is excluded from Git by default; treat it as sensitive customer data and define retention, access-control, encryption, and deletion policies before deployment.

## Repository layout

```text
.
├── app.py                         # Supported local CLI entry point
├── guardrails/
│   ├── policy.yaml                # Hard safety rules
│   ├── test_cases.yaml            # Adversarial scenarios
│   └── test_harness.py            # Deterministic grader
├── personas/
│   ├── persona_config.yaml        # Test personas
│   └── risk_delinquency.yaml      # Risk-tier posture configuration
├── scripts/
│   ├── bluetooth_audio.py         # Audio capture and playback
│   ├── llm_gemini.py              # Gemini + offline policy agent
│   ├── llm_ollama.py              # Local Ollama/Qwen backend
│   ├── orchestrator.py             # Simulated/live call loop
│   ├── setup_huggingface.sh       # Dependency and model setup
│   ├── stt_indic_conformer.py     # Speech recognition wrapper
│   └── tts_parler.py              # Speech synthesis wrapper
├── ui/                            # Dependency-light local web app
├── requirements.txt
└── SKILL.md                       # Agent skill definition
```

## Development checks

Run these before publishing changes:

```bash
python app.py
python app.py test
python -m compileall -q app.py scripts guardrails ui
```

## Extending the project

When changing prompts, personas, or risk-tier behavior:

1. Keep hard policy in `guardrails/policy.yaml`.
2. Add an adversarial test that demonstrates the expected behavior.
3. Run the full guardrail suite.
4. Review generated wording with compliance and domain specialists.
5. Test live integrations only after the offline behavior is accepted.

## License

No license has been selected yet. Until a license file is added, the project remains under the copyright holder's default rights.
