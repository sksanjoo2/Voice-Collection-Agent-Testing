"""
Orchestrator: the call loop.

  Bluetooth mic -> Indic Conformer STT -> Gemini Flash (persona/risk aware)
      -> escalation check -> Parler TTS -> Bluetooth speaker

Usage:
    python orchestrator.py --risk-tier medium_risk           # simulated (no audio, stdin/stdout)
    python orchestrator.py --risk-tier medium_risk --live     # real Bluetooth call
    python orchestrator.py --guardrail-check                  # refuses to run --live unless guardrails/test_harness.py passed

Safety gate: --live refuses to start unless the guardrail harness has been
run and passed in this environment. This is intentionally not something you
can bypass with a flag -- fix the guardrail failures instead.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bluetooth_audio import BluetoothAudioIO, BluetoothAudioConfig  # noqa: E402
from stt_indic_conformer import IndicConformerSTT  # noqa: E402
from tts_parler import ParlerTTS  # noqa: E402
from llm_gemini import GeminiFlashAgent  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def guardrails_pass() -> bool:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "guardrails" / "test_harness.py")],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def run_simulated(risk_tier: str) -> None:
    """Text-in/text-out loop for fast local iteration -- no audio, no models required to load."""
    agent = GeminiFlashAgent(risk_tier=risk_tier)
    print(f"[simulated call | risk_tier={risk_tier}] Type debtor lines, Ctrl+C to end.")
    print(agent.build_system_prompt())
    try:
        while True:
            line = input("debtor> ")
            turn = agent.respond(line)
            print(f"agent> {turn.text}")
            if turn.should_escalate_to_human:
                print("[orchestrator] Escalation trigger fired -- handing off to human agent.")
                break
            if turn.should_end_call:
                print("[orchestrator] Call ended by agent.")
                break
    except (KeyboardInterrupt, EOFError):
        print("\n[orchestrator] Call ended.")


def run_live(risk_tier: str, bt_device_substring: str) -> None:
    audio = BluetoothAudioIO(BluetoothAudioConfig(device_name_substring=bt_device_substring))
    audio.connect()

    stt = IndicConformerSTT()
    stt.load()  # TODO(integration): will raise until scripts/stt_indic_conformer.py is wired up

    tts = ParlerTTS()
    tts.load()  # TODO(integration): will raise until scripts/tts_parler.py is wired up

    agent = GeminiFlashAgent(risk_tier=risk_tier)
    agent.load()  # TODO(integration): will raise until scripts/llm_gemini.py is wired up

    print(f"[live call | risk_tier={risk_tier} | bt_device~='{bt_device_substring}'] Listening...")
    while True:
        pcm = audio.record_utterance()
        if pcm.size == 0:
            continue
        stt_result = stt.transcribe(pcm)
        print(f"debtor> {stt_result.text}")

        turn = agent.respond(stt_result.text)
        print(f"agent> {turn.text}")

        tts_result = tts.synthesize(turn.text)
        audio.play(tts_result.pcm, sample_rate=tts_result.sample_rate)

        if turn.should_escalate_to_human or turn.should_end_call:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice collection agent orchestrator")
    parser.add_argument("--risk-tier", default="medium_risk",
                         choices=["low_risk", "medium_risk", "high_risk", "hardship_flagged"])
    parser.add_argument("--live", action="store_true", help="Route through real Bluetooth audio")
    parser.add_argument("--bt-device", default="headset", help="Substring to match the paired Bluetooth device name")
    args = parser.parse_args()

    if args.live:
        print("[orchestrator] --live requested: running guardrail harness first (non-bypassable)...")
        if not guardrails_pass():
            print("[orchestrator] REFUSING to start live call: guardrail checks failed.", file=sys.stderr)
            sys.exit(1)
        run_live(args.risk_tier, args.bt_device)
    else:
        run_simulated(args.risk_tier)


if __name__ == "__main__":
    main()
