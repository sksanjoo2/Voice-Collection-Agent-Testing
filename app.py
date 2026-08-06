"""Runnable command-line entry point for the voice collection agent.

The default command is a deterministic, offline demonstration. It deliberately
does not connect to a phone line or require model credentials.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "guardrails"))

from llm_gemini import GeminiFlashAgent  # noqa: E402
from test_harness import main as run_guardrails  # noqa: E402


RISK_TIERS = ("low_risk", "medium_risk", "high_risk", "hardship_flagged")
DEMO_LINES = (
    "Hello, who is calling?",
    "I lost my job and cannot pay right now.",
)


def run_demo(risk_tier: str) -> int:
    """Run a safe, deterministic conversation without external services."""
    agent = GeminiFlashAgent(risk_tier=risk_tier)
    print(f"Offline voice-agent demo (risk tier: {risk_tier})")
    print("No call, model download, or API request will be made.\n")

    for debtor_line in DEMO_LINES:
        print(f"Debtor: {debtor_line}")
        turn = agent.respond(debtor_line)
        print(f"Agent:  {turn.text}\n")
        if turn.should_escalate_to_human:
            print("Outcome: handed off to a human support specialist.")
            return 0
        if turn.should_end_call:
            print("Outcome: call ended safely.")
            return 0

    print("Outcome: demo completed.")
    return 0


def run_chat(risk_tier: str) -> int:
    """Run the offline agent interactively in the terminal."""
    agent = GeminiFlashAgent(risk_tier=risk_tier)
    print("Offline interactive mode. Type /quit to leave.")
    while True:
        try:
            debtor_line = input("debtor> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCall ended.")
            return 0
        if debtor_line.casefold() in {"/quit", "/exit"}:
            print("Call ended.")
            return 0
        if not debtor_line:
            continue

        turn = agent.respond(debtor_line)
        print(f"agent> {turn.text}")
        if turn.should_escalate_to_human:
            print("Escalating to a human agent.")
            return 0
        if turn.should_end_call:
            print("Call ended by the agent.")
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the voice collection agent safely")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("demo", "chat", "test", "ui"),
        default="demo",
        help="demo (default), interactive chat, guardrail tests, or web UI",
    )
    parser.add_argument("--risk-tier", choices=RISK_TIERS, default="medium_risk")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "ui":
        venv_python = ROOT / ".venv" / "bin" / "python"
        python = str(venv_python) if venv_python.exists() else sys.executable
        return subprocess.call([python, str(ROOT / "ui" / "server.py")])
    if args.command == "test":
        return run_guardrails()
    if args.command == "chat":
        return run_chat(args.risk_tier)
    return run_demo(args.risk_tier)


if __name__ == "__main__":
    raise SystemExit(main())
