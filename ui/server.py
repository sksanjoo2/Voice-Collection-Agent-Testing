"""Dependency-free local web UI server for the voice collection agent."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import io
import json
import re
import sys
import threading
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from llm_gemini import GeminiFlashAgent  # noqa: E402
from llm_ollama import OllamaQwenAgent  # noqa: E402
from stt_indic_conformer import IndicConformerSTT  # noqa: E402
from tts_parler import ParlerTTS  # noqa: E402
import numpy as np  # noqa: E402

SESSIONS: dict[str, GeminiFlashAgent] = {}
SESSION_META: dict[str, dict] = {}
TTS_MODEL: ParlerTTS | None = None
STT_MODEL: IndicConformerSTT | None = None
MODEL_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
CONVERSATIONS_DIR = ROOT / "conversations"
SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def _session_dir(session_id: str) -> Path:
    if not SAFE_SESSION_ID.fullmatch(session_id):
        raise ValueError("Invalid session ID")
    return CONVERSATIONS_DIR / session_id


def _write_conversation(session_id: str, metadata: dict, turns: list | None = None) -> None:
    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    public_metadata = {key: value for key, value in metadata.items() if key != "turn"}
    conversation_path = session_dir / "conversation.json"
    if turns is None and conversation_path.exists():
        turns = json.loads(conversation_path.read_text(encoding="utf-8")).get("turns", [])
    payload = {"session": public_metadata, "turns": turns or []}
    conversation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
RISK_TIERS = {"low_risk", "medium_risk", "high_risk", "hardship_flagged"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid JSON") from exc

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/api/health":
            self._json(200, {"ok": True, "mode": "offline"})
        else:
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        try:
            body, path = self._body(), urlparse(self.path).path
            if path == "/api/call/start":
                self._start(body)
            elif path == "/api/call/turn":
                self._turn(body)
            elif path == "/api/call/end":
                session_id = str(body.get("session_id", ""))
                SESSIONS.pop(session_id, None)
                metadata = SESSION_META.pop(session_id, None)
                if metadata is not None:
                    metadata["status"] = "ended"
                    metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
                    with LOG_LOCK:
                        _write_conversation(session_id, metadata)
                self._json(200, {"ok": True})
            elif path == "/api/tts":
                self._tts(body)
            elif path == "/api/stt":
                self._stt(body)
            else:
                self._json(404, {"error": "Unknown endpoint"})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"Server error: {exc}"})

    def _start(self, body: dict) -> None:
        llm = body.get("llm", "offline")
        if llm not in {"offline", "gemini", "ollama"}:
            raise ValueError("Unknown LLM selection")
        tier, sid = body.get("risk_tier", "medium_risk"), str(body.get("session_id", ""))
        if tier not in RISK_TIERS or not SAFE_SESSION_ID.fullmatch(sid):
            raise ValueError("Invalid call configuration")
        language = str(body.get("language", "hi"))
        if llm == "ollama":
            agent = OllamaQwenAgent(
                risk_tier=tier,
                model_id=str(body.get("ollama_model", "qwen2.5:7b")).strip(),
                endpoint=str(body.get("ollama_endpoint", "http://127.0.0.1:11434")).strip(),
                language=language,
            )
            agent.load()
        else:
            agent = GeminiFlashAgent(risk_tier=tier, language=language)
        if llm == "gemini":
            agent.load(api_key=str(body.get("api_key", "")).strip() or None)
        SESSIONS[sid] = agent
        SESSION_META[sid] = {
            "turn": 0,
            "llm": (
                "gemini-3.5-flash" if llm == "gemini" else
                str(body.get("ollama_model", "qwen2.5:7b")) if llm == "ollama" else "offline"
            ),
            "llm_backend": llm,
            "stt": str(body.get("stt", "unknown")),
            "stt_model": (
                "ai4bharat/indic-conformer-600m-multilingual"
                if body.get("stt") == "indic" else str(body.get("stt", "unknown"))
            ),
            "tts": str(body.get("tts", "unknown")),
            "tts_model": (
                "ai4bharat/indic-parler-tts"
                if body.get("tts") == "parler" else str(body.get("tts", "unknown"))
            ),
            "language": language,
            "risk_tier": tier,
            "mode": str(body.get("mode", "unknown")),
            "status": "active",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        with LOG_LOCK:
            _write_conversation(sid, SESSION_META[sid], [])
        self._json(200, {"ok": True, "session_id": sid})

    def _turn(self, body: dict) -> None:
        session_id = str(body.get("session_id", ""))
        agent = SESSIONS.get(session_id)
        if agent is None:
            raise ValueError("Start a call first")
        text = str(body.get("text", "")).strip()
        if not text or len(text) > 4000:
            raise ValueError("Enter a message up to 4,000 characters")
        turn = agent.respond(text)
        meta = SESSION_META.get(session_id, {})
        meta["turn"] = int(meta.get("turn", 0)) + 1
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "turn": meta["turn"],
            "llm": meta.get("llm"),
            "stt": {
                "backend": meta.get("stt"),
                "model": meta.get("stt_model"),
                "language": meta.get("language"),
                "transcript": text,
                "sample_rate": 16000 if meta.get("stt") == "indic" else None,
            },
            "tts": {
                "backend": meta.get("tts"),
                "model": meta.get("tts_model"),
                "text": turn.text,
                "sample_rate": 44100 if meta.get("tts") == "parler" else None,
            },
            "language": meta.get("language"),
            "risk_tier": meta.get("risk_tier"),
            "mode": meta.get("mode"),
            "debtor": text,
            "agent": turn.text,
            "escalated": turn.should_escalate_to_human,
            "ended": turn.should_end_call,
        }
        with LOG_LOCK:
            conversation_path = _session_dir(session_id) / "conversation.json"
            payload = json.loads(conversation_path.read_text(encoding="utf-8"))
            turns = payload.get("turns", [])
            turns.append(record)
            _write_conversation(session_id, meta, turns)
        self._json(200, {"text": turn.text, "escalate": turn.should_escalate_to_human,
                         "ended": turn.should_end_call})

    def _tts(self, body: dict) -> None:
        global TTS_MODEL
        text = str(body.get("text", "")).strip()
        if not text or len(text) > 1000:
            raise ValueError("TTS text must be between 1 and 1,000 characters")
        with MODEL_LOCK:
            if TTS_MODEL is None:
                TTS_MODEL = ParlerTTS()
                TTS_MODEL.load()
            result = TTS_MODEL.synthesize(text)
        pcm16 = (result.pcm.clip(-1, 1) * 32767).astype("<i2").tobytes()
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(result.sample_rate)
            wav.writeframes(pcm16)
        self._json(200, {"audio": base64.b64encode(output.getvalue()).decode("ascii"),
                         "sample_rate": result.sample_rate})

    def _stt(self, body: dict) -> None:
        global STT_MODEL
        try:
            raw = base64.b64decode(str(body.get("audio", "")), validate=True)
            with wave.open(io.BytesIO(raw), "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
            if width != 2:
                raise ValueError("Expected 16-bit WAV audio")
            pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1)
        except (ValueError, wave.Error) as exc:
            raise ValueError("Invalid WAV recording") from exc
        if pcm.size > sample_rate * 30:
            raise ValueError("Recording must be 30 seconds or shorter")
        language = str(body.get("language", "hi"))
        with MODEL_LOCK:
            if STT_MODEL is None:
                STT_MODEL = IndicConformerSTT()
                STT_MODEL.load()
            result = STT_MODEL.transcribe(pcm, sample_rate, language)
        self._json(200, {"text": result.text, "language": result.language})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Voice agent UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
