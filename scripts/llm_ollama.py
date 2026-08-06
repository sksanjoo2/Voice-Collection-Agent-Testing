"""Ollama/Qwen chat backend with deterministic collection guardrails."""
from __future__ import annotations

import json
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from llm_gemini import GeminiFlashAgent, LLMTurn


class OllamaQwenAgent(GeminiFlashAgent):
    def __init__(self, risk_tier: str, model_id: str, endpoint: str, language: str = "en"):
        super().__init__(risk_tier=risk_tier, model_id=model_id, language=language)
        self.endpoint = endpoint.rstrip("/")

    def load(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError("Ollama endpoint must be an http(s) URL.")
        request = Request(f"{self.endpoint}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=8) as response:
                models = json.load(response).get("models", [])
        except Exception as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {self.endpoint}: {exc}") from exc
        names = {item.get("name") for item in models} | {item.get("model") for item in models}
        if self.model_id not in names:
            available = ", ".join(sorted(name for name in names if name)) or "none"
            raise RuntimeError(f"Ollama model '{self.model_id}' is unavailable. Installed: {available}")
        self._client = True

    def respond(self, debtor_utterance: str) -> LLMTurn:
        if self._client is None:
            raise RuntimeError("Call load() before using Ollama.")
        if self._ended:
            return LLMTurn("This call has already ended.", should_end_call=True)

        shadow = GeminiFlashAgent(self.risk_tier, language=self.language)
        shadow._turn_number = self._turn_number
        safety_turn = shadow._respond_offline(debtor_utterance)
        if safety_turn.should_end_call or safety_turn.should_escalate_to_human:
            self._turn_number = shadow._turn_number
            self._ended = True
            self.history.extend(shadow.history)
            return safety_turn

        messages = [{"role": "system", "content": self.build_system_prompt()}]
        messages.extend(
            {"role": "assistant" if item["role"] == "assistant" else "user", "content": item["text"]}
            for item in self.history
        )
        messages.append({"role": "user", "content": debtor_utterance})
        payload = json.dumps({
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 180},
        }).encode()
        request = Request(
            f"{self.endpoint}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        text = str(result.get("message", {}).get("content", "")).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response.")
        self._turn_number += 1
        if self._turn_number == 1:
            company = self._policy["company_name"]
            if company not in text:
                text = self._disclosure(company) + text
        self.history.extend([
            {"role": "user", "text": debtor_utterance},
            {"role": "assistant", "text": text},
        ])
        return LLMTurn(text=text)
