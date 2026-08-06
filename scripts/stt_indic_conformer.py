"""
STT stage: AI4Bharat Indic Conformer.

Indic Conformer (https://github.com/AI4Bharat/IndicConformerASR /
ai4bharat/indic-conformer on HuggingFace) supports multiple Indian languages
plus code-mixed English. This wrapper exposes a single `transcribe()` call so
the orchestrator doesn't need to know model internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class STTResult:
    text: str
    language: str
    confidence: float


class IndicConformerSTT:
    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual",
                 device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load AI4Bharat's remote-code model on CUDA when available."""
        import torch
        from transformers import AutoModel

        self.device = "cuda" if self.device == "auto" and torch.cuda.is_available() else (
            "cpu" if self.device == "auto" else self.device
        )
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device).eval()

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16_000,
                    language_hint: Optional[str] = None) -> STTResult:
        """
        Transcribes a mono float32 PCM utterance.

        `language_hint` (e.g. 'hi', 'ta', 'en') can be passed if the call
        flow already knows the debtor's preferred language; otherwise the
        model's language-ID head should be used.
        """
        if self._model is None:
            # Stub path for local development / guardrail testing without GPU.
            return STTResult(text="[stt-stub: no model loaded]", language=language_hint or "unknown", confidence=0.0)

        import torch
        import torchaudio

        waveform = torch.as_tensor(pcm, dtype=torch.float32).flatten().unsqueeze(0)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        language = language_hint or "hi"
        with torch.inference_mode():
            text = self._model(waveform.to(self.device), language, "ctc")
        if isinstance(text, (list, tuple)):
            text = text[0]
        return STTResult(text=str(text).strip(), language=language, confidence=1.0)
