"""
TTS stage: Parler-TTS.

Parler-TTS (https://github.com/huggingface/parler-tts) is prompt-controllable:
voice style, pace, and tone are steered by a natural-language "description"
alongside the text to speak. We use that description slot to keep the voice
calm and neutral regardless of persona/risk tier -- the *words* the LLM
chooses may vary by risk tier, the *delivery* should not become aggressive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_VOICE_DESCRIPTION = (
    "A calm, clear, professional male voice speaking at a measured pace, "
    "neutral and courteous tone, minimal background noise."
)

@dataclass
class TTSResult:
    pcm: np.ndarray
    sample_rate: int


class ParlerTTS:
    def __init__(self, model_id: str = "ai4bharat/indic-parler-tts", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load Indic Parler-TTS on CUDA when available."""
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.device = "cuda" if self.device == "auto" and torch.cuda.is_available() else (
            "cpu" if self.device == "auto" else self.device
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def synthesize(self, text: str, voice_description: str = DEFAULT_VOICE_DESCRIPTION) -> TTSResult:
        if self._model is None:
            # Stub path: return silence so the pipeline is testable end-to-end
            # without GPU/model weights present.
            silence = np.zeros(int(24_000 * 0.5), dtype=np.float32)
            return TTSResult(pcm=silence, sample_rate=24_000)

        import torch

        description_ids = self._tokenizer(voice_description, return_tensors="pt").input_ids.to(self.device)
        prompt_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        with torch.inference_mode():
            generation = self._model.generate(
                input_ids=description_ids, prompt_input_ids=prompt_ids
            )
        pcm = generation.detach().float().cpu().numpy().squeeze()
        return TTSResult(pcm=pcm.astype(np.float32), sample_rate=self._model.config.sampling_rate)
