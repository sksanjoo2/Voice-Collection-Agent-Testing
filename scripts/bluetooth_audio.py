"""
Bluetooth audio I/O for the voice collection agent.

Captures mic audio from a paired Bluetooth device (e.g. a phone acting as a
Bluetooth headset/mic to a laptop, or a Bluetooth SIP dongle) and plays back
synthesized speech to the same device.

This module intentionally has no model logic in it -- it's a thin, swappable
audio transport so STT/TTS/LLM stages can be tested independently of any
particular OS Bluetooth stack.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):  # pragma: no cover - optional dependency / missing system lib (PortAudio)
    sd = None


SAMPLE_RATE = 16_000  # Indic Conformer expects 16kHz mono
CHANNELS = 1
FRAME_MS = 30  # 30ms frames, common for VAD-driven streaming ASR
SILENCE_THRESHOLD = 0.010
TRAILING_SILENCE_MS = 700


@dataclass
class BluetoothAudioConfig:
    device_name_substring: str = "headset"  # match against sd.query_devices()
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS


class BluetoothAudioIO:
    """
    Wraps capture + playback against a specific paired Bluetooth device.

    Usage:
        io = BluetoothAudioIO(BluetoothAudioConfig())
        io.connect()
        utterance = io.record_utterance()   # blocks until silence detected
        io.play(pcm_bytes, sample_rate=24000)
    """

    def __init__(self, config: BluetoothAudioConfig):
        self.config = config
        self._input_device_index: Optional[int] = None
        self._output_device_index: Optional[int] = None

    def connect(self) -> None:
        """Resolve the Bluetooth device by name and validate it's available."""
        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. `pip install sounddevice`."
            )
        devices = sd.query_devices()
        match_name = self.config.device_name_substring.lower()
        for idx, dev in enumerate(devices):
            name = dev["name"].lower()
            if match_name in name:
                if dev["max_input_channels"] > 0 and self._input_device_index is None:
                    self._input_device_index = idx
                if dev["max_output_channels"] > 0 and self._output_device_index is None:
                    self._output_device_index = idx
        if self._input_device_index is None or self._output_device_index is None:
            raise RuntimeError(
                f"Could not find a paired Bluetooth device matching "
                f"'{self.config.device_name_substring}'. Pair the device in your "
                f"OS Bluetooth settings first, then re-run. "
                f"Available devices: {[d['name'] for d in devices]}"
            )

    def record_utterance(self, max_seconds: float = 30.0) -> np.ndarray:
        """
        Records audio until trailing silence is detected or `max_seconds`
        elapses. Returns mono float32 PCM at sample_rate.

        The built-in detector is dependency-free RMS/energy VAD. For difficult
        production audio, this transport can be swapped for Silero or WebRTC VAD.
        """
        if self._input_device_index is None:
            raise RuntimeError("Call connect() before recording.")

        frame_len = int(self.config.sample_rate * FRAME_MS / 1000)
        silence_frames_needed = int(TRAILING_SILENCE_MS / FRAME_MS)
        silent_run = 0
        chunks: list[np.ndarray] = []
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_info, status):
            q.put(indata.copy())

        start = time.time()
        with sd.InputStream(
            device=self._input_device_index,
            channels=self.config.channels,
            samplerate=self.config.sample_rate,
            blocksize=frame_len,
            callback=_callback,
        ):
            while time.time() - start < max_seconds:
                frame = q.get()
                chunks.append(frame)
                energy = float(np.abs(frame).mean())
                if energy < SILENCE_THRESHOLD:
                    silent_run += 1
                else:
                    silent_run = 0
                if silent_run >= silence_frames_needed and chunks:
                    break

        return np.concatenate(chunks, axis=0).flatten() if chunks else np.array([])

    def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Plays synthesized speech out through the Bluetooth output device."""
        if self._output_device_index is None:
            raise RuntimeError("Call connect() before playback.")
        sd.play(pcm, samplerate=sample_rate, device=self._output_device_index, blocking=True)
