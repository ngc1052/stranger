"""Configuration for microphone capture and transcription."""

from __future__ import annotations

from dataclasses import dataclass

# Whisper models are trained on 16 kHz mono audio.
WHISPER_SAMPLE_RATE = 16_000


@dataclass
class SpeechConfig:
    """Tunable settings for the speech pipeline.

    model_size:    faster-whisper model name: tiny, base, small, medium,
                   large-v3, or any *.en variant. Larger is more accurate and
                   slower. Downloaded to the Hugging Face cache on first use.
    device:        microphone to use: None for the system default, an integer
                   index, or a case-insensitive substring of the device name.
    chunk_seconds: length of each audio chunk handed to the transcriber.
    compute_device: "cpu", "cuda", or "auto" for the Whisper model.
    compute_type:  numeric precision for the model, e.g. "int8" (fast on CPU),
                   "float16" (GPU), "float32".
    language:      ISO language code, or None to let Whisper detect it.
    beam_size:     decoding beam width; 1 is fastest, 5 is Whisper's default.
    """

    model_size: str = "small"
    device: int | str | None = None
    chunk_seconds: float = 5.0
    compute_device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "en"
    beam_size: int = 1
