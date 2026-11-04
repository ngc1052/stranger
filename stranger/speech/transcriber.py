"""Local speech recognition with faster-whisper."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import WHISPER_SAMPLE_RATE, SpeechConfig


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Transcriber:
    """Wraps a faster-whisper model. Loading is lazy and happens once."""

    def __init__(self, config: SpeechConfig | None = None):
        self.config = config or SpeechConfig()
        self._model = None

    def load(self) -> None:
        """Load (and on first use, download) the model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.compute_device,
                compute_type=self.config.compute_type,
            )

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe 16 kHz mono float32 audio."""
        self.load()
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=True,
        )
        collected = [Segment(s.start, s.end, s.text.strip()) for s in segments]  # generator runs here
        return Transcript(
            text=" ".join(s.text for s in collected if s.text).strip(),
            segments=collected,
            language=info.language,
            audio_seconds=len(audio) / WHISPER_SAMPLE_RATE,
            elapsed_seconds=time.perf_counter() - started,
        )
