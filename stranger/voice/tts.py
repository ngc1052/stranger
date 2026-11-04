"""Text-to-speech backends.

All backends return `SpokenAudio`: mono float32 samples plus a sample rate,
ready for `AudioPlayer`. Pick one with `make_tts(VoiceConfig(...))`:

  elevenlabs  ElevenLabs API. Needs ELEVENLABS_API_KEY. Voice is a voice id.
  openai      OpenAI speech API. Needs OPENAI_API_KEY. Voice is a voice name.
  piper       Local Piper neural voice, offline, no key (pip install piper-tts + a voice model).
  espeak      Local espeak-ng, no key, robotic but always available.
  auto        First of the above whose key (or model/binary) is present.

API calls use the standard library only, so no extra SDKs are required.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

ELEVENLABS_DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"   # "George": a warm, older British voice
OPENAI_DEFAULT_VOICE = "onyx"
ESPEAK_DEFAULT_VOICE = "en-gb"


@dataclass
class VoiceConfig:
    backend: str = "auto"            # elevenlabs | openai | espeak | auto
    voice: str | None = None         # backend-specific voice id/name; None = backend default
    output_device: int | str | None = None
    speed: float = 1.0               # 1.0 = normal; espeak and openai honour this
    elevenlabs_model: str = "eleven_multilingual_v2"
    openai_model: str = "gpt-4o-mini-tts"
    timeout_seconds: float = 30.0


@dataclass
class SpokenAudio:
    samples: np.ndarray              # float32 mono, -1..1
    sample_rate: int
    backend: str
    elapsed_seconds: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


class TextToSpeech(Protocol):
    name: str
    def synthesize(self, text: str) -> SpokenAudio: ...


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _post(url: str, headers: dict, body: dict, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"TTS request failed ({e.code}): {detail}") from e


class ElevenLabsTTS:
    name = "elevenlabs"
    SAMPLE_RATE = 24_000

    def __init__(self, config: VoiceConfig):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        self.voice = config.voice or os.environ.get("ELEVENLABS_VOICE_ID") or ELEVENLABS_DEFAULT_VOICE
        self.config = config

    def synthesize(self, text: str) -> SpokenAudio:
        t0 = time.perf_counter()
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice}?output_format=pcm_{self.SAMPLE_RATE}"
        raw = _post(url, {"xi-api-key": self.api_key},
                    {"text": text, "model_id": self.config.elevenlabs_model}, self.config.timeout_seconds)
        return SpokenAudio(pcm16_to_float(raw), self.SAMPLE_RATE, self.name, time.perf_counter() - t0)


class OpenAITTS:
    name = "openai"
    SAMPLE_RATE = 24_000

    def __init__(self, config: VoiceConfig):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.voice = config.voice or os.environ.get("OPENAI_TTS_VOICE") or OPENAI_DEFAULT_VOICE
        self.config = config

    def synthesize(self, text: str) -> SpokenAudio:
        t0 = time.perf_counter()
        raw = _post("https://api.openai.com/v1/audio/speech", {"Authorization": f"Bearer {self.api_key}"},
                    {"model": self.config.openai_model, "input": text, "voice": self.voice,
                     "response_format": "pcm", "speed": self.config.speed}, self.config.timeout_seconds)
        return SpokenAudio(pcm16_to_float(raw), self.SAMPLE_RATE, self.name, time.perf_counter() - t0)


class EspeakTTS:
    name = "espeak"

    def __init__(self, config: VoiceConfig):
        if not shutil.which("espeak-ng"):
            raise RuntimeError("espeak-ng is not installed (dnf/apt install espeak-ng)")
        self.voice = config.voice or ESPEAK_DEFAULT_VOICE
        self.words_per_minute = int(160 * config.speed)

    def synthesize(self, text: str) -> SpokenAudio:
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "out.wav"
            subprocess.run(["espeak-ng", "-v", self.voice, "-s", str(self.words_per_minute),
                            "-w", str(wav_path), text], check=True, capture_output=True)
            with wave.open(str(wav_path), "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
        return SpokenAudio(pcm16_to_float(frames), rate, self.name, time.perf_counter() - t0)


def _piper(config: VoiceConfig):
    from .piper_tts import PiperTTS          # imported lazily: piper-tts is optional
    return PiperTTS(config)


BACKENDS = {"elevenlabs": ElevenLabsTTS, "openai": OpenAITTS, "piper": _piper, "espeak": EspeakTTS}


def make_tts(config: VoiceConfig | None = None) -> TextToSpeech:
    config = config or VoiceConfig()
    if config.backend != "auto":
        if config.backend not in BACKENDS:
            raise ValueError(f"Unknown TTS backend {config.backend!r}; choose from {', '.join(BACKENDS)}")
        return BACKENDS[config.backend](config)
    errors = []
    for name in ("elevenlabs", "openai", "piper", "espeak"):
        try:
            return BACKENDS[name](config)
        except RuntimeError as e:
            errors.append(f"{name}: {e.args[0].splitlines()[0] if e.args else e}")
    raise RuntimeError("No TTS backend available. " + "; ".join(errors))
