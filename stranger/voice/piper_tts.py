"""Offline text-to-speech with Piper (https://github.com/OHF-Voice/piper1-gpl).

    pip install piper-tts
    python -m piper.download_voices en_GB-alan-medium --download-dir ~/.local/share/stranger/piper-voices

A voice is a pair of files, `<name>.onnx` and `<name>.onnx.json`. `PiperTTS`
finds them by voice name in the voices directory (or takes a full path to
the .onnx file) and returns `SpokenAudio` like the other backends, so the
same `AudioPlayer` plays it.
"""

from __future__ import annotations

import os
import time
import wave
from pathlib import Path

import numpy as np

from .tts import SpokenAudio, VoiceConfig, pcm16_to_float

DEFAULT_PIPER_VOICE = "en_GB-alan-medium"
DEFAULT_VOICES_DIR = Path(os.environ.get("STRANGER_PIPER_VOICES",
                                         Path.home() / ".local" / "share" / "stranger" / "piper-voices"))
INSTALL_HELP = """Piper voice not found. To install one:
  pip install piper-tts
  python -m piper.download_voices {voice} --download-dir {voices_dir}
Other voices: python -m piper.download_voices --help  (names like en_US-lessac-medium, en_GB-alan-medium)"""


def resolve_voice(voice: str | Path | None, voices_dir: Path = DEFAULT_VOICES_DIR) -> Path:
    """Return the path to the .onnx model for a voice name or path, or raise with install help."""
    name = str(voice or DEFAULT_PIPER_VOICE)
    candidate = Path(name).expanduser()
    if candidate.suffix == ".onnx" and candidate.is_file():
        return candidate
    for folder in (voices_dir, Path.cwd()):
        model = folder / f"{name}.onnx"
        if model.is_file() and model.with_suffix(".onnx.json").is_file():
            return model
    raise RuntimeError(INSTALL_HELP.format(voice=name, voices_dir=voices_dir))


def list_voices(voices_dir: Path = DEFAULT_VOICES_DIR) -> list[str]:
    if not voices_dir.is_dir():
        return []
    return sorted(p.name[:-5] for p in voices_dir.glob("*.onnx") if p.with_suffix(".onnx.json").is_file())


class PiperTTS:
    name = "piper"

    def __init__(self, config: VoiceConfig | None = None, voices_dir: Path = DEFAULT_VOICES_DIR):
        config = config or VoiceConfig()
        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as e:
            raise RuntimeError("piper-tts is not installed: pip install piper-tts") from e
        self.model_path = resolve_voice(config.voice, voices_dir)
        self.voice = self.model_path.name[:-5]
        self._voice = PiperVoice.load(self.model_path)
        self.sample_rate = int(self._voice.config.sample_rate)
        # length_scale > 1 slows speech down; speed 1.2 -> length_scale 0.83
        self._syn = SynthesisConfig(length_scale=1.0 / config.speed if config.speed else None)

    def synthesize(self, text: str) -> SpokenAudio:
        t0 = time.perf_counter()
        chunks = [chunk.audio_int16_bytes for chunk in self._voice.synthesize(text, self._syn)]
        samples = pcm16_to_float(b"".join(chunks)) if chunks else np.zeros(0, dtype=np.float32)
        return SpokenAudio(samples, self.sample_rate, self.name, time.perf_counter() - t0)

    def synthesize_to_wav(self, text: str, path: str | Path) -> Path:
        """Write a 16-bit mono WAV file and return its path (used by the test and for caching)."""
        path = Path(path)
        with wave.open(str(path), "wb") as wf:
            self._voice.synthesize_wav(text, wf, self._syn)
        return path
